import sys
sys.path.insert(1, '../')
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from Bio import PDB
import os
from multimer import modules_multimer, config_multimer, load_param_multimer, pdb_to_template
import random
import argparse
import re
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
from multimer.lr_schedulers import AlphaFoldLRScheduler

import warnings
warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")


def crop_feature(features, crop_size):
    seq_len = features['seq_length']
    crop_size = min(seq_len, crop_size)
    start_crop = random.randint(0, seq_len - crop_size)
    feat_skip = {'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates', 'cluster_bias_mask'}
    feat_1 = {'aatype', 'residue_index', 'all_atom_positions', 'all_atom_mask', 'asym_id', 'sym_id', 'entity_id', 'deletion_mean', 'entity_mask', 'seq_mask', 'renum_mask'}
    for k in features.keys():
        if k not in feat_skip:
            if k in feat_1:
                features[k] = features[k][start_crop: start_crop+crop_size]
            else:
                features[k] = features[k][:, start_crop: start_crop+crop_size]
    features['seq_length'] = crop_size
    return features

class MultimerDataset(Dataset):
    def __init__(self, json_data, device, preprocessed_data_dir):
        self.data = json_data
        self.device = device  # TODO: not sure if we need it.
        self.preprocessed_data_dir = preprocessed_data_dir
        self._preprocess_all()

    def _preprocess_all(self):
        self.processed_data = {}
        for i, single_dataset in enumerate(self.data):
            cif_path = single_dataset['cif_file']
            file_id = os.path.basename(cif_path)[:-4]
            file_path = f'{self.preprocessed_data_dir}/{file_id}.npz'
            assert os.path.exists(file_path), f'File not found: {file_path}'
            self.processed_data[i] = file_id

    def process(self, idx):
        np_example = dict(np.load(f'{self.preprocessed_data_dir}/{self.processed_data[idx]}.npz'))
        np_example = crop_feature(np_example, 384)
        # TODO: not sure if we need to specify device.
        np_example = {k: torch.tensor(v, device=self.device) for k,v in np_example.items()}

        return np_example

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        exp = self.process(idx)
        return exp

class TrainableFolding(pl.LightningModule):
    def __init__(
            self,
            config_multimer,
            json_data,
            batch_size,
            preprocessed_data_dir,
            model_weights_path
    ):
        super(TrainableFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.model = modules_multimer.DockerIteration(config_multimer.config_multimer)
        load_param_multimer.import_jax_weights_(self.model, model_weights_path)
        self.json_data = json_data
        self.batch_size = batch_size
        self.preprocessed_data_dir = preprocessed_data_dir
   
    def train_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.json_data,
            device=self.device,
            preprocessed_data_dir=self.preprocessed_data_dir
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def forward(self, batch):
        batch['msa_profile'] = modules_multimer.make_msa_profile(batch)
        batch = modules_multimer.sample_msa(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_msa'])
        batch = modules_multimer.make_masked_msa(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['masked_msa'])
        (batch['cluster_profile'],batch['cluster_deletion_mean']) = modules_multimer.nearest_neighbor_clusters(batch)
        batch['msa_feat'] = modules_multimer.create_msa_feat(batch)
        batch['extra_msa_feat'], batch['extra_msa_mask'] = modules_multimer.create_extra_msa_feature(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_extra_msa'])
        batch['pseudo_beta'], batch['pseudo_beta_mask'] = modules_multimer.pseudo_beta_fn(batch['aatype'], batch['all_atom_positions'], batch['all_atom_mask'])
        return self.model(batch)
   
    def training_step(self, batch, batch_idx):
        output, loss, loss_items = self.forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True)
        for k, v in loss_items.items():
            self.log(k, v, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def configure_optimizers(self, 
        learning_rate: float = 5e-5,
        eps: float = 1e-6,
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            warmup_no_steps=300,
            start_decay_after_n_steps=1000,
            decay_every_n_steps=500
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AbFinetuneScheduler",
            },
        }


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument("--trainer_dir_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--json_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
