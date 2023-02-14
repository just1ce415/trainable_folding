import sys

sys.path.insert(1, '../')
import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from alphadock import residue_constants
from multimer import config_multimer, load_param_multimer, modules_multimer, test_multimer
from multimer.lr_schedulers import AlphaFoldLRScheduler
from multimer.preprocess import find_mask_groups
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.plugins.training_type import DDPPlugin, DeepSpeedPlugin
from torch.utils.data import Dataset
from wandb import Molecule

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
    def __init__(self, json_data, preprocessed_data_dir):
        self.data = json_data
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
        np_example = {k: torch.tensor(v) for k, v in np_example.items()}

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
            train_data,
            val_data,
            batch_size,
            preprocessed_data_dir,
            model_weights_path,
            n_layers_in_lr_group
    ):
        super(TrainableFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.model = modules_multimer.DockerIteration(config_multimer.config_multimer)
        load_param_multimer.import_jax_weights_(self.model, model_weights_path)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.preprocessed_data_dir = preprocessed_data_dir
        self.n_layers_in_lr_group = n_layers_in_lr_group
   
    def train_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.train_data,
            preprocessed_data_dir=self.preprocessed_data_dir
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir
        )
        self.val_sample_names = mul_dataset.processed_data
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir
        )
        self.val_sample_names = mul_dataset.processed_data
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def forward(self, batch):
        return self.model(batch, is_eval_mode=self.trainer.evaluating)
   
    def training_step(self, batch, batch_idx):
        output, loss, loss_items = self.forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, logger=True)
        for k, v in loss_items.items():
            self.log(k, v, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample_name = self.val_sample_names[batch_idx]
        output, loss, loss_items = self.forward(batch)
        output['predicted_aligned_error']['asym_id'] = batch['asym_id'][0]
        confidences = test_multimer.get_confidence_metrics(output, True)

        plddt = confidences['plddt'].detach().cpu().numpy()
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )

        pdb_out = test_multimer.protein_to_pdb(batch['aatype'][0].cpu().numpy(),
                                               output['final_all_atom'].detach().cpu().numpy(),
                                               batch['residue_index'][0].cpu().numpy() + 1,
                                               batch['asym_id'][0].cpu().numpy(),
                                               output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0])

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        # new res is always the last, batch size is always 0
        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True)

        filename = f"{np.random.randint(1000000, 10000000)}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)

        wdb_logger.log_table(
            key=sample_name,
            columns=['id', 'pdb'],
            data=[[sample_name, Molecule(filename)]]
        )
        os.remove(filename)

        return sample_name

    def _get_predicted_structure(self, batch, output, sample_name, mode='val'):
        output['predicted_aligned_error']['asym_id'] = batch['asym_id'][0]
        confidences = test_multimer.get_confidence_metrics(output, True)

        plddt = confidences['plddt'].detach().cpu().numpy()
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            output['final_all_atom'].detach().cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0]
        )

        filename = f"{mode}_pred_{sample_name}_{np.random.randint(1000000, 10000000)}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)
        self.logger.experiment.log({f'{mode}_pred_{sample_name}': Molecule(filename)})
        os.remove(filename)

    def _get_true_structure(self, batch, sample_name):
        # a temp solution for b_factors
        b_factors = np.ones((len(batch['aatype'][0]), residue_constants.atom_type_num)) * 100.0
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            batch['all_atom_positions'][0].cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            batch['all_atom_mask'][0].cpu().numpy(), b_factors
        )

        filename = f"true_{sample_name}_{np.random.randint(1000000, 10000000)}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)
        self.logger.experiment.log({f'true_{sample_name}': Molecule(filename)})
        os.remove(filename)

    def _get_masked_true_structure(self, batch, sample_name):
        # a temp solution for b_factors
        b_factors = np.ones((len(batch['aatype'][0]), residue_constants.atom_type_num)) * 100.0
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            batch['all_atom_positions'][0].cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            batch['all_atom_mask'][0].cpu().numpy(), b_factors,
            batch['renum_mask'][0].cpu().numpy()
        )

        filename = f"masked_{sample_name}_{np.random.randint(1000000, 10000000)}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)
        self.logger.experiment.log({f'masked_{sample_name}': Molecule(filename)})
        os.remove(filename)

    def test_step(self, batch, batch_idx):
        sample_name = self.val_sample_names[batch_idx]
        output, loss, loss_items = self.forward(batch)
        self._get_predicted_structure(batch, output, sample_name, mode='baseline')
        self._get_true_structure(batch, sample_name)
        self._get_masked_true_structure(batch, sample_name)

        for k, v in loss_items.items():
            self.log(f'test_{k}', v, on_step=False, on_epoch=True, logger=True)

        # for each sample
        for k, v in loss_items.items():
            self.log(f'test_{k}_{sample_name}', v, on_step=False, on_epoch=True, logger=True)

        return {'sample_name': sample_name, 'mask_size': batch['renum_mask'][0].sum(), **loss_items}

    def test_epoch_end(self, test_step_outputs):
        columns = [key for key in test_step_outputs[0]]
        values = [output.values() for output in test_step_outputs]
        wdb_logger.log_table(
                key='test_metrics',
                columns=columns,
                data=values,
            )
    
    def configure_optimizers(self, 
        learning_rate: float = 5e-5,
        eps: float = 1e-6,
    ):
        # Ignored as long as a DeepSpeed optimizer is configured
        param_groups = defaultdict(list)
        for name, p in self.model.named_parameters():
            # split Evoformer layers into groups
            # more distant layers will have larger multiplier
            if self.n_layers_in_lr_group and re.match('Evoformer\.\d+', name):
                multiplier = int(name.split('.')[1]) // self.n_layers_in_lr_group + 1
            else:
                multiplier = 1
            param_groups[multiplier].append(p)
        optimizer = torch.optim.Adam(
            [
                {
                    'name': multiplier,
                    'multiplier': multiplier,
                    'params': params,
                    'lr': learning_rate,
                } for multiplier, params in param_groups.items()
            ],
            eps=eps
        )
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            max_lr=0.0001,
            warmup_no_steps=50,
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
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--trainer_dir_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--json_data_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--accum_grad_batches", type=int, default=16)
    parser.add_argument("--n_layers_in_lr_group", type=int, default=None)
    parser.add_argument("--val_size", type=float, default=0.05)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    SEED = 4

    with open(args.json_data_path) as f:
        json_data = json.load(f)

    n_train_samples = int(len(json_data) * (1 - args.val_size))
    random.Random(SEED).shuffle(json_data)
    train_data, val_data = json_data[:n_train_samples], json_data[n_train_samples:]
    callbacks = []
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_checkpoint_path, every_n_train_steps=1)
    callbacks.append(checkpoint_callback)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    model_module = TrainableFolding(
        config_multimer=config_multimer,
        train_data=train_data,
        val_data=val_data,
        batch_size=1,
        preprocessed_data_dir=args.preprocessed_data_dir,
        model_weights_path=args.model_weights_path,
        n_layers_in_lr_group=args.n_layers_in_lr_group,
    )
    if args.deepspeed_config_path is not None:
        if "SLURM_JOB_ID" in os.environ:
            cluster_environment = SLURMEnvironment()
        else:
            cluster_environment = None
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
            cluster_environment=cluster_environment,
        )
    elif args.gpus > 1 or args.num_nodes > 1:
        strategy = 'ddp'
    else:
        strategy = None

    wdb_logger = WandbLogger(
        name=args.wandb_name,
        save_dir=args.output_dir,
        id=args.wandb_id,
        resume='True',
        project='ab_ft_loop'
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks,
        logger=wdb_logger,
        max_epochs=2,
        default_root_dir=args.trainer_dir_path,
        accumulate_grad_batches=args.accum_grad_batches,
        log_every_n_steps=1,
        val_check_interval=0.1,
    )

    if args.resume_model_weights_only:
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module,
        ckpt_path=ckpt_path,
    )
