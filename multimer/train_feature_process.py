import sys

sys.path.insert(1, '../')
import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from alphadock import residue_constants
from multimer import (config_multimer, load_param_multimer, modules_multimer,
                      test_multimer)
from preprocess import crop_feature
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from wandb import Molecule


class MultimerDataset(Dataset):
    def __init__(self, json_data, preprocessed_data_dir):
        self.data = json_data
        self.preprocessed_data_dir = preprocessed_data_dir
        self._preprocess_all()

    def _preprocess_all(self):
        self.processed_data = {}
        i = 0
        for single_dataset in self.data:
            sdf_path = single_dataset['sdf']
            sample_id = os.path.basename(sdf_path)[:-4]
            file_path = f'{self.preprocessed_data_dir}/{sample_id}.npz'
            assert os.path.exists(file_path), f'File not found: {file_path}'
            # check that we have at least one res close to a new one
            n_close_res = np.load(f'{self.preprocessed_data_dir}/{sample_id}.npz')['loss_mask'].sum()
            if n_close_res <= 1:
                continue
            self.processed_data[i] = sample_id
            i += 1

    def process(self, idx):
        n_close_res = 0
        count = 0
        while n_close_res <= 1:
            np_example = dict(np.load(f'{self.preprocessed_data_dir}/{self.processed_data[idx]}.npz'))
            np_example = crop_feature(np_example, 384)
            n_close_res = np_example['loss_mask'].sum()
            count += 1
            if count == 100:
                raise RuntimeError("Did not find a good crop")
        np_example = {k: torch.tensor(v) for k, v in np_example.items()}

        return np_example

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.process(idx)


class NewResidueFolding(pl.LightningModule):
    def __init__(
            self,
            config_multimer,
            model_weights_path,
            val_sample_names,
            learning_rate=0.0001,
    ):
        super(NewResidueFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.model = modules_multimer.DockerIteration(config_multimer.config_multimer)
        load_param_multimer.import_jax_weights_(self.model, model_weights_path)
        self.val_sample_names = val_sample_names
        self.learning_rate = learning_rate

    def forward(self, batch):
        batch['msa_profile'] = modules_multimer.make_msa_profile(batch)
        batch = modules_multimer.sample_msa(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_msa'])
        batch = modules_multimer.make_masked_msa(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['masked_msa'])
        (batch['cluster_profile'], batch['cluster_deletion_mean']) = modules_multimer.nearest_neighbor_clusters(batch)
        batch['msa_feat'] = modules_multimer.create_msa_feat(batch)
        batch['extra_msa_feat'], batch['extra_msa_mask'] = modules_multimer.create_extra_msa_feature(batch, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_extra_msa'])
        batch['pseudo_beta'], batch['pseudo_beta_mask'] = modules_multimer.pseudo_beta_fn(batch['aatype'], batch['all_atom_positions'], batch['all_atom_mask'])

        return self.model(batch)

    def training_step(self, batch):
        output, loss, loss_items = self.forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        for k, v in loss_items.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sample_name = self.val_sample_names[batch_idx]
        output, loss, loss_items = self.forward(batch)
        output['predicted_aligned_error']['asym_id'] = batch['asym_id'][0]
        confidences = test_multimer.get_confidence_metrics(output, True)
        # out_converted = {}
        # for k, v in confidences.items():
        #     if (k != "plddt" and k != "aligned_confidence_probs" and k != "predicted_aligned_error"):
        #         out_converted[k] = confidences[k].detach().cpu().numpy().tolist()
        # out_json = out_converted

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
        self.log('val_new_res_plddt', plddt[0, -1], on_step=False, on_epoch=True, logger=True)
        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True)

        filename = f"{np.random.randint(1000000, 10000000)}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)

        wdb_logger.log_table(
            key=sample_name,
            columns=['id', 'new_res_lddt', 'pdb'],
            data=[[sample_name, loss_items['new_res_lddt'], Molecule(filename)]]
        )
        os.remove(filename)

        return sample_name

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--val_json_path', type=str)
    parser.add_argument('--preprocessed_data_dir', type=str)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_logger_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(13)
    torch.use_deterministic_algorithms(True, warn_only=True)

    json_data = json.load(open(args.json_path))
    train_dataset = MultimerDataset(json_data, args.preprocessed_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    if args.val_json_path:
        val_json_data = json.load(open(args.val_json_path))
        val_dataset = MultimerDataset(val_json_data, args.preprocessed_data_dir)
        val_loader = DataLoader(val_dataset, batch_size=1)
    else:
        val_loader = None

    model = NewResidueFolding(
        config_multimer=config_multimer,
        model_weights_path=args.model_weights_path,
        val_sample_names=val_dataset.processed_data
    )
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_checkpoint_path, every_n_train_steps=5)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    wdb_logger = WandbLogger(
        name=args.wandb_name,
        save_dir=args.wandb_logger_dir,
        id=args.wandb_id,
        resume=True,
        project='new_residue'
    )


    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            lr_monitor,
        ],
        logger=[wdb_logger],
        log_every_n_steps=1,
        resume_from_checkpoint=args.resume_from_checkpoint,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=10,
        accelerator="gpu",
        devices=args.gpus,
        strategy="deepspeed_stage_1",
        val_check_interval=200,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)

