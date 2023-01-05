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
        output, loss_items = self.forward(batch)
        for k, v in loss_items.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, logger=True)
        return loss_items['loss']

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

    def validation_step(self, batch, batch_idx):
        sample_name = self.val_sample_names[batch_idx]
        output, loss_items = self.forward(batch)
        self._get_predicted_structure(batch, output, sample_name)

        # Individual scores
        self.log(f'val_plddt_{sample_name}', loss_items['new_res_plddt'], on_step=True, on_epoch=False, logger=True)
        self.log(f'val_lddt_{sample_name}', loss_items['new_res_lddt'], on_step=True, on_epoch=False, logger=True)

        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True)

        return {'sample_name': sample_name, **loss_items}


    def validation_epoch_end(self, validation_step_outputs):
        columns = [key for key in validation_step_outputs[0]]
        values = [output.values() for output in validation_step_outputs]
        wdb_logger.log_table(
                key=f'val_step_{self.global_step}',
                columns=columns,
                data=values,
            )

    def test_step(self, batch, batch_idx):
        sample_name = self.val_sample_names[batch_idx]
        output, loss_items = self.forward(batch)
        self._get_predicted_structure(batch, output, sample_name, mode='test')
        self._get_true_structure(batch, sample_name)

        for k, v in loss_items.items():
            self.log(f'test_{k}', v, on_step=False, on_epoch=True, logger=True)

        # for each sample
        for k, v in loss_items.items():
            self.log(f'test_{k}_{sample_name}', v, on_step=False, on_epoch=True, logger=True)

        return {'sample_name': sample_name, **loss_items}

    def test_epoch_end(self, test_step_outputs):
        columns = [key for key in test_step_outputs[0]]
        values = [output.values() for output in test_step_outputs]
        wdb_logger.log_table(
                key='test_metrics',
                columns=columns,
                data=values,
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.learning_rate,
            max_lr=self.learning_rate*10,
            step_size_up=10,
            mode="triangular2",
            gamma=0.85,
            cycle_momentum=False,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


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
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_checkpoint_path,
        save_top_k=5,
        mode='max',
        monitor='val_new_res_is_confident',
        filename='step{step:03d}-confidence{val_new_res_is_confident:.2f}'
    )
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
        logger=wdb_logger,
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

