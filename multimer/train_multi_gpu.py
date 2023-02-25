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
    def __init__(self, json_data, preprocessed_data_dir, crop_size):
        self.data = json_data
        self.preprocessed_data_dir = preprocessed_data_dir
        self.crop_size = crop_size
        self._preprocess_all()

    def _preprocess_all(self):
        self.processed_data = {}
        self.seeds = {}
        for i, single_dataset in enumerate(self.data):
            file_id = single_dataset['sample_id']
            file_path = f'{self.preprocessed_data_dir}/{file_id}.npz'
            assert os.path.exists(file_path), f'File not found: {file_path}'
            self.processed_data[i] = file_id
            if single_dataset['dataset'] == 'train':
                self.seeds[i] = None
            else:
                self.seeds[i] = single_dataset['seed']

    def process(self, idx):
        if self.seeds[idx]:
            random.seed(self.seeds[idx])  # to stable the crop procedure for val and test
        n_groups = 0
        count = 0
        while n_groups < 2:
            np_example = dict(np.load(f'{self.preprocessed_data_dir}/{self.processed_data[idx]}.npz'))
            np_example = crop_feature(np_example, self.crop_size)
            n_groups = len(find_mask_groups(np_example['renum_mask']))
            count += 1
            if count == 100:
                raise RuntimeError("Did not find a good crop")

        if self.seeds[idx]:
            np_example['seed'] = self.seeds[idx]
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
            n_layers_in_lr_group,
            test_mode_name='test',
            crop_size=384,
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
        self.test_mode_name = test_mode_name
        self.crop_size = crop_size
   
    def train_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.train_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
            crop_size=self.crop_size,
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
            crop_size=self.crop_size,
        )
        self.val_sample_names = mul_dataset.processed_data
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
            crop_size=self.crop_size,
        )
        self.val_sample_names = mul_dataset.processed_data
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def forward(self, batch):
        return self.model(batch, is_eval_mode=self.trainer.evaluating)
   
    def training_step(self, batch, batch_idx):
        output, loss_items = self.forward(batch)
        for k, v in loss_items.items():
            self.log(k, v, on_step=True, on_epoch=False, logger=True)
        return loss_items['loss']

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(batch['seed'])
        sample_name = self.val_sample_names[batch_idx]
        output, loss_items = self.forward(batch)
        self._get_predicted_structure(batch, output, sample_name)

        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True)

        # for each sample
        for k, v in loss_items.items():
            self.log(f'val_{k}_{sample_name}', v, on_step=True, on_epoch=False, logger=True)

        return {'sample_name': sample_name, 'mask_size': batch['renum_mask'][0].sum(), **loss_items}

    def test_step(self, batch, batch_idx):
        torch.manual_seed(batch['seed'])
        sample_name = self.val_sample_names[batch_idx]
        output, loss_items = self.forward(batch)
        self._get_predicted_structure(batch, output, sample_name, mode=self.test_mode_name)
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
                key=f'{self.test_mode_name}_metrics',
                columns=columns,
                data=values,
            )

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
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument(
        "--test_mode_name", type=str, default="test",
        help="Just a prefix for test table and structures like: test, validation, baseline"
    )
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--trainer_dir_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--train_json_path", type=str, default=None)
    parser.add_argument("--val_json_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--accum_grad_batches", type=int, default=16)
    parser.add_argument("--n_layers_in_lr_group", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--step", type=str, default='train')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    train_data = json.load(open(args.train_json_path))
    val_data = json.load(open(args.val_json_path))

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
        test_mode_name=args.test_mode_name,
        crop_size=args.crop_size,
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
        project=args.project,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks,
        logger=wdb_logger,
        max_epochs=3,
        default_root_dir=args.trainer_dir_path,
        accumulate_grad_batches=args.accum_grad_batches,
        log_every_n_steps=1,
        val_check_interval=0.2,
        accelerator='gpu',
        num_sanity_val_steps=len(val_data),
    )

    if args.step == 'train':
        trainer.fit(model_module, ckpt_path=args.resume_from_ckpt)
    elif args.step == 'test':
        trainer.test(model_module, ckpt_path=args.resume_from_ckpt)
    else:
        print('Select "train" or "test" option for parameter "step"')
