import sys

sys.path.insert(1, '../')
import argparse
import json
import os
import shutil
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from alphadock import residue_constants
from multimer import config_multimer, load_param_multimer, modules_multimer, test_multimer
from multimer.preprocess import find_mask_groups
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset
from wandb import Molecule

import warnings
warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")

class MultimerDataset(Dataset):
    def __init__(self, json_data, preprocessed_data_dir):
        self.data = json_data
        self.preprocessed_data_dir = preprocessed_data_dir

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_id = sample['sample_id']
        features = dict(np.load(f'{self.preprocessed_data_dir}/{sample_id}.npz'))
        features = {k: torch.tensor(v) for k, v in features.items()}
        meta_info = {
            'sample_id': sample_id,
            'seed': sample['seed'],
            'id': idx
        }

        return meta_info, features

    def __len__(self):
        return len(self.data)

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
            learning_rate=0.0001,
            output_data_path=None,
            hparams=None,
    ):
        super(TrainableFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.config_multimer['model']['embeddings_and_evoformer']['evoformer_num_block'] += hparams['evoformer_num_block']
        self.model = modules_multimer.DockerIteration(self.config_multimer, huber_delta=hparams['huber_delta'])
        load_param_multimer.import_jax_weights_(self.model, model_weights_path)
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.preprocessed_data_dir = preprocessed_data_dir
        self.n_layers_in_lr_group = n_layers_in_lr_group
        self.test_mode_name = test_mode_name
        self.crop_size = crop_size
        self.learning_rate = learning_rate
        self.output_data_path = output_data_path
        self.save_hyperparameters(hparams)

    def train_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.train_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=True, drop_last=Truem)

    def val_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        mul_dataset = MultimerDataset(
            json_data=self.val_data,
            preprocessed_data_dir=self.preprocessed_data_dir,
        )
        return torch.utils.data.DataLoader(mul_dataset, batch_size=self.batch_size, shuffle=False)

    def forward(self, batch):
        return self.model(batch, is_eval_mode=self.trainer.evaluating)
   
    def training_step(self, batch, _):
        meta_info, features = batch
        output, loss_items = self.forward(features)
        for k, v in loss_items.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss_items['loss']

    def validation_step(self, batch, _):
        meta_info, features = batch
        seed = meta_info['seed'].item()
        torch.manual_seed(seed)
        output, loss_items = self.forward(features)
        sample_name = meta_info['sample_id'][0]
        # self._get_predicted_structure(features, output, sample_name)

        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return {'sample_name': sample_name, **loss_items}

    def test_step(self, batch, _):
        meta_info, features = batch
        seed = meta_info['seed'].item()
        torch.manual_seed(seed)
        sample_name = meta_info['sample_id'][0]

        output, loss_items = self.forward(features)

        self._get_predicted_structure(features, output, sample_name, seed, mode=self.test_mode_name)
        self._get_true_structure(features, output, sample_name, seed)
        self._get_masked_true_structure(features, sample_name, seed)

        for k, v in loss_items.items():
            self.log(f'test_{k}', v, on_step=False, on_epoch=True, logger=True, sync_dist=True)


        metrics = {
            'sample_name': sample_name,
            'seed': seed,
            **{
                key: float(value.item()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in loss_items.items()
            }
        }

        with open(f"{self.output_data_path}/json_metrics/{sample_name}_{seed:02d}_rank_{self.global_rank}.json", "w") as outfile:
            outfile.write(json.dumps(metrics, indent=4))

        return {'sample_name': sample_name, **loss_items}

    def get_structure_state(self, batch, _):
        """
        A function for compression, is not used now.
        """
        meta_info, features = batch
        seed = meta_info['seed'].item()
        torch.manual_seed(seed)
        sample_name = meta_info['sample_id'][0]

        output, loss_items, msa_activations, pair_activations = self.forward(batch)

        filename = f"{self.output_data_path}/{sample_name}/{batch['seed'].item()}"

        batch_data = {k: v.cpu().numpy() for k, v in batch.items()}
        batch_data['msa_activations'] = msa_activations.cpu().numpy()
        batch_data['pair_activations'] = pair_activations.cpu().numpy()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **batch_data)


    def _get_predicted_structure(self, batch, output, sample_name, seed, mode='val'):
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

        filename = f"{self.output_data_path}/structures/{sample_name}/{mode}_s_{seed:02d}_r_{self.global_rank}.pdb"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(pdb_out)

        return filename

    def _get_true_structure(self, batch, output, sample_name, seed):
        # a temp solution for b_factors
        b_factors = np.ones((len(batch['aatype'][0]), residue_constants.atom_type_num)) * 100.0
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            batch['all_atom_positions'][0].cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            output['final_atom_mask'].cpu().numpy(), b_factors
        )

        filename = f"{self.output_data_path}/structures/{sample_name}/true_s_{seed:02d}_r_{self.global_rank}.pdb"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(pdb_out)

        return filename

    def _get_masked_true_structure(self, batch, sample_name, seed):
        # a temp solution for b_factors
        b_factors = np.ones((len(batch['aatype'][0]), residue_constants.atom_type_num)) * 100.0
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            batch['all_atom_positions'][0].cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            batch['all_atom_mask'][0].cpu().numpy(), b_factors,
            batch['loss_mask'][0].cpu().numpy()
        )

        filename = f"{self.output_data_path}/structures/{sample_name}/masked_s_{seed:02d}_r_{self.global_rank}.pdb"
        with open(filename, 'w') as f:
            f.write(pdb_out)

        return filename

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.learning_rate,
            max_lr=self.learning_rate*10,
            step_size_up=100,
            mode="triangular",
            gamma=0.85,
            cycle_momentum=False,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


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
        "--test_mode_name", type=str, default="test",
        help="Just a prefix for test table and structures like: test, validation, baseline"
    )
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--trainer_dir_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--train_json_path", type=str, default=None)
    parser.add_argument("--val_json_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--output_data_path", type=str, default=None)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--n_layers_in_lr_group", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--evoformer_num_block", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--step", type=str, default='train')
    parser.add_argument("--hyperparams_seed", type=int, default=None)
    args = parser.parse_args()

    if args.step == 'search':
        # Mainly for compression
        random.seed(args.hyperparams_seed)

        search_config = {
            'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
            'accumulate_grad_batches': {'values': [1, 3, 6, 10]},
            # the following are for compression
            'huber_delta': {'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
            'evoformer_num_block': {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        }

        config = {}

        for key, value in search_config.items():
            if 'values' in value:
                config[key] = random.choice(value['values'])

        print(config)

    else:
        config = {
            'learning_rate': args.learning_rate,
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'evoformer_num_block': args.evoformer_num_block,
            'huber_delta': 0.2,
        }

    train_data = json.load(open(args.train_json_path))
    val_data = json.load(open(args.val_json_path))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_checkpoint_path,
        save_top_k=5,
        mode='min',
        monitor='val_structure_loss',
        filename='{step:03d}-{val_structure_loss:.2f}',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    model = TrainableFolding(
        config_multimer=config_multimer,
        train_data=train_data,
        val_data=val_data,
        batch_size=1,
        preprocessed_data_dir=args.preprocessed_data_dir,
        model_weights_path=args.model_weights_path,
        n_layers_in_lr_group=args.n_layers_in_lr_group,
        test_mode_name=args.test_mode_name,
        crop_size=args.crop_size,
        learning_rate=config['learning_rate'],
        output_data_path=args.output_data_path,
        hparams=config,
    )

    logger = WandbLogger(
        save_dir=args.wandb_output_dir,
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume='True',
        offline=args.wandb_offline,
        entity='abc_sbu',
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        # TODO: fix the setup to enable this
        strategy="deepspeed_stage_1",
        # strategy="ddp",
        num_sanity_val_steps=0,
    )


    # print([p[0] for p in model.named_modules()])
    # import sys
    # sys.exit()

    # ### TO FREEZE
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # # learnable_layers_names = ['model.InputEmbedder.msa_summarization', 'model.InputEmbedder.msa_norm']
    # learnable_layers_names = ['model.Evoformer', 'model.msa_scale', 'model.pair_scale']
    # learnable_layers = [p[1] for p in model.named_modules() if p[0] in learnable_layers_names]
    #
    # for layer in learnable_layers:
    #     for param in layer.parameters():
    #         param.requires_grad = True

    if (args.step == 'train') or (args.step == 'search'):
        trainer.fit(model, ckpt_path=args.resume_from_ckpt)

        if trainer.global_rank == 0:
            best_model_path = checkpoint_callback.best_model_path
            destination_path = f"{args.model_checkpoint_path}/best.ckpt"
            shutil.copytree(best_model_path, destination_path)

    elif args.step == 'test':
        os.makedirs(f'{args.output_data_path}/json_metrics', exist_ok=True)

        trainer.test(model, ckpt_path=args.resume_from_ckpt)

        folder_path = f"{args.output_data_path}/json_metrics/"
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        list_of_dicts = [json.load(open(os.path.join(folder_path, file))) for file in json_files]
        df = pd.DataFrame(list_of_dicts)

        df.to_csv(f'{args.output_data_path}/metrics.csv', index=False)

    else:
        print('Select "train" or "test" option for parameter "step"')
