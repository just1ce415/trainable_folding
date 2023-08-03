import sys

sys.path.insert(1, '../')

import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from multimer import config_multimer, modules_multimer
from multimer.loss_multimer import softmax_cross_entropy
from test_multimer import compute_plddt


class SinglePredictedLddtDataset(Dataset):
    def __init__(self, folder_path):
        self.data_list = self.load_data_from_folder(folder_path)

    def load_data_from_folder(self, folder_path):
        data_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.npz'):
                file_path = os.path.join(folder_path, filename)

                with np.load(file_path) as data:
                    data = {k: torch.tensor(v) for k, v in dict(data).items()}
                    data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

def lddt_loss(logits, batch, score, config):
    all_atom_mask = batch['all_atom_mask']
    no_bins = config['predicted_lddt']['num_bins']
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    errors = torch.unsqueeze(errors, dim=-1)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
            1e-10 + torch.sum(all_atom_mask, dim=-1)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)
    pred_lddt = compute_plddt(logits)

    # Calculate MAE
    mae = torch.abs(pred_lddt - score*100).float().mean()

    return loss, mae


class TrainConfidence(pl.LightningModule):
    def __init__(
            self,
            config_multimer,
            hparams=None,
            train_data_folder=None,
            val_data_folder=None,
    ):
        super(TrainConfidence, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.hyperparams = hparams
        self.save_hyperparameters(hparams)
        self.train_data_folder = train_data_folder
        self.val_data_folder = val_data_folder
        self.model = modules_multimer.PredictedLddt(config_multimer.config_multimer, hparams['num_layers'])


    def train_dataloader(self):
        dataset = SinglePredictedLddtDataset(self.train_data_folder)
        return DataLoader(dataset, batch_size=1, shuffle=True)

    def val_dataloader(self):
        dataset = SinglePredictedLddtDataset(self.val_data_folder)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        targets = batch['lddt'].float()
        loss, mae = lddt_loss(logits, batch, targets, self.config_multimer['model']['heads'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        targets = batch['lddt'].float()
        loss, mae = lddt_loss(logits, batch, targets, self.config_multimer['model']['heads'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mae': mae}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hyperparams['learning_rate'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--step", type=str, default='train')
    parser.add_argument("--hyperparams_seed", type=int, default=0)
    parser.add_argument("--train_folder_path", type=str, default=None)
    parser.add_argument("--val_folder_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--init_model_weights_path", type=str, default=None)
    parser.add_argument("--plddt_model_weights_path", type=str, default=None)
    parser.add_argument("--new_model_weights_path", type=str, default=None)

    args = parser.parse_args()

    if args.step == 'update_model':
        # load model and remove predicted lddt weights
        init_model = torch.load(args.init_model_weights_path, map_location='cpu')
        init_model = {k: v for k, v in init_model.items() if 'PredictedLddt' not in k}

        # load plddt model and add weights to init model
        plddt_model = torch.load(args.plddt_model_weights_path, map_location='cpu')
        for k, v in plddt_model.items():
            init_model[k] = v

        #save new model
        torch.save(init_model, args.new_model_weights_path)

        import sys
        sys.exit()


    seed = args.hyperparams_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    config = {
        'learning_rate': args.learning_rate,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'num_layers': args.num_layers,
    }
    if args.step == 'search':
        search_config = {
            'learning_rate': {'values': [0.00001, 0.000001, 0.0000001]},
            'accumulate_grad_batches': {'values': [16, 64, 128, 256]},
            'num_layers': {'values': list(range(1, 6))},

        }

        config = {}

        for key, value in search_config.items():
            if 'values' in value:
                config[key] = random.choice(value['values'])

    print(config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.model_checkpoint_path,
        monitor='val_mae',
        mode='min',
        filename=f'best_model_pl',
        save_top_k=1
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    model = TrainConfidence(
        config_multimer=config_multimer,
        hparams=config,
        train_data_folder=args.train_folder_path,
        val_data_folder=args.val_folder_path,
    )

    logger = WandbLogger(
        save_dir=args.wandb_output_dir,
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume='False',
        offline=args.wandb_offline,
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="deepspeed_stage_1",
        num_sanity_val_steps=0,
    )

    if (args.step == 'train') or (args.step == 'search'):
        trainer.fit(model)
        best_model_path = f'{checkpoint_callback.best_model_path}/checkpoint/mp_rank_00_model_states.pt'
        checkpoint = torch.load(best_model_path, map_location='cpu')['module']
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('_forward_module.model', 'model.PredictedLddt')
            new_state_dict[new_key] = value
        torch.save(new_state_dict, f'{args.model_checkpoint_path}/plddt_weights.pt')
