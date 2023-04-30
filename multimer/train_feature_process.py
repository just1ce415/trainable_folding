import sys

sys.path.insert(1, '../')
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import random
from alphadock import residue_constants
from multimer import (config_multimer, load_param_multimer, modules_multimer,
                      test_multimer)
from utils.crop_features import crop_feature
from utils.pdb_utils import reconstruct_residue, get_normalised_rmsd
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader, Dataset
import prody

import warnings

warnings.filterwarnings("ignore", "None of the inputs have requires_grad=True. Gradients will be None")

# Suppress output
prody.confProDy(verbosity='none')


class MultimerDataset(Dataset):
    def __init__(self, json_data, preprocessed_data_dir, seed_train=False):
        self.data = json_data
        self.preprocessed_data_dir = preprocessed_data_dir
        self.seed_train = seed_train

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_id = sample['sample_id']

        n_close_res = 0
        count = 0
        while n_close_res < 1:
            features = dict(np.load(f'{self.preprocessed_data_dir}/{sample_id}.npz'))
            if self.seed_train or (sample['is_val'] == 1):
                random.seed(sample['seed'])
            crop_size = sample['crop_size']
            if len(features['aatype']) > crop_size:
                features = crop_feature(features, crop_size)
            # Check if we need to find a good crop
            if 'is_inference' not in sample or sample['is_inference'] == 0:
                n_close_res = features['loss_mask'].sum()
                count += 1
                if count == 100:
                    raise RuntimeError("Did not find a good crop")
            else:
                break

        features = {k: torch.tensor(v) for k, v in features.items()}
        meta_info = {
            'sample_id': sample_id,
            'seed': sample['seed'],
            'id': idx
        }

        return meta_info, features

    def __len__(self):
        return len(self.data)


class NewResidueFolding(pl.LightningModule):
    def __init__(
            self,
            config_multimer,
            model_weights_path,
            learning_rate=0.0001,
            test_mode_name='test',
            output_data_path=None,
    ):
        super(NewResidueFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.model = modules_multimer.DockerIteration(config_multimer.config_multimer)
        load_param_multimer.import_jax_weights_(self.model, model_weights_path)
        self.learning_rate = learning_rate
        self.test_mode_name = test_mode_name
        self.output_data_path = output_data_path

    def forward(self, features):
        return self.model(features, is_eval_mode=self.trainer.evaluating)

    def training_step(self, batch):
        meta_info, features = batch
        output, loss_items = self.forward(features)
        for k, v in loss_items.items():
            self.log(f'train_{k}', v, on_step=True, on_epoch=True, logger=True)
        return loss_items['loss']

    def _get_predicted_structure(self, batch, output, sample_id, seed, mode='val'):
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

        filename = f"{self.output_data_path}/structures/{sample_id}/{mode}_s_{seed:02d}_r_{self.global_rank}.pdb"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(pdb_out)

        reconstruct_residue(filename, filename)

        return filename

    def _get_true_structure(self, batch, output, sample_id, seed):
        # a temp solution for b_factors
        b_factors = np.ones((len(batch['aatype'][0]), residue_constants.atom_type_num)) * 100.0
        pdb_out = test_multimer.protein_to_pdb(
            batch['aatype'][0].cpu().numpy(),
            batch['all_atom_positions'][0].cpu().numpy(),
            batch['residue_index'][0].cpu().numpy() + 1,
            batch['asym_id'][0].cpu().numpy(),
            output['final_atom_mask'].cpu().numpy(), b_factors
        )

        filename = f"{self.output_data_path}/structures/{sample_id}/true_s_{seed:02d}_r_{self.global_rank}.pdb"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(pdb_out)

        reconstruct_residue(filename, filename)

        return filename

    def _align_structures(self, batch, output, sample_id, seed, mode='val', save_pdb=False):
        pred_filename = self._get_predicted_structure(batch, output, sample_id, seed, mode=mode)
        true_filename = self._get_true_structure(batch, output, sample_id, seed)

        # Load the PDB files using the parsePDB function
        true = prody.parsePDB(true_filename)
        pred = prody.parsePDB(pred_filename)

        # Clean up pdbs:
        os.remove(pred_filename)
        os.remove(true_filename)

        # Align the two structures
        prody.superpose(true.select('calpha'), pred.select('calpha'))

        # Get NEW residue and calculate distance
        ca_true = true.select('resname NEW and name CA')[-1]
        ca_pred = pred.select('resname NEW and name CA')[-1]
        distance = prody.calcDistance(ca_true, ca_pred)

        res_true = true.select('resname NEW')
        res_pred = pred.select('resname NEW')
        rmsd = prody.calcRMSD(res_true, res_pred)

        normalised_rmsd = get_normalised_rmsd(
            res_true, res_pred,
            self.config_multimer['virtual_point']['scale'],
            self.config_multimer['virtual_point']['axis']
        )

        full_rmsd = prody.calcRMSD(true.select('resname REC'), pred.select('resname REC'))

        # Save aligned structures
        if save_pdb:
            new_pred_filename = f"{self.output_data_path}/structures/{sample_id}/{mode}_{rmsd:0.2f}_s_{seed:02d}.pdb"
            prody.writePDB(new_pred_filename, pred)
            if seed == 0:
                prody.writePDB(true_filename, true)

        return distance, rmsd, normalised_rmsd, full_rmsd

    def validation_step(self, batch, _):
        meta_info, features = batch
        seed = meta_info['seed'].item()
        sample_id = meta_info['sample_id'][0]
        torch.manual_seed(seed)
        output, loss_items = self.forward(features)
        distance, rmsd, normalised_rmsd, full_rmsd = self._align_structures(features, output, sample_id, seed, mode='val')

        loss_items['new_res_distance'] = distance
        loss_items['new_res_rmsd'] = rmsd
        loss_items['new_res_good_rmsd'] = (rmsd < 2.0) * 1.0
        loss_items['new_res_normalised_rmsd'] = normalised_rmsd
        loss_items['new_res_full_rmsd'] = full_rmsd

        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return {'sample_id': sample_id, **loss_items}

    def test_step(self, batch, batch_idx):
        meta_info, features = batch
        seed = meta_info['seed'].item()
        sample_id = meta_info['sample_id'][0]
        torch.manual_seed(seed)
        output, loss_items = self.forward(features)
        distance, rmsd, normalised_rmsd, full_rmsd = self._align_structures(
            features, output, sample_id, seed, mode=self.test_mode_name, save_pdb=True
        )
        loss_items['new_res_distance'] = distance
        loss_items['new_res_rmsd'] = rmsd
        loss_items['new_res_good_rmsd'] = (rmsd < 2.0) * 1.0
        loss_items['new_res_normalised_rmsd'] = normalised_rmsd
        loss_items['new_res_full_rmsd'] = full_rmsd

        metrics = {
            'sample_id': sample_id,
            'seed': seed,
            **{
                key: value.item() if isinstance(value, torch.Tensor) else value
                for key, value in loss_items.items()
            }
        }

        with open(f"{self.output_data_path}/json_metrics/{sample_id}_{seed:02d}.json", "w") as outfile:
            outfile.write(json.dumps(metrics, indent=4))

        return {'sample_id': sample_id, **loss_items}

    def predict_step(self, batch, _):
        meta_info, features = batch
        seed = meta_info['seed'].item()
        sample_id = meta_info['sample_id'][0]
        torch.manual_seed(seed)
        output, new_res_plddt = self.model(features, is_eval_mode=True, is_inference_mode=True)

        pred_filename = self._get_predicted_structure(features, output, sample_id, seed, mode=f'{new_res_plddt:0.4f}')

        metrics = {
            'sample_id': sample_id,
            'seed': seed,
            'new_res_plddt': new_res_plddt,
        }

        with open(f"{self.output_data_path}/json_metrics/{sample_id}_{seed:02d}.json", "w") as outfile:
            outfile.write(json.dumps(metrics, indent=4))

        return


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.learning_rate,
            max_lr=self.learning_rate * 10,
            step_size_up=10,
            mode="triangular",
            gamma=0.85,
            cycle_momentum=False,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_path', type=str)
    parser.add_argument('--val_json_path', type=str)
    parser.add_argument('--test_json_path', type=str)
    parser.add_argument('--preprocessed_data_dir', type=str)
    parser.add_argument("--model_weights_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--output_data_path", type=str, default=None)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--wandb_output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--step", type=str, default='train')
    parser.add_argument("--test_mode_name", type=str, default="test", )
    args = parser.parse_args()

    np.random.seed(13)
    torch.use_deterministic_algorithms(True)

    model = NewResidueFolding(
        config_multimer=config_multimer,
        model_weights_path=args.model_weights_path,
        learning_rate=0.00001,
        test_mode_name=args.test_mode_name,
        output_data_path=args.output_data_path,
    )

    # old_checkpoint = '/gpfs/alpine/bip215/proj-shared/eglukhov/new_residue/output/large_mol/2783482/checkpoints/stepstep=084-distanceval_new_res_distance=6.65.ckpt/global_step85/mp_rank_00_model_states.pt'
    # checkpoint = torch.load(old_checkpoint, map_location='cpu')['module']
    # new_weights = model.state_dict()
    # for k, v in checkpoint.items():
    #     new_weights[k.replace("_forward_module.", "")] = v

    # # UPDATE weights with trained ones
    # conf_checkpoint = torch.load('/projectnb2/sc3dm/eglukhov/new_residue/checkpoints/v6/conf/model.ckpt', map_location='cpu')['state_dict']
    # for k, v in conf_checkpoint.items():
    #     new_weights[f'model.PredictedLddtNewRes.{k}'] = v

    # model.load_state_dict(new_weights)

    # ### TO FREEZE
    # for param in model.parameters():
    #     param.requires_grad = False

    # learnable_layers_names = ['model.InputEmbedder.msa_summarization', 'model.InputEmbedder.msa_norm']
    # learnable_layers_names = ['model.PredictedLddtNewRes']
    # learnable_layers = [p[1] for p in model.named_modules() if p[0] in learnable_layers_names]

    # for layer in learnable_layers:
    #     for param in layer.parameters():
    #         param.requires_grad = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_checkpoint_path,
        save_top_k=5,
        mode='min',
        monitor='val_new_res_rmsd',
        filename='step{step:03d}-rmsd{val_new_res_rmsd:.2f}',
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = WandbLogger(
        save_dir=args.wandb_output_dir,
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume='True',
        offline=args.wandb_offline,
    )

    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            lr_monitor,
        ],
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.num_nodes,
        strategy="deepspeed_stage_1",
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=0,
    )

    if args.step == 'train':
        train_json_data = json.load(open(args.train_json_path))
        train_dataset = MultimerDataset(train_json_data, args.preprocessed_data_dir)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        val_json_data = json.load(open(args.val_json_path))
        val_dataset = MultimerDataset(val_json_data, args.preprocessed_data_dir)
        val_loader = DataLoader(val_dataset, batch_size=1)

        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_ckpt)

        # After training is finished, copy the best model to another path
        if trainer.global_rank == 0:
            best_model_path = checkpoint_callback.best_model_path
            destination_path = f"{args.model_checkpoint_path}/best.ckpt"
            shutil.copytree(best_model_path, destination_path)

    elif (args.step == 'test') or (args.step == 'predict'):
        json_data = json.load(open(args.test_json_path))
        dataset = MultimerDataset(json_data, args.preprocessed_data_dir)
        loader = DataLoader(dataset, batch_size=1)

        os.makedirs(f'{args.output_data_path}/json_metrics', exist_ok=True)

        if args.step == 'test':
            trainer.test(model, loader, ckpt_path=args.resume_from_ckpt)
        else:
            trainer.predict(model, loader, ckpt_path=args.resume_from_ckpt)

        folder_path = f"{args.output_data_path}/json_metrics/"
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
        list_of_dicts = [json.load(open(os.path.join(folder_path, file))) for file in json_files]
        df = pd.DataFrame(list_of_dicts)

        df.to_csv(f'{args.output_data_path}/metrics.csv', index=False)
    else:
        print('Select "train" or "test" option for parameter "step"')
