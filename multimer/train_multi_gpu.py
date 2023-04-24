import sys

sys.path.insert(1, '../')
import argparse
import json
import os
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


def crop_feature(features, crop_size):
    seq_len = features['seq_length']
    crop_size = min(seq_len, crop_size)
    start_crop = random.randint(0, seq_len - crop_size)
    feat_skip = {'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates', 'cluster_bias_mask', 'msa_activations', 'pair_activations', 'seed', 'sample_id'}
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
            file_path = f'{self.preprocessed_data_dir}/{file_id}/{single_dataset["seed"]}.npz'
            assert os.path.exists(file_path), f'File not found: {file_path}'
            self.processed_data[i] = file_id
            # if single_dataset['dataset'] == 'train':
            #     self.seeds[i] = None
            # else:
            if True:
                self.seeds[i] = single_dataset['seed']

    def process(self, idx):
        if self.seeds[idx] is not None:
            random.seed(self.seeds[idx])  # to stable the crop procedure for val and test
        n_groups = 0
        count = 0
        while n_groups < 2:
            np_example = dict(np.load(f'{self.preprocessed_data_dir}/{self.processed_data[idx]}/{self.seeds[idx]}.npz'))
            np_example = {k: v[0] for k, v in np_example.items()}
            # np_example = crop_feature(np_example, self.crop_size)
            n_groups = len(find_mask_groups(np_example['renum_mask']))
            count += 1
            if count == 100:
                raise RuntimeError("Did not find a good crop")

        # np_example['sample_id'] = idx
        # if self.seeds[idx] is not None:
        #     np_example['seed'] = self.seeds[idx]
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
            learning_rate=0.0001,
            output_data_path=None,
            hparams=None,
    ):
        super(TrainableFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.config_multimer['model']['embeddings_and_evoformer']['evoformer_num_block'] = hparams['evoformer_num_block']
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
   
    def training_step(self, batch, _):
        torch.manual_seed(batch['seed'])
        output, loss_items = self.forward(batch)
        for k, v in loss_items.items():
            self.log(k, v, on_step=True, on_epoch=False, logger=True)
        return loss_items['loss']

    def validation_step(self, batch, _):
        torch.manual_seed(batch['seed'])
        sample_id = batch['sample_id'].item()
        sample_name = self.val_sample_names[sample_id]
        output, loss_items = self.forward(batch)
        # self._get_predicted_structure(batch, output, sample_name)

        for k, v in loss_items.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, logger=True)

        # # for each sample
        # for k, v in loss_items.items():
        #     self.log(f'val_{k}_{sample_name}', v, on_step=True, on_epoch=False, logger=True)

        return {'sample_name': sample_name, 'mask_size': batch['renum_mask'][0].sum(), **loss_items}

    def test_step(self, batch, _):
        torch.manual_seed(batch['seed'])
        sample_id = batch['sample_id'].item()
        sample_name = self.val_sample_names[sample_id]

        output, loss_items, msa_activations, pair_activations = self.forward(batch)

        filename = f"{self.output_data_path}/{sample_name}/{batch['seed'].item()}"

        batch_data = {k: v.cpu().numpy() for k, v in batch.items()}
        batch_data['msa_activations'] = msa_activations.cpu().numpy()
        batch_data['pair_activations'] = pair_activations.cpu().numpy()

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **batch_data)

        # self._get_predicted_structure(batch, output, sample_name, mode=self.test_mode_name)
        # self._get_true_structure(batch, sample_name)
        # self._get_masked_true_structure(batch, sample_name)

        for k, v in loss_items.items():
            self.log(f'test_{k}', v, on_step=False, on_epoch=True, logger=True)

        seed = batch['seed'].item()
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

        return {'sample_name': sample_name, 'mask_size': batch['renum_mask'][0].sum(), **loss_items}

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
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--step", type=str, default='train')
    parser.add_argument("--hyperparams_seed", type=int, default=None)
    args = parser.parse_args()

    if args.step == 'search':
        random.seed(args.hyperparams_seed)

        search_config = {
            'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
            'accumulate_grad_batches': {'values': [1, 3, 6, 10]},
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
        }

    train_data = json.load(open(args.train_json_path))
    val_data = json.load(open(args.val_json_path))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_checkpoint_path,
        save_top_k=5,
        mode='min',
        monitor='val_structure_loop_loss',
        filename='{step:03d}-{val_structure_loop_loss:.2f}',
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
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        strategy="deepspeed_stage_2_offload",
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
