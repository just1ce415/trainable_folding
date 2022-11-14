import sys
sys.path.insert(1, '../')
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from Bio import PDB
from multimer import mmcif_parsing, pipeline_multimer, feature_processing
import os
from multimer import msa_pairing, modules_multimer, config_multimer, load_param_multimer, pdb_to_template
from torch import optim
import random
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.optim.lr_scheduler import ReduceLROnPlateau
from multimer.lr_schedulers import AlphaFoldLRScheduler

def process_unmerged_features(all_chain_features):
  """Postprocessing stage for per-chain features before merging."""
  num_chains = len(all_chain_features)
  for chain_features in all_chain_features.values():
    # Convert deletion matrices to float.
    chain_features['deletion_matrix'] = np.asarray(
        chain_features.pop('deletion_matrix_int'), dtype=np.float32)
    if 'deletion_matrix_int_all_seq' in chain_features:
      chain_features['deletion_matrix_all_seq'] = np.asarray(
          chain_features.pop('deletion_matrix_int_all_seq'), dtype=np.float32)

    chain_features['deletion_mean'] = np.mean(
        chain_features['deletion_matrix'], axis=0)

    # Add assembly_num_chains.
    chain_features['assembly_num_chains'] = np.asarray(num_chains)

  # Add entity_mask.
  for chain_features in all_chain_features.values():
    chain_features['entity_mask'] = (
        chain_features['entity_id'] != 0).astype(np.int32)

def pair_and_merge(all_chain_features, is_homomer):
  """Runs processing on features to augment, pair and merge.

  Args:
    all_chain_features: A MutableMap of dictionaries of features for each chain.

  Returns:
    A dictionary of features.
  """

  process_unmerged_features(all_chain_features)

  np_chains_list = list(all_chain_features.values())

  pair_msa_sequences = not is_homomer

  if pair_msa_sequences:
    np_chains_list = msa_pairing.create_paired_features(
        chains=np_chains_list)
    np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
  np_chains_list = feature_processing.crop_chains(
      np_chains_list,
      msa_crop_size=feature_processing.MSA_CROP_SIZE,
      pair_msa_sequences=pair_msa_sequences,
      max_templates=feature_processing.MAX_TEMPLATES)
  np_example = msa_pairing.merge_chain_features(
      np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences,
      max_templates=feature_processing.MAX_TEMPLATES)
  np_example = feature_processing.process_final(np_example)
  return np_example


def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: str):
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        pipeline_multimer.make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = pipeline_multimer._get_atom_positions(
        mmcif_object, chain_id, max_ca_ca_distance=15000.0
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats

def process_single_chain(mmcif_object, chain_id, a3m_file, is_homomer_or_monomer, hhr_file=None):
    mmcif_feat = make_mmcif_features(mmcif_object, chain_id)
    chain_feat = mmcif_feat
    with open(a3m_file, "r") as fp:
        msa = pipeline_multimer.parse_a3m(fp.read())
    msa_feat = pipeline_multimer.make_msa_features((msa,))
    chain_feat.update(msa_feat)
    if hhr_file is not None:
        with open (hhr_file) as f:
            hhr = f.read()
        pdb_temp = pipeline_multimer.get_template_hits(output_string=hhr)
        templates_result = pipeline_multimer.get_templates(query_sequence=mmcif_object.chain_to_seqres[chain_id], hits=pdb_temp)
        temp_feat = templates_result.features
        chain_feat.update(temp_feat)

    if not is_homomer_or_monomer:
        all_seq_features = pipeline_multimer.make_msa_features([msa])
        valid_feats = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int',
                        'msa_uniprot_accession_identifiers','msa_species_identifiers',)
        feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

        chain_feat.update(feats)
    return chain_feat

def process_single_chain_pdb(all_position, all_mask, renum_mask, resolution, description, sequence, a3m_file, is_homomer_or_monomer, hhr_file=None):
    pdb_feat = pdb_to_template.make_pdb_features(all_position, all_mask, renum_mask, sequence, description, resolution)
    chain_feat = pdb_feat
    with open(a3m_file, "r") as fp:
        msa = pipeline_multimer.parse_a3m(fp.read())
    msa_feat = pipeline_multimer.make_msa_features((msa,))
    chain_feat.update(msa_feat)
    if hhr_file is not None:
        with open (hhr_file) as f:
            hhr = f.read()
        pdb_temp = pipeline_multimer.get_template_hits(output_string=hhr)
        templates_result = pipeline_multimer.get_templates(query_sequence=sequence, hits=pdb_temp)
        temp_feat = templates_result.features
        chain_feat.update(temp_feat)

    if not is_homomer_or_monomer:
        all_seq_features = pipeline_multimer.make_msa_features([msa])
        valid_feats = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int',
                        'msa_uniprot_accession_identifiers','msa_species_identifiers',)
        feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

        chain_feat.update(feats)
    return chain_feat

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
    def __init__(self, json_data, pre_alignment_path, device):
        self.data = json_data
        self.pre_align = pre_alignment_path
        self.device = device
    def process(self, idx):
        single_dataset = self.data[idx]
        cif_path = single_dataset['cif_file']
        file_id = os.path.basename(cif_path)[:-4]
        chains = single_dataset['chains']
        resolution = single_dataset['resolution']
        #with open(cif_path, 'r') as f:
        #    mmcif_string = f.read()
        #mmcif_obj = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string).mmcif_object
        sequences = []
        #for c in mmcif_obj.chain_to_seqres.keys():
        #    sequences.append(mmcif_obj.chain_to_seqres[c])
        ############
        for chain in chains:
            sequences.append(single_dataset['sequences'][chain])
        #############
        is_homomer = len(set(sequences))==1
        all_chain_features={}
        for chain in chains:
            #################
            all_atom_positions, all_atom_mask, renum_mask = pdb_to_template.align_seq_pdb(single_dataset['renum_seq'][chain], single_dataset['cif_file'], chain)
            description = '_'.join([file_id, chain])
            sequence = single_dataset['sequences'][chain]
            #################
            a3m_file = os.path.join(self.pre_align, f'{file_id}_{chain}', 'mmseqs/aggregated.a3m')
            hhr_file = os.path.join(self.pre_align, f'{file_id}_{chain}', 'mmseqs/aggregated.hhr')
            chain_features = process_single_chain_pdb(all_atom_positions, all_atom_mask, renum_mask, resolution, description, sequence, a3m_file, is_homomer, hhr_file=hhr_file)
            chain_features = pipeline_multimer.convert_monomer_features(chain_features,
                                                chain_id=chain)
            all_chain_features[chain] = chain_features
        all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
        np_example = pair_and_merge(all_chain_features, is_homomer)
        np_example = pipeline_multimer.pad_msa(np_example, 512)
        np_example = crop_feature(np_example, 384)
        np_example = {k: torch.tensor(v, device=self.device) for k,v in np_example.items()}

        return np_example

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #try:
        exp = self.process(idx)
        #except:
        #    print('+++++', idx)
        #    return {}
        return exp

class TrainableFolding(pl.LightningModule):
    def __init__(self, config_multimer, json_data, batch_size):
        super(TrainableFolding, self).__init__()
        self.config_multimer = config_multimer.config_multimer
        self.model = modules_multimer.DockerIteration(config_multimer.config_multimer)
        load_param_multimer.import_jax_weights_(self.model)
        self.json_data = json_data
        self.batch_size = batch_size
   
    def train_dataloader(self):
        mul_dataset = MultimerDataset(self.json_data, '/data1/thunguyen/antibody_MSAs/', self.device)
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
    

def main(args):
    with open('./ab_loop_multimer_test.json') as f:
        json_data = json.load(f)
    callbacks = []
    checkpoint_callback = ModelCheckpoint(dirpath='/data1/thunguyen/checkpoints', every_n_train_steps=5)
    callbacks.append(checkpoint_callback)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    model_module = TrainableFolding(config_multimer, json_data, 1)
    if(args.deepspeed_config_path is not None):
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

    loggers = []
    wdb_logger = WandbLogger(
        name='ab_ft_loop_train',
        save_dir=args.output_dir,
        id=args.wandb_id,
        resume='True',
        project='ab_ft_loop'
    )
    loggers.append(wdb_logger)

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        max_epochs=200,
        default_root_dir='/data1/thunguyen/',
        accumulate_grad_batches=10,
        log_every_n_steps=1
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module,
#        datamodule=data_module,
        ckpt_path=ckpt_path,
    )
    #trainer.save_checkpoint(
    #    os.path.join(trainer.logger.log_dir, "checkpoints", "final.ckpt")
    #)

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
        "output_dir", type=str,
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
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
