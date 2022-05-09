import traceback
import random
import itertools
import numpy as np
import prody
from collections import defaultdict, Counter
from path import Path
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from functools import partial
import torch
import sys
import time
import datetime
#import horovod.torch as hvd

from alphadock.config import DATA_DIR, DTYPE_FLOAT
from alphadock import utils
from alphadock import features_summit
from alphadock import residue_constants


class DockingDataset(Dataset):
    def __init__(
            self,
            data,
            config_data,
            dataset_dir='.',
            seed=123456,
            shuffle=False
    ):
        self.dataset_dir = Path(dataset_dir).abspath()
        self.config = config_data
        self.data = data

        self.rng = np.random.default_rng(seed)

        if shuffle:
            self.rng.shuffle(self.data)

        #if sample_to_size is not None:
        #    probs = np.array([1. / x['seqclus_size'] for x in self.data])
        #    probs /= probs.sum()
        #    self.data = self.rng.choice(self.data, size=min(sample_to_size, len(self.data)), replace=False, p=probs)
        #    #self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def make_features(self, sequence, a3m_files, cif_file=None, asym_ids=None):
        if self.config['crop_size'] is not None:
            crop_start = self.rng.integers(0, max(1, len(sequence) - self.config['crop_size']))
            crop_range = [crop_start, crop_start + self.config['crop_size']]
        else:
            crop_range = None

        out_dict = {}
        out_dict['target'] = features_summit.target_sequence_featurize(
            sequence,
            crop_range=crop_range,
            af_compatible=self.config['target_af_compatible'],
            relpos_max=self.config['relpos_max']
        )

        clamp_fape = self.rng.random() < self.config['clamp_fape_prob']
        if cif_file is not None:
            out_dict['ground_truth'] = features_summit.cif_featurize(
                cif_file,
                asym_ids[0], # choose first asym id
                crop_range=crop_range
            )
            assert len(out_dict['target']['rec_1d']) == len(out_dict['ground_truth']['gt_aatype']), \
                (len(out_dict['target']['rec_1d']), len(out_dict['ground_truth']['gt_aatype']))

            out_dict['ground_truth']['clamp_fape'] = torch.tensor(0)
            if clamp_fape:
                out_dict['ground_truth']['clamp_fape'] = torch.tensor(1)

        out_dict['msa'] = features_summit.msa_featurize(
            a3m_files,
            self.rng,
            self.config['msa_max_clusters'],
            self.config['msa_max_extra'],
            use_cache=self.config['msa_use_cache'],
            crop_range=crop_range,
            num_block_del=self.config['msa_block_del_num'],
            block_del_size=self.config['msa_block_del_size'],
            random_replace_fraction=self.config['msa_random_replace_fraction'],
            uniform_prob=self.config['msa_uniform_prob'],
            profile_prob=self.config['msa_profile_prob'],
            same_prob=self.config['msa_same_prob']
        )

        #assert first_seq == seq
        assert out_dict['msa']['main'].shape[1] == out_dict['target']['rec_1d'].shape[0], \
            (out_dict['msa']['main'].shape[1], out_dict['target']['rec_1d'].shape[0], item)

        return out_dict

    def _get_item(self, ix):
        item = self.data[ix]

        #print('sample', ix, ':', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item['pdb_id'], item['entity_id']); sys.stdout.flush()
        t0 = time.time()

        out_dict = self.make_features(
            item['entity_info']['pdbx_seq_one_letter_code_can'],
            [self.dataset_dir / x for x in item['a3m_files']],
            self.dataset_dir / item['cif_file'] if item['cif_file'] is not None else None,
            item['entity_info']['asym_ids'] if item['cif_file'] is not None else None
        )
        out_dict['target']['ix'] = ix

        print('time retrieving', ix, ':', time.time() - t0, '(s)'); sys.stdout.flush()
        return out_dict

    def __getitem__(self, ix):
        return self._get_item(ix)


class DockingDatasetSimulated(Dataset):
    def __init__(
            self,
            size=10,
            num_res=400,
            num_atoms=50,
            num_frag_main=16,
            num_frag_extra=16,
            num_hh=2,
    ):
        self.size = size
        self.num_res = num_res
        self.num_atoms = num_atoms
        self.num_frag_main = num_frag_main
        self.num_frag_extra = num_frag_extra
        self.num_hh = num_hh

    def __len__(self):
        return self.size

    def _get_item(self):
        aatype = 'S'
        aaorder = residue_constants.restype_order_with_x[aatype]
        num_res = self.num_res
        num_atoms = self.num_atoms
        num_frag_main = self.num_frag_main
        num_frag_extra = self.num_frag_extra
        num_hh = self.num_hh

        inputs = {
            'target': {
                'rec_1d': torch.ones(torch.Size([num_res, 39]), dtype= torch.float32),
                'rec_2d': torch.ones(torch.Size([num_res, num_res, 40]) ,dtype= torch.float32),
                'rec_relpos': torch.zeros(torch.Size([num_res, num_res, 65]) , dtype= torch.float32),
                'rec_atom14_coords': torch.zeros(torch.Size([num_res, 14, 3]) , dtype=torch.float32),
                'rec_atom14_has_coords': torch.zeros(torch.Size([num_res, 14]), dtype=torch.float32),
                'rec_atom37_coords': torch.zeros(torch.Size([num_res, 37, 3]) , dtype=torch.float32),
                'rec_atom37_has_coords': torch.zeros(torch.Size([num_res, 37]) , dtype=torch.float32),
                'rec_aatype': torch.zeros(torch.Size([num_res]), dtype=torch.int64),
                'rec_bb_affine': torch.zeros(torch.Size([num_res, 7]) , dtype= torch.float32),
                'rec_bb_affine_mask': torch.ones(torch.Size([num_res]) , dtype= torch.float32),
                'rec_atom14_atom_is_ambiguous': torch.zeros(torch.Size([num_res, 14]) , dtype= torch.float32),
                'rec_atom14_atom_exists': torch.zeros(torch.Size([num_res, 14]), dtype=  torch.float32),
                'rec_index': torch.from_numpy(np.arange(num_res), dtype=torch.int64),
                'rigidgroups_gt_frames': torch.zeros(torch.Size([num_res, 8, 12]) , dtype= torch.float32),
                'rigidgroups_gt_exists': torch.zeros(torch.Size([num_res, 8]) , dtype= torch.float32),
                'rigidgroups_group_exists': torch.zeros(torch.Size([num_res, 8]) , dtype= torch.float32),
                'rigidgroups_group_is_ambiguous': torch.zeros(torch.Size([num_res, 8]), dtype=  torch.float32),
                'rigidgroups_alt_gt_frames': torch.zeros(torch.Size([num_res, 8, 12]) , dtype= torch.float32),
                'rec_torsions_sin_cos': torch.zeros(torch.Size([num_res, 7, 2]) , dtype=torch.float32),
                'rec_torsions_sin_cos_alt': torch.zeros(torch.Size([num_res, 7, 2]) , dtype=torch.float32),
                'rec_torsions_mask': torch.zeros(torch.Size([num_res, 7]) , dtype=torch.float32),
                'lig_1d': torch.ones(torch.Size([num_atoms, 47]) , dtype= torch.float32),
                'lig_2d': torch.ones(torch.Size([num_atoms, num_atoms, 6]), dtype=  torch.float32),
                'lig_bonded_2d': torch.ones(torch.Size([num_atoms, num_atoms]), dtype=  torch.float32),
                'lig_starts': torch.tensor([0], dtype=torch.int64),
                'lig_ends': torch.tensor([num_atoms], dtype=torch.int64),
                'lig_atom_types': torch.zeros(num_atoms, dtype=torch.int64),
                'lig_init_coords': torch.zeros([num_atoms, 3], dtype=torch.float32),
                'ix': torch.tensor(0)
            },
            'ground_truth': {
                'gt_aatype': torch.zeros(torch.Size([num_res]) , dtype=torch.int64),
                'gt_atom14_coords': torch.zeros(torch.Size([num_res, 14, 3]) , dtype=torch.float32),
                'gt_atom14_has_coords': torch.zeros(torch.Size([num_res, 14]) , dtype=torch.float32),
                'gt_atom14_coords_alt': torch.zeros(torch.Size([num_res, 14, 3]) , dtype=torch.float32),
                'gt_atom14_has_coords_alt': torch.zeros(torch.Size([num_res, 14]) , dtype=torch.float32),
                'gt_atom14_atom_is_ambiguous': torch.zeros(torch.Size([num_res, 14]) , dtype=torch.float32),
                'gt_torsions_sin_cos': torch.zeros(torch.Size([num_res, 7, 2]) , dtype=torch.float32),
                'gt_torsions_sin_cos_alt': torch.zeros(torch.Size([num_res, 7, 2]) , dtype=torch.float32),
                'gt_torsions_mask': torch.zeros(torch.Size([num_res, 7]) , dtype=torch.float32),

                'gt_rigidgroups_gt_frames': torch.zeros(torch.Size([num_res, 8, 12]) , dtype= torch.float32),
                'gt_rigidgroups_alt_gt_frames': torch.zeros(torch.Size([num_res, 8, 12]) , dtype= torch.float32),
                'gt_rigidgroups_gt_exists': torch.zeros(torch.Size([num_res, 8]) , dtype= torch.float32),
                'gt_rigidgroups_group_is_ambiguous': torch.zeros(torch.Size([num_res, 8]), dtype=  torch.float32),
                'gt_bb_affine': torch.zeros(torch.Size([num_res, 7]) , dtype= torch.float32),
                'gt_bb_affine_mask': torch.ones(torch.Size([num_res]) , dtype= torch.float32),

                'gt_residue_index': torch.zeros(torch.Size([num_res]) , dtype=torch.int64),
                'gt_has_frame': torch.zeros(torch.Size([num_res]) , dtype=torch.float32),
                'gt_lig_coords': torch.zeros(torch.Size([4, num_atoms, 3]) , dtype=torch.float32),
                'gt_lig_has_coords': torch.zeros(torch.Size([4, num_atoms]) , dtype=torch.float32),
                'clamp_fape': torch.tensor(0)
            },
            'hhpred': {
                'lig_1d': torch.ones(torch.Size([num_hh, num_atoms, 49]) , dtype=torch.float32),
                'rec_1d': torch.ones(torch.Size([num_hh, num_res, 24]) , dtype=torch.float32),
                'll_2d': torch.ones(torch.Size([num_hh, num_atoms, num_atoms, 142]) , dtype=torch.float32),
                'rr_2d': torch.ones(torch.Size([num_hh, num_res, num_res, 84]) , dtype=torch.float32),
                'rl_2d': torch.ones(torch.Size([num_hh, num_res, num_atoms, 110]) , dtype=torch.float32),
                'lr_2d': torch.ones(torch.Size([num_hh, num_atoms, num_res, 110]) , dtype=torch.float32),
            },
            'fragments': {
                'main': torch.ones(torch.Size([num_frag_main, num_atoms, 238]) , dtype=torch.float32),
                'extra': torch.ones(torch.Size([num_frag_extra, num_atoms, 238]) , dtype=torch.float32),
            }
        }
        inputs['target']['rec_atom14_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
        inputs['target']['rec_atom37_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom37_mask[aaorder])
        inputs['target']['rec_aatype'][:] = aaorder
        inputs['target']['rec_bb_affine'][:, 0] = 1
        inputs['target']['rec_atom14_atom_exists'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
        inputs['target']['rigidgroups_gt_frames'][:] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        inputs['target']['rigidgroups_gt_exists'][:, :5] = 1
        inputs['target']['rigidgroups_group_exists'][:, :5] = 1
        inputs['target']['rigidgroups_alt_gt_frames'] = inputs['target']['rigidgroups_gt_frames'].clone()
        inputs['target']['rec_torsions_sin_cos'][:, :,  0] = 1
        inputs['target']['rec_torsions_sin_cos_alt'][:, :, 0] = 1
        inputs['target']['rec_torsions_mask'][:, :4] = 1

        inputs['ground_truth']['gt_aatype'][:] = aaorder
        inputs['ground_truth']['gt_atom14_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
        inputs['ground_truth']['gt_atom14_has_coords_alt'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
        inputs['ground_truth']['gt_torsions_sin_cos'][:, :, 0] = 1
        inputs['ground_truth']['gt_torsions_sin_cos_alt'][:, :, 0] = 1
        inputs['ground_truth']['gt_torsions_mask'][:, :4] = 1
        inputs['ground_truth']['gt_has_frame'][:] = 1
        inputs['ground_truth']['gt_lig_has_coords'][:] = 1
        inputs['ground_truth']['gt_bb_affine'][:, 0] = 1
        inputs['ground_truth']['gt_rigidgroups_gt_frames'][:] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        inputs['ground_truth']['gt_rigidgroups_gt_exists'][:, :5] = 1
        inputs['ground_truth']['gt_rigidgroups_alt_gt_frames'] = inputs['ground_truth']['gt_rigidgroups_gt_frames'].clone()

        if num_hh == 0:
            del inputs['hhpred']

        if num_frag_extra + num_frag_main == 0:
            del inputs['fragments']
        elif num_frag_main == 0:
            del inputs['fragments']['main']
        elif num_frag_extra == 0:
            del inputs['fragments']['extra']

        return inputs

    def __getitem__(self, ix):
        return self._get_item()


def main():
    ds = DockingDataset(DATA_DIR , 'train_split/train_12k_cleaned.json')
    #print(ds[0]['affinity_class'])
    #print(ds[0])

    item = ds[0]
    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)


def find_bug():
    ds = DockingDataset(
        DATA_DIR,
        'train_split/train_12k.json',
        max_hh_templates=6,
        max_frag_main=16,
        max_frag_extra=16,
        seed=1 * 100
    )
    for i in range(88, len(ds)):
        item = ds[i]
        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None].cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)


def simulate():
    ds = DockingDatasetSimulated(num_hh=0, num_frag_extra=0)
    item = ds[0]
    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)


if __name__ == '__main__':
    import config
    import tqdm
    ds = DockingDataset(
        config.DATA_DIR,
        '15k/folding/debug_15k.json',
        shuffle=True,
        seed=122 * 100
    )
    for i, x in enumerate(ds.data):
        if x['pdb_id'] == '1jwy' and x['entity_id'] == '1':
            print(i)
            break

    ds[6788]
    exit(0)

    for item in tqdm.tqdm(ds):
        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None] #.cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

