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

from config import DATA_DIR, DTYPE_FLOAT
from alphadock import utils
from alphadock import features_summit
from alphadock import residue_constants


class DockingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            json_file,
            max_hh_templates=0,
            max_msa_main=128,
            max_msa_extra=1024,
            max_msa_size=4096,
            crop_size=256,
            use_hh_prob=0.5,
            sample_to_size=None,
            clamp_fape_prob=0.9,
            seed=123456,
            shuffle=False

    ):
        self.dataset_dir = Path(dataset_dir).abspath()
        self.json_file = self.dataset_dir / json_file

        dataset = utils.read_json(self.json_file)
        self.data = dataset #['cases']
        #self.template_pool = ['_'.join(x) for x in dataset['template_pool']]

        self.max_hh_templates = max_hh_templates
        self.max_msa_main = max_msa_main
        self.max_msa_extra = max_msa_extra
        self.max_msa_size = max_msa_size
        self.use_hh_prob = use_hh_prob
        self.clamp_fape_prob = clamp_fape_prob
        self.rng = np.random.default_rng(seed)
        self.crop_size = crop_size

        if shuffle:
            self.rng.shuffle(self.data)

        #if sample_to_size is not None:
        #    probs = np.array([1. / x['seqclus_size'] for x in self.data])
        #    probs /= probs.sum()
        #    self.data = self.rng.choice(self.data, size=min(sample_to_size, len(self.data)), replace=False, p=probs)
        #    #self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def _get_item(self, ix):
        item = self.data[ix]

        print('sample', ix, ':', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), item['pdb_id'], item['entity_id']); sys.stdout.flush()
        buf = time.time()

        entity_info = item['entity_info']
        seq = entity_info['pdbx_seq_one_letter_code_can']
        crop_start = self.rng.integers(0, max(1, len(seq) - self.crop_size))
        crop_range = [crop_start, crop_start + self.crop_size]

        # process target group
        out_dict = features_summit.cif_featurize(self.dataset_dir / item['cif_file'], self.rng.choice(entity_info['asym_ids']), crop_range=crop_range)
        out_dict['ground_truth']['clamp_fape'] = torch.tensor(0)
        if self.rng.random() < self.clamp_fape_prob:
            out_dict['ground_truth']['clamp_fape'] = torch.tensor(1)
        out_dict['target']['ix'] = ix

        if self.max_msa_main > 0:
            out_dict['msa'] = features_summit.msa_featurize(
                [self.dataset_dir / x for x in item['a3m_files']],
                self.rng,
                self.max_msa_main,
                self.max_msa_extra,
                crop_range=crop_range
            )

            #assert first_seq == seq
            assert out_dict['msa']['main'].shape[1] == out_dict['target']['rec_1d'].shape[0], \
                (out_dict['msa']['main'].shape[1], out_dict['target']['rec_1d'].shape[0], ix, item)

        print('time retrieving', ix, ':', time.time() - buf, '(s)'); sys.stdout.flush()

        # process hhpred
        #if self.rng.random() < self.use_hh_prob and self.max_hh_templates > 0:
        #    hhpred = self._get_hh_list(case_dict, group_dict, group_dir, fragment_matches)
        #    if len(hhpred) > 0:
         #       out_dict['hhpred'] = hhpred

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

