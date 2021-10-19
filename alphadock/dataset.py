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
#import horovod.torch as hvd

from config import DATA_DIR
from alphadock import utils
from alphadock import features_summit
from alphadock import residue_constants


class DockingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            json_file,
            max_hh_templates=2,
            max_frag_main=32,
            max_frag_extra=64,
            use_hh_prob=0.5,
            sample_to_size=None,
            clamp_fape_prob=0.5,
            max_num_res=None,
            seed=123456
    ):
        self.dataset_dir = Path(dataset_dir).abspath()
        self.json_file = self.dataset_dir / json_file

        dataset = utils.read_json(self.json_file)
        self.data = dataset['cases']
        self.template_pool = ['_'.join(x) for x in dataset['template_pool']]

        self.max_hh_templates = max_hh_templates
        self.max_frag_main = max_frag_main
        self.max_frag_extra = max_frag_extra
        self.use_hh_prob = use_hh_prob
        self.clamp_fape_prob = clamp_fape_prob
        self.rng = np.random.default_rng(seed)

        if max_num_res is not None:
            self.data = [x for x in self.data if len(utils.read_json(DATA_DIR / 'cases' / x['case_name'] / 'case.json')['entity_info']['pdbx_seq_one_letter_code']) <= max_num_res]

        if sample_to_size is not None:
            probs = np.array([1. / x['seqclus_size'] for x in self.data])
            probs /= probs.sum()
            self.data = self.rng.choice(self.data, size=min(sample_to_size, len(self.data)), replace=False, p=probs)
            #self.rng = np.random.default_rng(seed)

    #def set_seed(self, seed):
    #    self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def _get_hh_list(self, case_dict, group_dict, group_dir, fragment_matches):
        fragment_matches_dict = {}
        for x in fragment_matches:
            fragment_matches_dict[x['tpl_chemid']] = {'tar_matches': x['tar_matches'], 'tpl_matches': x['tpl_matches']}

        args = []
        hh_match_id = 0
        for lig_id, lig_dict in enumerate(group_dict['ligands']):
            hh_json = group_dir / lig_dict['sdf_id'] + '.templates.json'
            if not hh_json.exists():
                continue
            hh_matches = utils.read_json(hh_json)

            for hh_match in hh_matches:
                # omit self templates
                if hh_match['hhpred']['hh_pdb'][:4].lower() == case_dict['pdb_id'].lower():
                    continue

                tpl_case_dict = utils.read_json(DATA_DIR / 'cases' / hh_match['hhpred']['hh_pdb'] / 'case.json')
                tpl_chemid = hh_match['lig_match']['ref_chemid']

                lig_match = fragment_matches_dict.get(tpl_chemid, [])
                if len(lig_match) == 0:
                    continue

                tpl_sdf_id = None
                for tpl_group_dict in tpl_case_dict['ligand_groups']:
                    for tpl_lig_dict in tpl_group_dict['ligands']:
                        if tpl_lig_dict['chemid'] == tpl_chemid:
                            if (hh_match['hhpred']['hh_pdb'].upper() + '_' + tpl_group_dict['name']) not in self.template_pool:
                                continue
                            tpl_sdf_id = tpl_lig_dict['sdf_id']

                if tpl_sdf_id is None:
                    continue
                tar_matches = lig_match['tar_matches']
                tpl_matches = lig_match['tpl_matches']
                for tar_m in tar_matches:
                    for tpl_m in tpl_matches:
                        args.append({
                            'hh_match_id': hh_match_id,
                            'hh_prob': hh_match['hhpred']['hh_prob'],
                            'mcs_tanimoto': hh_match['lig_match']['mcs_tanimoto'],
                            'seqclus30': tpl_case_dict['seqclus30'],
                            'chemid': tpl_chemid,
                            'args': [case_dict, group_dict, lig_id, hh_match, tpl_case_dict, tpl_sdf_id, tar_m, tpl_m]
                        })
                hh_match_id += 1

        if len(args) == 0:
            return {}

        seqclus30_counter = Counter([x['seqclus30'] for x in args])
        #seqclus30_counter_sum = sum(seqclus30_counter.values())
        chemid_counter = Counter([x['chemid'] for x in args])
        #chemid_counter_sum = sum(chemid_counter.values())
        weights = np.array([1. / (seqclus30_counter[x['seqclus30']] * chemid_counter[x['chemid']]) for x in args])
        weights /= weights.sum()

        hh_list = defaultdict(list)
        args_selected = self.rng.choice(args, size=min(len(args), self.max_hh_templates), replace=False, p=weights)
        for arg in args_selected:
            buf = features_summit.hh_template_featurize(*arg['args'])
            for k, v in buf.items():
                hh_list[k].append(v)
            #print({k: v.shape for k, v in buf.items()})

        for k in hh_list.keys():
            hh_list[k] = np.stack(hh_list[k])

        return hh_list

    def _get_frag_list(self, fragment_matches, size):
        choices = []
        for match in fragment_matches:
            for mapping in match['mapping_to_target']:
                choices.append([match, mapping])
        if len(choices) == 0:
            return []

        # score matches by the target coverage. matches that cover more
        # common positions get larger scores and the probability of
        # selection is divided by this number: more common positions
        # result in decreased selection probability
        mapping_cat = (np.stack([x for _, x in choices]) > -1).astype(np.int)
        mapping_pos_counts = mapping_cat.sum(0)
        match_scores = np.matmul(mapping_cat, mapping_pos_counts)

        seqclus30_counter = Counter([x[0]['tpl_seqclus30'] for x in choices])
        chemid_counter = Counter([x[0]['tpl_chemid'] for x in choices])
        weights = np.array([1. / (seqclus30_counter[x[0]['tpl_seqclus30']] * chemid_counter[x[0]['tpl_chemid']]) for xi, x in enumerate(choices)])
        weights /= match_scores
        weights /= weights.sum()

        selected = self.rng.choice(np.array(choices, dtype=object), size=min(len(choices), size), replace=False, p=weights)
        return selected

    def _get_item(self, ix):
        item = self.data[ix]
        #item = {}
        #item['case_name'] = '3ASK_D'
        #item['group_name'] = 'M3L'

        case_dir = DATA_DIR / 'cases' / item['case_name']
        case_dict = utils.read_json(case_dir / 'case.json')
        group_dict = {x['name']: x for x in case_dict['ligand_groups']}[item['group_name']]
        group_dir = case_dir / group_dict['name']

        print(': getting sample #', ix)
        print(':', group_dir)
        sys.stdout.flush()

        out_dict = {}

        # process target group
        target_dict = features_summit.target_rec_featurize(case_dict)
        target_dict.update(features_summit.target_group_featurize(case_dict, group_dict))
        # add atom types from onehot encoding
        target_dict['lig_atom_types'] = np.where(target_dict['lig_1d'][:, :len(features_summit.ELEMENTS_ORDER)] > 0)[1]
        target_dict['ix'] = ix

        out_dict['target'] = target_dict
        out_dict['ground_truth'] = features_summit.ground_truth_featurize(case_dict, group_dict)
        out_dict['ground_truth']['clamp_fape'] = torch.tensor(0)
        if self.rng.random() < self.clamp_fape_prob:
            out_dict['ground_truth']['clamp_fape'] = torch.tensor(1)

        fragment_matches = []
        frag_json = Path(f"{DATA_DIR}/featurized/{case_dict['case_name']}.{group_dict['name']}.fragment_matches.json")
        if frag_json.exists():
            fragment_matches = utils.read_json(frag_json)
        fragment_matches = [x for x in fragment_matches if (x['tpl_chain'] + '_' + x['tpl_group']) in self.template_pool]

        # process hhpred
        if self.rng.random() < self.use_hh_prob and self.max_hh_templates > 0:
            hhpred = self._get_hh_list(case_dict, group_dict, group_dir, fragment_matches)
            if len(hhpred) > 0:
                out_dict['hhpred'] = hhpred

        # process fragments
        if len(fragment_matches) > 0 and (self.max_frag_main + self.max_frag_extra) > 0:
            out_dict['fragments'] = {}
            selected_fragments = self._get_frag_list(fragment_matches, self.max_frag_main + self.max_frag_extra)

            selected_main = selected_fragments[:self.max_frag_main]
            tpl_case_dicts = [utils.read_json(DATA_DIR / 'cases' / x[0]['tpl_chain'] / 'case.json') for x in selected_main]
            tpl_group_dicts = [tpl_case_dicts[i]['ligand_groups'][x[0]['tpl_group_id']] for i, x in enumerate(selected_main)]
            tpl_mappings = [x[1] for x in selected_main]
            out_dict['fragments']['main'] = features_summit.fragment_extra_list_featurize(
                tpl_case_dicts,
                tpl_group_dicts,
                tpl_mappings
            )

            selected_extra = selected_fragments[self.max_frag_main:self.max_frag_main + self.max_frag_extra]
            if len(selected_extra) > 0:
                tpl_case_dicts = [utils.read_json(DATA_DIR / 'cases' / x[0]['tpl_chain'] / 'case.json') for x in selected_extra]
                tpl_group_dicts = [tpl_case_dicts[i]['ligand_groups'][x[0]['tpl_group_id']] for i, x in enumerate(selected_extra)]
                tpl_mappings = [x[1] for x in selected_extra]
                out_dict['fragments']['extra'] = features_summit.fragment_extra_list_featurize(
                    tpl_case_dicts,
                    tpl_group_dicts,
                    tpl_mappings
                )

        if False:
            # Print fragment matches aligned to the target ligand group
            num_ele = len(features_summit.ELEMENTS_ORDER)
            tar_onehot = np.where(target_dict['lig_1d'][:, :num_ele] > 0)[1]
            print([features_summit.ELEMENTS_ORDER[x] for x in tar_onehot])

            #print(np.where(out_dict['fragments']['extra'][:, :, :num_ele] > 0))
            for ei in range(out_dict['fragments']['extra'].shape[0]):
                extra_onehot = out_dict['fragments']['extra'][ei, :, :num_ele].copy()
                extra_mask = np.any(extra_onehot > 0, axis=1)
                print([features_summit.ELEMENTS_ORDER[np.where(extra_onehot[xi])[0][0]] if x else '-' for xi, x in enumerate(extra_mask)])

        return out_dict

    def _get_item_simple(self, ix):
        item = self.data[ix]
        rank = 0

        item = {}
        item['case_name'] = '3ASK_D'
        item['group_name'] = 'M3L'

        case_dir = DATA_DIR / 'cases' / item['case_name']
        case_dict = utils.read_json(case_dir / 'case.json')
        group_dict = {x['name']: x for x in case_dict['ligand_groups']}[item['group_name']]
        group_dir = case_dir / group_dict['name']
        sys.stdout.flush()

        out_dict = {}

        # process target group
        target_dict = features_summit.target_rec_featurize(case_dict)
        target_dict.update(features_summit.target_group_featurize(case_dict, group_dict))
        # add atom types from onehot encoding
        target_dict['lig_atom_types'] = np.where(target_dict['lig_1d'][:, :len(features_summit.ELEMENTS_ORDER)] > 0)[1]
        target_dict['ix'] = ix

        out_dict['target'] = target_dict
        out_dict['ground_truth'] = features_summit.ground_truth_featurize(case_dict, group_dict)
        if self.rng.random() < self.clamp_fape_prob:
            out_dict['ground_truth']['clamp_fape'] = torch.tensor(1)

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
                'lig_starts': torch.tensor([0], dtype=torch.int64),
                'lig_ends': torch.tensor([num_atoms], dtype=torch.int64),
                'lig_atom_types': torch.zeros(num_atoms, dtype=torch.int64),
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
    ds = DockingDataset(DATA_DIR , 'train_split/train_12k.json')
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
    main()

