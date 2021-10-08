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

from config import DATA_DIR
from alphadock import utils
from alphadock import features_summit


class DockingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            json_file,
            max_hh_templates=2,
            max_frag_templates=2,
            max_frag_extra=2,
            seed=123456
    ):

        self.dataset_dir = Path(dataset_dir).abspath()
        if not self.dataset_dir.exists():
            raise OSError(f'Directory {self.dataset_dir} does not exist')

        #self.subset = subset

        self.json_file = self.dataset_dir / json_file
        if not self.json_file.exists():
            raise OSError(f'File {self.json_file} does not exist')

        self.data = utils.read_json(self.json_file)
        #if subset is not None:
        #    self.data = [v for k, v in enumerate(self.data) if k in subset]

        self.max_hh_templates = max_hh_templates
        self.max_frag_templates = max_frag_templates
        self.max_frag_extra = max_frag_extra
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def _get_hh_list(self, case_dict, group_dict, group_dir, fragment_matches):
        fragment_matches_dict = {}
        for x in fragment_matches:
            fragment_matches_dict[x['tpl_chemid']] = {'tar_matches': x['tar_matches'], 'tpl_matches': x['tpl_matches']}

        args = []
        hh_match_id = 0
        for lig_id, lig_dict in enumerate(group_dict['ligands']):
            hh_matches = utils.read_json(group_dir / lig_dict['sdf_id'] + '.templates.json')
            #print(lig_dict['sdf_id'])
            for hh_match in hh_matches:
                tpl_case_dict = utils.read_json(DATA_DIR / 'cases' / hh_match['hhpred']['hh_pdb'] / 'case.json')
                tpl_chemid = hh_match['lig_match']['ref_chemid']

                lig_match = fragment_matches_dict.get(tpl_chemid, [])
                if len(lig_match) == 0:
                    continue

                for tpl_group_dict in tpl_case_dict['ligand_groups']:
                    for tpl_lig_dict in tpl_group_dict['ligands']:
                        if tpl_lig_dict['chemid'] == tpl_chemid:
                            tpl_sdf_id = tpl_lig_dict['sdf_id']

                tar_matches = lig_match['tar_matches']
                tpl_matches = lig_match['tpl_matches']
                #print(len(tar_matches) * len(tpl_matches))
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

        selected = self.rng.choice(choices, size=min(len(choices), size), replace=False, p=weights)
        return selected

    def _get_item(self, ix):
        item = self.data[ix]

        #item = {}
        #item['case_name'] = '4X63_A'
        #item['group_name'] = '3XV_SAH'

        case_dir = DATA_DIR / 'cases' / item['case_name']
        case_dict = utils.read_json(case_dir / 'case.json')
        group_dict = {x['name']: x for x in case_dict['ligand_groups']}[item['group_name']]
        group_dir = case_dir / group_dict['name']
        print(group_dir)

        out_dict = {}

        # process target group
        target_dict = features_summit.target_rec_featurize(case_dict)
        target_dict.update(features_summit.target_group_featurize(case_dict, group_dict))
        out_dict['target'] = target_dict
        out_dict['ground_truth'] = features_summit.ground_truth_featurize(case_dict, group_dict)

        fragment_matches = utils.read_json(f"{DATA_DIR}/featurized/{case_dict['case_name']}.{group_dict['name']}.fragment_matches.json")

        # process hhpred
        hhpred = self._get_hh_list(case_dict, group_dict, group_dir, fragment_matches)
        if len(hhpred) > 0:
            out_dict['hhpred'] = hhpred

        # process fragments
        if len(fragment_matches) > 0:
            selected_fragments = self._get_frag_list(fragment_matches, self.max_frag_templates)
            tpl_case_dicts = [utils.read_json(DATA_DIR / 'cases' / x[0]['tpl_chain'] / 'case.json') for x in selected_fragments]
            tpl_group_dicts = [tpl_case_dicts[i]['ligand_groups'][x[0]['tpl_group_id']] for i, x in enumerate(selected_fragments)]
            tpl_mappings = [x[1] for x in selected_fragments]
            frag_dict = features_summit.fragment_template_list_featurize(
                tpl_case_dicts,
                tpl_group_dicts,
                tpl_mappings
            )
            out_dict['fragments'] = frag_dict

            selected_fragments_extra = self._get_frag_list(fragment_matches, self.max_frag_extra)
            tpl_case_dicts = [utils.read_json(DATA_DIR / 'cases' / x[0]['tpl_chain'] / 'case.json') for x in selected_fragments_extra]
            tpl_group_dicts = [tpl_case_dicts[i]['ligand_groups'][x[0]['tpl_group_id']] for i, x in enumerate(selected_fragments_extra)]
            tpl_mappings = [x[1] for x in selected_fragments_extra]
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

    def __getitem__(self, ix):
        # if there is an error, fall back to the first sample
        #try:
        return self._get_item(ix)
        #except Exception:
        #    traceback.print_exc()
        #    return self._get_item(0)


def main():
    ds = DockingDataset(DATA_DIR , 'train_split/debug.json')
    #print(ds[0]['affinity_class'])
    #print(ds[0])

    item = ds[0]
    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)


if __name__ == '__main__':
    main()

