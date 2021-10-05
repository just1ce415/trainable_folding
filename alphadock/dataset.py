import traceback
import random
import itertools
import numpy as np
import prody
from path import Path
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from functools import partial
import torch

from config import DATA_DIR
import utils
import features


class DockingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            json_file
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

    def __len__(self):
        return len(self.data)

    def _get_item(self, ix):
        item = self.data[ix]
        case_dir = DATA_DIR / 'cases' / item['case_name']
        case_dict = utils.read_json(case_dir / 'case.json')
        group_dict = {x['name']: x for x in case_dict['ligand_groups']}[item['group_name']]
        group_dir = case_dir / group_dict['name']

        out_dict = {}

        # process target group
        target_dict = features.target_rec_featurize(case_dict)
        target_dict.update(features.target_group_featurize(case_dict, group_dict))
        out_dict['target'] = target_dict
        out_dict['ground_truth'] = features.ground_truth_featurize(case_dict, group_dict)

        # process hhpred
        #hh_templates = item['hh_templates']
        # for each tpl get: tar_mol, tar_case_dict, tar_group_dict, tar_lig_id, tpl_mol,  match
        # hh_templates_featurize(case_dict, hh_templates))

        # process fragments
        if len(item['fragment_templates']) > 0:
            frag_dict = features.fragment_template_list_featurize(group_dict, item['fragment_templates'][:])
            out_dict['fragments'] = frag_dict

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

