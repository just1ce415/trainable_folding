import sys
sys.path.insert(1, '../')
from alphadock import all_atom, residue_constants, dataset, config, docker, utils
import pickle
import numpy as np
import torch
import json
from copy import deepcopy
from path import Path
from torch import optim
import math
from monomer import pdb_to_template
from monomer import all_atom_monomer

OUT_DIR = Path('.')
LOG_PDB_EVERY_NSTEPS = 500
GLOBAL_STEP = 0
DATALOADER_KWARGS = {'num_workers': 0, 'pin_memory': True}
CONFIG_DICT = deepcopy(config.config)
TB_WRITE_STEP = False
SAVE_MODEL_EVERY_NEPOCHS = 1

MAX_NAN_ITER_FRAC = 0.05
CLIP_GRADIENT = True
CLIP_GRADIENT_VALUE = 0.1
USE_AMP = False
USE_AMP_SCALER = False

model = None
scheduler = None
optimizer = None
amp_scaler = None
tb_writer = None

HOROVOD_RANK = 0
HOROVOD = False

def pred_to_pdb(out_pdb, input_dict, out_dict):
    out_pdb = Path(out_pdb)
    with open(out_pdb, 'w') as f:
        f.write(f'HEADER {out_pdb.basename().stripext()}.pred\n')
        atom14_to_pdb_stream(
            f,
            input_dict['target']['rec_aatype'][0].cpu(),
            out_dict['final_all_atom']['atom_pos_tensor'].detach().cpu(),
            input_dict['target']['rec_index'][0].cpu(),
            bfactors=out_dict['struct_out']['rec_lddt'][0, -1].detach().cpu().argmax(dim=-1) + 50,
            chain='A',
            serial_start=1,
            resnum_start=1
        )
        if 'ground_truth' in input_dict:
            f.write(f'HEADER {out_pdb.basename().stripext()}.crys\n')
            all_atom.atom14_to_pdb_stream(
                f,
                input_dict['ground_truth']['gt_aatype'][0].cpu(),
                input_dict['ground_truth']['gt_atom14_coords'][0].detach().cpu(),
                atom14_mask=input_dict['ground_truth']['gt_atom14_has_coords'][0].detach().cpu(),
                chain='A',
                serial_start=1,
                resnum_start=1
            )
def format_pdb_line(serial, name, resname, chain, resnum, x, y, z, element, hetatm=False, bfactor=None):
    name = name if len(name) == 4 else ' ' + name
    bfactor = " "*6 if bfactor is None else f"{bfactor: 6.2f}"
    line = f'{"HETATM" if hetatm else "ATOM  "}{serial:>5d} {name:4s} {resname:3s} {chain:1s}{resnum:>4d}' \
           f'    {x: 8.3f}{y: 8.3f}{z: 8.3f}{" "*6}{bfactor}{" "*10}{element:>2s}'
    return line


def atom14_to_pdb_stream(stream, aatypes, atom14_coords, residue_idx, atom14_mask=None, bfactors=None, chain='A', serial_start=1, resnum_start=1):
    assert len(aatypes.shape) == 1, aatypes.shape
    assert len(atom14_coords.shape) == 3, atom14_coords.shape
    assert atom14_coords.shape[0] == aatypes.shape[0], (atom14_coords.shape, aatypes.shape)
    assert atom14_coords.shape[-1] == 3, atom14_coords.shape
    if atom14_mask is not None:
        assert len(atom14_mask.shape) == 2, atom14_mask.shape
        assert atom14_mask.shape[0] == aatypes.shape[0], (atom14_mask.shape, aatypes.shape)
    if bfactors is not None:
        assert len(bfactors.shape) == 1, bfactors.shape
        assert bfactors.shape[0] == aatypes.shape[0], (bfactors.shape[0], aatypes.shape[0])

    serial = serial_start
    for resi, aatype in enumerate(aatypes):
        if aatype >= len(residue_constants.restypes):
            continue
        aa1 = residue_constants.restypes[aatype]
        resname = residue_constants.restype_1to3[aa1]
        for ix, name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if name == '':
                continue
            if atom14_mask is not None and atom14_mask[resi, ix] < 1.0:
                continue
            x, y, z = atom14_coords[resi, ix]
            element = name[0]
            bfactor = None if bfactors is None else bfactors[resi]
            pdb_line = format_pdb_line(serial, name, resname, chain, residue_idx[resi]+1, x, y, z, element, bfactor=bfactor)
            stream.write(pdb_line + '\n')
            serial += 1
    return serial

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, residue_constants.restype_order["G"])
    ca_idx = residue_constants.atom_order["CA"]
    cb_idx = residue_constants.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

if __name__ == '__main__':
    device = 'cuda:0'
    config_dict = deepcopy(config.config)
    config_dict = utils.merge_dicts(config_dict, {
        'model': {
            'msa_bert_block': True,
            'Evoformer': {'device': device, 'EvoformerIteration': {'checkpoint': True}},
            'InputEmbedder': {'device': device, 'ExtraMsaStack': {'device': device, 'ExtraMsaStackIteration': {'checkpoint': True}}},
            'StructureModule': {'device': device}
        }
    })
    config_dict['data']['crop_size'] = None
    config_dict['data']['msa_block_del_num'] = 0
    with open('test/test5/test5.pdb', "r") as fp:
        pdb_string = fp.read()
    protein_object_A = pdb_to_template.from_pdb_string(pdb_string, 'A')
    has_ca = protein_object_A.atom_mask[:, 0] == 1
    template_aatype = torch.unsqueeze(torch.tensor(protein_object_A.aatype[has_ca]), 0)
    template_all_atom_masks = torch.unsqueeze(torch.tensor(protein_object_A.atom_mask[has_ca]), 0)
    template_all_atom_positions = torch.unsqueeze(torch.tensor(protein_object_A.atom_positions[has_ca]), 0)
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor(new_order_list, dtype=torch.int64).expand(1, -1)
    template_aatype_new = torch.gather(new_order, 1, index=template_aatype)
    template_pseudo_beta, template_pseudo_mask = pseudo_beta_fn(template_aatype_new, template_all_atom_positions, template_all_atom_masks)
    template_torsion = all_atom_monomer.atom37_to_torsion_angles(template_aatype_new, template_all_atom_positions, template_all_atom_masks)

    model = docker.DockerIteration(config_dict['model'], config_dict)
    model.modules_to_devices()
    dict_pth = torch.load('params_model_1_ptm.pth')
    model.load_state_dict(dict_pth['model_state_dict'])
    dset = dataset.DockingDataset(
        utils.read_json('test/test5/test.json'),
        config_dict['data'],
        seed=874630,
        shuffle=False
    )
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, **DATALOADER_KWARGS)
    for batch in loader:
        main_extra = torch.zeros((1,1, template_aatype_new.shape[1], 25))
        extra_mask = torch.zeros((1, main_extra.shape[1], main_extra.shape[2]))
        batch['msa']['extra'] = main_extra
        batch['msa']['extra_mask'] = extra_mask
        batch['template'] = {}
        batch['template']['template_aatype'] = torch.unsqueeze(template_aatype_new, 0)
        batch['template']['template_all_atom_masks'] = torch.unsqueeze(template_all_atom_masks, 0).type(torch.float32)
        batch['template']['template_all_atom_positions'] = torch.unsqueeze(template_all_atom_positions, 0).type(torch.float32)
        batch['template']['template_pseudo_beta'] = torch.unsqueeze(template_pseudo_beta, 0).type(torch.float32)
        batch['template']['template_pseudo_beta_mask'] = torch.unsqueeze(template_pseudo_mask, 0).type(torch.float32)
        batch['template']['template_mask'] = torch.tensor([[1.0]])
        batch['template']['template_torsion_angles_sin_cos'] = torch.unsqueeze(template_torsion['template_torsion_angles_sin_cos'], 0).type(torch.float32)
        batch['template']['template_alt_torsion_angles_sin_cos'] = torch.unsqueeze(template_torsion['template_alt_torsion_angles_sin_cos'], 0).type(torch.float32)
        batch['template']['template_torsion_angles_mask'] = torch.unsqueeze(template_torsion['template_torsion_angles_mask'], 0).type(torch.float32)
        for k in ['target', 'msa', 'template']:
            for i, j in batch[k].items():
                print(i, j.shape)
        for k, v in batch['target'].items():
            batch['target'][k] = v.to('cuda:0')
        for k, v in batch['msa'].items():
            batch['msa'][k] = v.to('cuda:0')
        for k, v in batch['template'].items():
            batch['template'][k] = v.to('cuda:0')
        num_recycles = 4
        with torch.no_grad():
            for recycle_iter in range (num_recycles):
                output = model(batch, recycling=output['recycling_input'] if recycle_iter > 0 else None)
        pred_to_pdb('test.pdb', batch, output)


