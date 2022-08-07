import torch
import prody
import numpy as np
import pytest

import all_atom
import docker
import config
import features
import features_summit
import r3
import quat_affine
import residue_constants
import utils


def test():
    ag = prody.parsePDB(config.TEST_DATA_DIR / '1ACM_3mer.pdb')
    print(ag.calpha.getSequence())
    feats = features.ag_to_features(ag, ag.calpha.getSequence(), ag.calpha.getSequence())
    print(np.stack(feats['atom14_coords'])[..., 0])

    atom14_coords = feats['atom14_coords']
    atom14_mask = feats['atom14_has_coords']
    print(atom14_mask)

    feats = features.rec_literal_to_numeric(feats)
    aatype = torch.tensor(feats['seq_aatype_num'])
    print(aatype)

    atom37_coords = all_atom.atom14_to_atom37(torch.tensor(atom14_coords), aatype)
    atom37_mask = all_atom.atom14_to_atom37(torch.tensor(atom14_mask), aatype)
    print(atom37_coords.shape)
    #print(atom37_coords)
    print(atom37_mask.shape)
    #print(atom37_mask)

    frames = all_atom.atom37_to_frames(aatype, atom37_coords, atom37_mask)
    print(frames.keys())
    print(frames['rigidgroups_gt_frames'][:, 0, :])
    bb_frames = r3.rigids_from_tensor_flat12(frames['rigidgroups_gt_frames'][:, 0, :])
    print('bb_frames.trans', bb_frames.trans)
    bb_frames = r3.rigids_to_quataffine(bb_frames)
    print('bb_frames.translation', bb_frames.translation)
    bb_frames.quaternion = quat_affine.rot_to_quat(bb_frames.rotation)
    print('bb_frames.translation', bb_frames.translation)
    #torsions = torch.ones((3, 14))
    torsions = all_atom.atom37_to_torsion_angles(aatype[None], atom37_coords[None], atom37_mask[None], placeholder_for_undefined=True)['torsion_angles_sin_cos'].reshape(3, 14)
    print(bb_frames)
    print(torsions)

    print('bb_frames.to_tensor()', bb_frames.to_tensor())
    out = all_atom.backbone_affine_and_torsions_to_all_atom(bb_frames.to_tensor(), torsions, aatype)
    print(out['atom_pos'])

    out_coords = out['atom_pos']

    out_ag = ag.copy()
    for a in out_ag:
        #residue_constants.restype_3to1[a.getResname()]
        out_atom_id = residue_constants.restype_name_to_atom14_names[a.getResname()].index(a.getName())
        out_res_id = a.getResnum() - 4
        out_atom_coords = [out_coords.x[out_res_id][out_atom_id], out_coords.y[out_res_id][out_atom_id], out_coords.z[out_res_id][out_atom_id]]
        a.setCoords(out_atom_coords)

    tol = 1.0
    #print(out_ag.getCoords() - ag.getCoords())
    assert np.all((out_ag.getCoords() - ag.getCoords()) < tol)
    #prody.writePDB(config.TEST_DATA_DIR / 'out.pdb', out_ag)

    #all_atom.frame_aligned_point_error()


def test_atom37_to_frames():
    case_name = '3ASK_D'
    group_name = 'M3L'

    case_dir = config.TEST_DATA_DIR / case_name
    case_dict = utils.read_json(case_dir / 'case.json')
    group_dict = {x['name']: x for x in case_dict['ligand_groups']}[group_name]

    rec_ag = prody.parsePDB(case_dir / 'rec_orig.pdb')
    rec_dict = features_summit.ag_to_features(
        rec_ag,
        case_dict['entity_info']['pdb_aln'],
        case_dict['entity_info']['entity_aln'],
        no_mismatch=True
    )
    rec_dict = features_summit.rec_literal_to_numeric(rec_dict, seq_include_gap=False)

    atom14_gt_positions = rec_dict['atom14_coords']  # (N, 14, 3)
    atom14_gt_exists = rec_dict['atom14_has_coords']  # (N, 14)
    renaming_mats = all_atom.RENAMING_MATRICES[rec_dict['seq_aatype_num']]  # (N, 14, 14)
    atom14_alt_gt_positions = np.sum(atom14_gt_positions[:, :, None, :] * renaming_mats[:, :, :, None], axis=1)
    atom14_alt_gt_exists = np.sum(rec_dict['atom14_has_coords'][:, :, None] * renaming_mats, axis=1)
    atom14_atom_is_ambiguous = (renaming_mats * np.eye(14)[None]).sum(1) == 0

    atom37_gt_positions = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_positions).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    atom37_gt_exists = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_exists).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    gt_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(rec_dict['seq_aatype_num'][None]), atom37_gt_positions[None].float(), atom37_gt_exists[None].float())

    rec_all_frames = all_atom.atom37_to_frames(torch.from_numpy(rec_dict['seq_aatype_num']), atom37_gt_positions.float(), atom37_gt_exists.float())
    #rec_bb_affine = r3.rigids_to_quataffine(r3.rigids_from_tensor_flat12(rec_all_frames['rigidgroups_gt_frames'][..., 0, :]))

    # Assert that all existing rigid groups rotation matrices are proper -> orthogonal
    rot_mats = rec_all_frames['rigidgroups_gt_frames'][:, :, :9].reshape(-1, 3, 3)
    rot_mats = rot_mats[rec_all_frames['rigidgroups_gt_exists'].flatten() > 0]
    eye_mats = torch.matmul(rot_mats, rot_mats.transpose(1, 2))
    eye_mats = eye_mats.numpy()
    eye_mats_ref = np.tile(np.eye(3), (eye_mats.shape[0], 1, 1))
    np.testing.assert_allclose(eye_mats, eye_mats_ref, atol=1e-5)


if __name__ == '__main__':
    test_atom37_to_frames()
