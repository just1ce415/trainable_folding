import torch
import prody
import numpy as np
import pytest

import all_atom
import docker
import config
import features
import r3
import quat_affine
import residue_constants
import lddt


def mock_ground_truth():
    ag = prody.parsePDB(config.TEST_DATA_DIR / 'ARG_ASP.pdb')
    ag_dict = features.ag_to_features(ag, ag.calpha.getSequence(), ag.calpha.getSequence())
    ag_dict['atom14_coords'] = np.stack(ag_dict['atom14_coords'])
    ag_dict['atom14_has_coords'] = np.stack(ag_dict['atom14_has_coords'])
    ag_dict['has_frame'] = np.array(ag_dict['has_frame'])
    aatype_num = np.array([features.AATYPE_WITH_X[x] for x in ag_dict['seq_aatype']]).astype(config.DTYPE_INT)

    ag_swapped = prody.parsePDB(config.TEST_DATA_DIR / 'ARG_ASP_swapped.pdb')
    ag_swapped_dict = features.ag_to_features(ag_swapped, ag_swapped.calpha.getSequence(), ag_swapped.calpha.getSequence())
    ag_swapped_dict['atom14_coords'] = np.stack(ag_swapped_dict['atom14_coords'])
    ag_swapped_dict['atom14_has_coords'] = np.stack(ag_swapped_dict['atom14_has_coords'])

    ag_ambig = np.zeros_like(ag_dict['atom14_has_coords'])
    ag_ambig[1, [6, 7]] = 1

    atom37_gt_positions = all_atom.atom14_to_atom37(torch.from_numpy(ag_dict['atom14_coords']), torch.from_numpy(aatype_num))
    atom37_gt_exists = all_atom.atom14_to_atom37(torch.from_numpy(ag_dict['atom14_has_coords']), torch.from_numpy(aatype_num))
    gt_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(aatype_num[None]), atom37_gt_positions[None], atom37_gt_exists[None])
    gt_frames = all_atom.atom37_to_frames(torch.from_numpy(aatype_num[None]), atom37_gt_positions[None], atom37_gt_exists[None])

    # 4 atoms laid out as a square
    lig_coords = [
        [[25, 33, 1], [25, 33, 5], [25, 37, 5], [25, 37, 1]],
        [[25, 37, 5], [25, 37, 1], [25, 33, 1], [25, 33, 5]]
    ]

    lig_has_coords = [
        [1,1,1,1],
        [1,1,1,1]
    ]

    out = {
        'gt_aatype': aatype_num,  # same as for target
        'gt_atom14_coords': ag_dict['atom14_coords'].astype(config.DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords': ag_dict['atom14_has_coords'].astype(config.DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_coords_alt': ag_swapped_dict['atom14_coords'].astype(config.DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords_alt': ag_swapped_dict['atom14_has_coords'].astype(config.DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_atom_is_ambiguous': ag_ambig.astype(config.DTYPE_FLOAT),

        'gt_torsions_sin_cos': gt_torsions['torsion_angles_sin_cos'][0].numpy(),
        'gt_torsions_sin_cos_alt': gt_torsions['alt_torsion_angles_sin_cos'][0].numpy(),
        'gt_torsions_mask': gt_torsions['torsion_angles_mask'][0].numpy(),

        'gt_residue_index': np.arange(len(aatype_num), dtype=config.DTYPE_INT),  # (N_res)
        'gt_has_frame': ag_dict['has_frame'].astype(config.DTYPE_FLOAT),  # (N_res)
        'gt_lig_coords': np.array(lig_coords).astype(config.DTYPE_FLOAT),  # (N_symm, N_atoms, 3)
        'gt_lig_has_coords': np.array(lig_has_coords).astype(config.DTYPE_FLOAT),  # (N_symm, N_atoms)
    }

    out = {k: torch.as_tensor(v)[None] for k, v in out.items()}

    out.update(gt_frames)
    return out


def test():
    from config import config
    gt_dict = mock_ground_truth()

    for k, v in gt_dict.items():
        print(k, v.shape)

    # loss must be zero
    print(docker.torsion_loss(
        {'ground_truth': gt_dict},
        {'rec_torsions': gt_dict['gt_torsions_sin_cos'][None]}
    ))

    # chi_loss = 0, norm_loss = 1
    print(docker.torsion_loss(
        {'ground_truth': gt_dict},
        {'rec_torsions': 2 * gt_dict['gt_torsions_sin_cos'][None]}
    ))

    # all losses equal to zero
    print(docker.torsion_loss(
        {'ground_truth': gt_dict},
        {'rec_torsions': gt_dict['gt_torsions_sin_cos'][None].tile(3, 1, 1, 1)}
    ))

    print(gt_dict['gt_lig_coords'])

    # should be equal to 1 for each atom
    tmp = lddt.lddt(
        predicted_points_a=gt_dict['gt_lig_coords'][0, 0][None],
        predicted_points_b=gt_dict['gt_lig_coords'][0, 0][None],
        true_points_a=gt_dict['gt_lig_coords'][0, 0][None],
        true_points_b=gt_dict['gt_lig_coords'][0, 0][None],
        true_points_a_mask=gt_dict['gt_lig_has_coords'][0, 0][None, :, None],
        true_points_b_mask=gt_dict['gt_lig_has_coords'][0, 0][None, :, None],
        cutoff=15,
        exclude_self=True,
        reduce_axis=1,
        per_residue=True
    )

    a_crd = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=torch.float)
    b_crd = torch.tensor([[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]], dtype=torch.float)
    b_pred_crd = torch.tensor([[[0, 0, -0.75]], [[0, 0, -1.25]], [[0, 0, -3.0]]], dtype=torch.float)
    tmp = lddt.lddt(
        predicted_points_a=a_crd[None],
        predicted_points_b=b_pred_crd,
        true_points_a=a_crd[None],
        true_points_b=b_crd,
        true_points_a_mask=torch.ones_like(a_crd[..., 0])[None, :, None],
        true_points_b_mask=torch.ones_like(b_crd[..., 0])[:, :, None],
        cutoff=15,
        exclude_self=False,
        reduce_axis=1,
        per_residue=True
    )
    print(tmp)

    rec_T = gt_dict['gt_atom14_coords'][0, :, 1, :]
    rec_T = torch.cat([torch.zeros(rec_T.shape[0], 4), rec_T], dim=-1)[None, None] / config['position_scale']

    lig_T = gt_dict['gt_lig_coords'][0, 1:2]
    lig_T = torch.cat([torch.zeros(lig_T.shape[0], lig_T.shape[1], 4), lig_T], dim=-1)[None] / config['position_scale']

    perf_lddt = torch.zeros(config['loss']['lddt_rec_num_bins'])
    perf_lddt[-1] = 10.0

    tmp = docker.lddt_loss(
        {'ground_truth': gt_dict},
        {
            'rec_T': rec_T,
            'lig_T': lig_T,
            'rec_lddt': perf_lddt.tile(1, 1, 2, 1),
            'lig_lddt': perf_lddt.tile(1, 1, 4, 1)
        },
        config
    )
    assert tmp['lig_best_mask_id_per_traj'][0] == 1


test()
