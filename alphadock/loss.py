import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from alphadock import r3
from alphadock import quat_affine
from alphadock import all_atom
from alphadock import utils
from alphadock import lddt
from alphadock import violations
from alphadock import residue_constants


def total_loss(batch, struct_out, final_all_atom, config):
    # process predictions
    rec_traj = struct_out['rec_T'][0]  # (N_traj, N_res, 7)
    num_traj = rec_traj.shape[0]

    rec_final_pred_frames = final_all_atom['frames']  # r3.Rigids (N, 8)
    rec_final_atom14_pred_coords_vecs = final_all_atom['atom_pos']  # r3.Vecs (N, 14)
    rec_final_atom14_pred_coords_tensor = final_all_atom['atom_pos_tensor']  # Tensor (N, 14, 3)
    rec_final_atom14_pred_mask = batch['target']['rec_atom14_atom_exists'][0]  # Tensor (N, 14)
    #lig_final_pred_coords_vecs = r3.vecs_from_tensor(lig_final_pred_coords_tensor)   # Vecs (Natoms)

    # process ground truth
    # (N_traj, N_res)
    gt_bb_frames = r3.rigids_from_tensor_flat12(batch['ground_truth']['gt_rigidgroups_gt_frames'][0, :, 0, :].repeat(num_traj, 1, 1))
    gt_bb_mask = batch['ground_truth']['gt_rigidgroups_gt_exists'][0, :, 0].repeat(num_traj, 1)

    pred_bb_frames = r3.rigids_from_quataffine(quat_affine.QuatAffine.from_tensor(rec_traj))

    fape_clamp_distance = config['loss']['fape_clamp_distance'] if batch['ground_truth'].get('clamp_fape', 1) > 0 else None

    loss_bb_rec_rec = all_atom.frame_aligned_point_error(
        pred_bb_frames,
        gt_bb_frames,
        gt_bb_mask,
        pred_bb_frames.trans,
        gt_bb_frames.trans,
        gt_bb_mask,
        config['loss']['fape_loss_unit_distance'],
        fape_clamp_distance
    )

    # compute all atom FAPE
    renamed = all_atom.compute_renamed_ground_truth(
        atom14_gt_positions=batch['ground_truth']['gt_atom14_coords'][0],
        atom14_alt_gt_positions=batch['ground_truth']['gt_atom14_coords_alt'][0],
        atom14_atom_is_ambiguous=batch['ground_truth']['gt_atom14_atom_is_ambiguous'][0],
        atom14_gt_exists=batch['ground_truth']['gt_atom14_has_coords'][0],
        atom14_alt_gt_exists=batch['ground_truth']['gt_atom14_has_coords_alt'][0],
        atom14_pred_positions=rec_final_atom14_pred_coords_tensor,
        atom14_atom_exists=rec_final_atom14_pred_mask
    )

    alt_naming_is_better = renamed['alt_naming_is_better']

    # (N, 8, 12)
    renamed_gt_frames_flat12 = (
        (1. - alt_naming_is_better[:, None, None])
        * batch['ground_truth']['gt_rigidgroups_gt_frames'][0]
        + alt_naming_is_better[:, None, None]
        * batch['ground_truth']['gt_rigidgroups_alt_gt_frames'][0]
    )

    renamed_gt_frames_flat = r3.rigids_from_tensor_flat12(renamed_gt_frames_flat12.reshape(-1, 12))
    renamed_gt_frames_mask_flat = batch['ground_truth']['gt_rigidgroups_gt_exists'].flatten()
    renamed_gt_coords_flat = r3.vecs_from_tensor(renamed['renamed_atom14_gt_positions'].reshape(-1, 3))
    renamed_gt_coords_mask_flat = renamed['renamed_atom14_gt_exists'].flatten()
    rec_final_pred_frames_flat = r3.apply_tree_rigids(lambda x: x.flatten(), rec_final_pred_frames)
    rec_final_atom14_pred_coords_flat = r3.apply_tree_vecs(lambda x: x.flatten(), rec_final_atom14_pred_coords_vecs)

    # Compute frame_aligned_point_error score for the final layer.
    loss_aa_rec_rec = all_atom.frame_aligned_point_error(
        rec_final_pred_frames_flat,
        renamed_gt_frames_flat,
        renamed_gt_frames_mask_flat,
        rec_final_atom14_pred_coords_flat,
        renamed_gt_coords_flat,
        renamed_gt_coords_mask_flat,
        config['loss']['fape_loss_unit_distance'],
        fape_clamp_distance
    )
    loss_chi = torsion_loss(batch, struct_out)

    lddt_vals = lddt_calc(batch, struct_out)
    lddt_loss_rec_rec = lddt_loss_calc(
        lddt_pred=struct_out['rec_lddt'][0],
        lddt_true=lddt_vals['rec_rec_lddt_true_per_residue'],
        mask=batch['ground_truth']['gt_atom14_has_coords'][0, :, 1],
        bin_size=config['loss']['lddt_rec_bin_size'],
        num_bins=config['StructureModule']['StructureModuleIteration']['PredictRecLDDT']['num_bins']
    )

    gly_index = residue_constants.restype_order['G']
    gt_cbeta_index = [4 if x != gly_index else 1 for x in batch['ground_truth']['gt_aatype'][0]]
    gt_rec_cbeta = batch['ground_truth']['gt_atom14_coords'][0, range(len(gt_cbeta_index)), gt_cbeta_index]
    gt_rec_cbeta_mask = batch['ground_truth']['gt_atom14_has_coords'][0, range(len(gt_cbeta_index)), gt_cbeta_index]
    loss_rr_dmat_pred = distogram_loss(
        struct_out['distogram']['rr'][0],
        gt_rec_cbeta,
        gt_rec_cbeta,
        gt_rec_cbeta_mask,
        gt_rec_cbeta_mask,
        config['StructureModule']['PredictDistogram']['rec_min_dist'],
        config['StructureModule']['PredictDistogram']['rec_max_dist']
    )

    loss_total = \
        loss_bb_rec_rec.mean() * config['loss']['loss_bb_rec_rec_weight'] + \
        loss_aa_rec_rec * config['loss']['loss_aa_rec_rec_weight'] + \
        loss_chi['chi_loss'].mean() * config['loss']['loss_chi_value_weight'] + \
        loss_chi['norm_loss'].mean() * config['loss']['loss_chi_norm_weight'] + \
        lddt_loss_rec_rec * config['loss']['loss_rec_rec_lddt_weight'] + \
        loss_rr_dmat_pred * config['loss']['loss_pred_dmat_rr_weight']

    violation_loss = None
    if config['loss']['loss_violation_weight'] > 0:
        violations_dict = violations.find_structural_violations(
            batch,
            rec_final_atom14_pred_coords_tensor,
            config
        )
        violation_loss = structural_violation_loss(batch, violations_dict)
        loss_total += config['loss']['loss_violation_weight'] * violation_loss

    out_dict = {
        'loss_total': loss_total,  # Scalar
        'loss_torsions': loss_chi,  # chi_loss: (Traj), norm_loss: (Traj)
        'loss_fape': {
            'loss_bb_rec_rec': loss_bb_rec_rec,  # (Traj)
            'loss_aa_rec_rec': loss_aa_rec_rec,  # Scalar
        },
        'lddt_values': lddt_vals, # rec_rec_lddt_true: (Traj, N), lig_rec_lddt_true: (Traj, N), lig_best_mask_per_traj: (Traj, N), lig_best_mask_id_per_traj: (Traj)
        'lddt_loss_rec_rec': lddt_loss_rec_rec,  # Scalar

        'loss_pred_dmat': {
            'rr': loss_rr_dmat_pred,
        }
    }

    if violation_loss is not None:
        out_dict['violations'] = {
            'loss': violation_loss
        }
        out_dict['violations'].update(violations_dict)

    return out_dict


def lig_dmat_loss(
        lig_traj,  # (Ntraj, natoms, 3)
        lig_gt_coords,  # (nsym, natoms, 3)
        lig_gt_mask,   # (nsym, natoms)
        clip=10,
        epsilon=1e-6,
        square=False
):
    '''
    Calculates masked mean difference between squared distance matrices
    of ground truth ligand and predicted trajectories.
    '''
    dmat_gt = torch.square(lig_gt_coords[:, None] - lig_gt_coords[:, :, None]).sum(-1)
    dmat_pred = torch.square(lig_traj[:, None] - lig_traj[:, :, None]).sum(-1)

    if not square:
        dmat_gt = torch.sqrt(dmat_gt + epsilon)
        dmat_pred = torch.sqrt(dmat_pred + epsilon)
    else:
        clip = clip**2

    delta = torch.clip(torch.abs(dmat_pred[:, None] - dmat_gt[None]), 0, clip)
    mask_2d = lig_gt_mask[:, None] * lig_gt_mask[:, :, None]

    delta *= mask_2d[None]
    losses = (delta / clip).sum([-2, -1]) / (mask_2d.sum([-2, -1])[None] + epsilon)
    losses = losses.min(-1).values  # select best ligand symmtry
    return losses  # (Ntraj)


def torsion_loss(batch, struct_out):
    eps = 1e-6

    pred_torsions_unnorm = struct_out['rec_torsions'][0]  # (Ntraj, Nres, 7, 2) - unnormalized
    gt_torsions = batch['ground_truth']['gt_torsions_sin_cos'][0]   # (Nres, 7, 2)
    gt_torsions_alt = batch['ground_truth']['gt_torsions_sin_cos_alt'][0]   # (Nres, 7, 2)
    gt_torsions_mask = batch['ground_truth']['gt_torsions_mask'][0]   # (Nres, 7)

    pred_torsions_norm = all_atom.l2_normalize(pred_torsions_unnorm, axis=-1)
    chi_squared = utils.squared_difference(pred_torsions_norm, gt_torsions[None]).sum(-1)
    chi_squared_alt = utils.squared_difference(pred_torsions_norm, gt_torsions_alt[None]).sum(-1)
    chi_loss = (torch.minimum(chi_squared, chi_squared_alt) * gt_torsions_mask[None]).sum((1, 2)) / (gt_torsions_mask.sum() + eps)

    norm_loss = torch.abs(torch.sqrt(torch.sum(torch.square(pred_torsions_unnorm), dim=-1) + eps) - 1.0)
    norm_loss *= gt_torsions_mask[None]
    norm_loss = norm_loss.sum((1, 2)) / (gt_torsions_mask.sum() + eps)

    return {
        'chi_loss': chi_loss,
        'norm_loss': norm_loss
    }


def distogram_loss(pred, gt_coords_a, gt_coords_b, gt_mask_a, gt_mask_b, min_dist, max_dist):
    assert len(pred.shape) == 3, pred.shape
    assert len(gt_coords_a.shape) == 2 and gt_coords_a.shape[-1] == 3, gt_coords_a.shape
    assert len(gt_coords_b.shape) == 2 and gt_coords_b.shape[-1] == 3, gt_coords_b.shape
    assert len(gt_mask_a.shape) == 1, gt_mask_a.shape
    assert len(gt_mask_b.shape) == 1, gt_mask_b.shape

    dmat_gt = torch.sqrt(torch.square(gt_coords_a[None, :, :] - gt_coords_b[:, None, :]).sum(-1) + 10e-10)
    mask_2d = gt_mask_a[:, None] * gt_mask_b[None, :]
    num_bins = pred.shape[-1]
    dmat_labels = utils.dmat_to_dgram(dmat_gt, min_dist, max_dist, num_bins)[0]
    dmat_labels[mask_2d.flatten() < 1] = -100
    loss = F.cross_entropy(pred.reshape(-1, num_bins), dmat_labels, ignore_index=-100)
    return loss


def lddt_calc(batch, struct_out):
    rec_pred_coords = struct_out['rec_T'][0, :, :, -3:]   # (Ntraj, Nres, 3)
    rec_true_coords = batch['ground_truth']['gt_atom14_coords'][0, :, 1]   # (Nres, 3)
    rec_true_mask = batch['ground_truth']['gt_atom14_has_coords'][0, :, 1]   # (Nres)

    num_traj = rec_pred_coords.shape[0]

    rec_rec_lddt = lddt.lddt(
        rec_pred_coords,
        rec_pred_coords,
        rec_true_coords[None],
        rec_true_coords[None],
        rec_true_mask[None, :, None],
        rec_true_mask[None, :, None],
        per_residue=True,
        exclude_self=True
    )   # (Ntraj, Nres)

    rec_rec_lddt_total = lddt.lddt(
        rec_pred_coords,
        rec_pred_coords,
        rec_true_coords[None],
        rec_true_coords[None],
        rec_true_mask[None, :, None],
        rec_true_mask[None, :, None],
        per_residue=False,
        exclude_self=True
    )   # (Ntraj)

    out = {
        'rec_rec_lddt_true_per_residue': rec_rec_lddt,
        'rec_rec_lddt_true_total': rec_rec_lddt_total
    }

    return out


def lddt_loss_calc(
        lddt_pred,
        lddt_true,
        mask,
        bin_size,
        num_bins
):
    # TODO: decide on the proper range here
    # (Ntraj, Nres)
    lddt_label = torch.minimum(
        torch.div(lddt_true * 100, bin_size),
        torch.full_like(lddt_true, num_bins - 1)
    )

    # set masked residues and atoms to -100
    lddt_label = lddt_label * mask[None] + (mask[None] - 1) * 100

    # calc mean loss
    loss = F.cross_entropy(lddt_pred.flatten(end_dim=1), lddt_label.flatten().to(dtype=torch.long), ignore_index=-100, reduction='mean')
    return loss


def structural_violation_loss(batch, violations):
    """Computes loss for structural violations."""

    # Put all violation losses together to one large loss.
    num_rec_atoms = torch.sum(batch['target']['rec_atom14_atom_exists'][0])
    loss = (
            violations['between_residues']['bonds_c_n_loss_mean'] +
            violations['between_residues']['angles_ca_c_n_loss_mean'] +
            violations['between_residues']['angles_c_n_ca_loss_mean'] +
            (torch.sum(
                violations['between_residues']['clashes_per_atom_loss_sum'] +
                violations['within_residues']['per_atom_loss_sum']
            ) / (1e-6 + num_rec_atoms)) +
            violations['lig_rec']['clashes_mean_loss'] +
            violations['lig']['per_atom_loss_sum'].mean()
        )
    return loss


def _loss_dmat_checking():
    from alphadock.dataset import DockingDataset
    from alphadock.config import DATA_DIR
    ds = DockingDataset(DATA_DIR , 'train_split/train_12k.json')
    item = ds[0]
    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    lig_gt_mask = item['ground_truth']['gt_lig_has_coords'][0]
    lig_gt_coords = item['ground_truth']['gt_lig_coords'][0]
    print(lig_gt_coords)
    print(lig_gt_mask)

    lig_traj = lig_gt_coords[[0, 2]].clone()
    #lig_traj[1, 0, 0] += 10
    lig_traj[:, :] = lig_traj[0].mean(0)
    #lig_gt_mask[:] = 0

    print(lig_dmat_loss(lig_traj, lig_gt_coords, lig_gt_mask))


if __name__ == '__main__':
    pass
    #_loss_dmat_checking()
