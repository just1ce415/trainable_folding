import torch
import torch.nn.functional as F

from alphadock import r3
from alphadock import quat_affine
from alphadock import all_atom
from alphadock import utils
from alphadock import lddt


def total_loss(batch, struct_out, final_all_atom, config):
    # process predictions
    rec_traj = struct_out['rec_T'][0]  # (N_traj, N_res, 7)
    lig_traj = struct_out['lig_T'][0]  # (N_traj, N_atoms, 7)
    num_traj = rec_traj.shape[0]

    rec_final_pred_frames = final_all_atom['frames']  # r3.Rigids (N, 8)
    rec_final_atom14_pred_coords_vecs = final_all_atom['atom_pos']  # r3.Vecs (N, 14)
    rec_final_atom14_pred_coords_tensor = final_all_atom['atom_pos_tensor']  # Tensor (N, 14, 3)
    rec_final_atom14_pred_mask = batch['target']['rec_atom14_atom_exists'][0]  # Tensor (N, 14)

    lig_final_pred_coords_tensor = lig_traj[-1, :, -3:]  #   Tensor (Natoms, 3)
    #lig_final_pred_coords_vecs = r3.vecs_from_tensor(lig_final_pred_coords_tensor)   # Vecs (Natoms)

    # process ground truth
    # (N_traj, N_res)
    gt_bb_frames = r3.rigids_from_tensor_flat12(batch['ground_truth']['gt_rigidgroups_gt_frames'][0, :, 0, :].repeat(num_traj, 1, 1))
    gt_bb_mask = batch['ground_truth']['gt_rigidgroups_gt_exists'][0, :, 0].repeat(num_traj, 1)

    num_lig_symm = batch['ground_truth']['gt_lig_coords'].shape[1]
    lig_gt_mask = batch['ground_truth']['gt_lig_has_coords'][0]
    lig_gt_coords = batch['ground_truth']['gt_lig_coords'][0]

    pred_bb_frames = r3.rigids_from_quataffine(quat_affine.QuatAffine.from_tensor(rec_traj))

    '''tmpnum = 3
    loss_bb_rec_rec = all_atom.frame_aligned_point_error(
        r3.apply_tree_rigids(lambda x: x[:, :tmpnum], pred_bb_frames),
        r3.apply_tree_rigids(lambda x: x[:, :tmpnum], gt_bb_frames),
        gt_bb_mask[:, :tmpnum],
        r3.apply_tree_vecs(lambda x: x[:, :tmpnum], pred_bb_frames.trans),
        r3.apply_tree_vecs(lambda x: x[:, :tmpnum], gt_bb_frames.trans),
        gt_bb_mask[:, :tmpnum],
        config['loss']['fape_loss_unit_distance'],
        config['loss']['fape_clamp_distance']
    )'''

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

    loss_bb_rec_lig = all_atom.frame_aligned_point_error(
        r3.apply_tree_rigids(lambda x: x.repeat_interleave(num_lig_symm, dim=0), pred_bb_frames),
        r3.apply_tree_rigids(lambda x: x.repeat_interleave(num_lig_symm, dim=0), gt_bb_frames),
        gt_bb_mask.repeat_interleave(num_lig_symm, dim=0),
        r3.vecs_from_tensor(lig_traj[:, :, -3:].repeat_interleave(num_lig_symm, dim=0)),
        r3.vecs_from_tensor(lig_gt_coords.repeat(num_traj, 1, 1)),
        lig_gt_mask.repeat(num_traj, 1),
        config['loss']['fape_loss_unit_distance'],
        fape_clamp_distance
    )
    loss_bb_rec_lig = loss_bb_rec_lig.reshape(num_traj, num_lig_symm)

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

    loss_aa_rec_lig = all_atom.frame_aligned_point_error(
        r3.apply_tree_rigids(lambda x: x.repeat(num_lig_symm, 1), rec_final_pred_frames_flat),
        r3.apply_tree_rigids(lambda x: x.repeat(num_lig_symm, 1), renamed_gt_frames_flat),
        renamed_gt_frames_mask_flat.repeat(num_lig_symm, 1),
        r3.vecs_from_tensor(lig_final_pred_coords_tensor.repeat(num_lig_symm, 1, 1)),
        r3.vecs_from_tensor(lig_gt_coords),
        lig_gt_mask,
        config['loss']['fape_loss_unit_distance'],
        fape_clamp_distance
    )

    loss_chi = torsion_loss(batch, struct_out)

    lddt_vals = lddt_calc(batch, struct_out)
    if True:
        lddt_loss_rec_rec = lddt_loss_calc(
            lddt_pred=struct_out['rec_lddt'][0],
            lddt_true=lddt_vals['rec_rec_lddt_true_per_residue'],
            mask=batch['ground_truth']['gt_atom14_has_coords'][0, :, 1],
            bin_size=config['loss']['lddt_rec_bin_size'],
            num_bins=config['loss']['lddt_rec_num_bins']
        )
        lddt_loss_lig_rec = lddt_loss_calc(
            lddt_pred=struct_out['lig_lddt'][0],
            lddt_true=lddt_vals['lig_rec_lddt_true_per_atom'],
            mask=lddt_vals['lig_best_mask_per_traj'],
            bin_size=config['loss']['lddt_lig_bin_size'],
            num_bins=config['loss']['lddt_lig_num_bins']
        )

    #loss_aff = torch.tensor(0.0, dtype=rec_traj.dtype, device=rec_traj.device)
    loss_aff = None
    if 'gt_affinity_label' in batch['ground_truth']:
        loss_aff = F.cross_entropy(struct_out['lig_affinity'][:, batch['ground_truth']['gt_affinity_lig_id'][0]], batch['ground_truth']['gt_affinity_label'].flatten())

    loss_total = loss_bb_rec_rec.mean() * config['loss']['loss_bb_rec_rec_weight'] + \
        loss_bb_rec_lig.min(-1).values.mean() * config['loss']['loss_bb_rec_lig_weight'] + \
        loss_aa_rec_rec * config['loss']['loss_aa_rec_rec_weight'] + \
        loss_aa_rec_lig.min() * config['loss']['loss_aa_rec_lig_weight'] + \
        loss_chi['chi_loss'].mean() * config['loss']['loss_chi_value_weight'] + \
        loss_chi['norm_loss'].mean() * config['loss']['loss_chi_norm_weight'] + \
        lddt_loss_rec_rec * config['loss']['loss_rec_rec_lddt_weight'] + \
        lddt_loss_lig_rec * config['loss']['loss_lig_rec_lddt_weight']

    if loss_aff is not None:
        loss_total += loss_aff * config['loss']['loss_affinity_weight']

    out_dict = {
        'loss_total': loss_total,  # Scalar
        'loss_torsions': loss_chi,  # chi_loss: (Traj), norm_loss: (Traj)
        'loss_fape': {
            'loss_bb_rec_rec': loss_bb_rec_rec,  # (Traj)
            'loss_bb_rec_lig': loss_bb_rec_lig,  # (Traj, Symm)
            'loss_aa_rec_rec': loss_aa_rec_rec,  # Scalar
            'loss_aa_rec_lig': loss_aa_rec_lig,  # (Symm)
        },

        'lddt_values': lddt_vals, # rec_rec_lddt_true: (Traj, N), lig_rec_lddt_true: (Traj, N), lig_best_mask_per_traj: (Traj, N), lig_best_mask_id_per_traj: (Traj)
        'lddt_loss_rec_rec': lddt_loss_rec_rec,  # Scalar
        'lddt_loss_lig_rec': lddt_loss_lig_rec,  # Scalar
    }

    if loss_aff is not None:
        out_dict['loss_affinity'] = loss_aff  # Scalar

    return out_dict


def lig_lig_dmat_loss(
        lig_traj,  # (Ntraj, natoms, 3)
        lig_gt_coords,  # (nsym, natoms, 3)
        lig_gt_mask,   # (nsym, natoms, 3)
        clip=10
):
    pass


def torsion_loss(batch, struct_out):
    eps = 1e-6

    pred_torsions_unnorm = struct_out['rec_torsions'][0]  # (Ntraj, Nres, 7, 2) - unnormalized
    gt_torsions = batch['ground_truth']['gt_torsions_sin_cos'][0]   # (Nres, 7, 2)
    gt_torsions_alt = batch['ground_truth']['gt_torsions_sin_cos_alt'][0]   # (Nres, 7, 2)
    gt_torsions_mask = batch['ground_truth']['gt_torsions_mask'][0]   # (Nres, 7)

    pred_torsions_norm = all_atom.l2_normalize(pred_torsions_unnorm, axis=-1)
    chi_squared = utils.squared_difference(pred_torsions_norm, gt_torsions[None]).sum(-1)
    chi_squared_alt = utils.squared_difference(pred_torsions_norm, gt_torsions_alt[None]).sum(-1)
    chi_loss = (torch.minimum(chi_squared, chi_squared_alt) * gt_torsions_mask[None]).sum((1, 2)) / gt_torsions_mask.sum()

    norm_loss = torch.abs(torch.sqrt(torch.sum(torch.square(pred_torsions_unnorm), dim=-1) + eps) - 1.0)
    norm_loss *= gt_torsions_mask[None]
    norm_loss = norm_loss.sum((1, 2)) / gt_torsions_mask.sum()

    return {
        'chi_loss': chi_loss,
        'norm_loss': norm_loss
    }


def lddt_calc(batch, struct_out):
    rec_pred_coords = struct_out['rec_T'][0, :, :, -3:]   # (Ntraj, Nres, 3)
    lig_pred_coords = struct_out['lig_T'][0, :, :, -3:]   # (Ntraj, Natoms, 3)
    rec_true_coords = batch['ground_truth']['gt_atom14_coords'][0, :, 1]   # (Nres, 3)
    rec_true_mask = batch['ground_truth']['gt_atom14_has_coords'][0, :, 1]   # (Nres)
    lig_true_coords = batch['ground_truth']['gt_lig_coords'][0]   #  (Nsymm, Natoms, 3)
    lig_true_mask = batch['ground_truth']['gt_lig_has_coords'][0]   #  (Nsymm, Natoms)

    num_traj = rec_pred_coords.shape[0]
    num_symm = lig_true_coords.shape[0]

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

    # compute lddt for each ligand symm and select the best symm id for each traj
    lig_rec_lddt_for_mask_selection = lddt.lddt(
        lig_pred_coords.repeat_interleave(num_symm, dim=0),
        rec_pred_coords.repeat_interleave(num_symm, dim=0),
        lig_true_coords.repeat(num_traj, 1, 1),
        rec_true_coords[None].repeat(num_traj * num_symm, 1, 1),
        lig_true_mask[:, :, None].repeat(num_traj, 1, 1),
        rec_true_mask[None, :, None].repeat(num_traj * num_symm, 1, 1),
        per_residue=False,
        exclude_self=False
    )  # (Ntraj * Nsymm)

    lig_best_mask_id_per_traj = lig_rec_lddt_for_mask_selection.reshape(num_traj, num_symm).max(-1).indices
    lig_best_mask_per_traj = lig_true_mask[lig_best_mask_id_per_traj]

    # compute lddt using the best mask for each trajectory frame
    lig_rec_lddt = lddt.lddt(
        lig_pred_coords,
        rec_pred_coords,
        lig_true_coords[lig_best_mask_id_per_traj],
        rec_true_coords[None].repeat(num_traj, 1, 1),
        lig_best_mask_per_traj[:, :, None],
        rec_true_mask[None, :, None].repeat(num_traj, 1, 1),
        per_residue=True,
        exclude_self=False
    )  # (Ntraj, Natoms)

    # compute lddt using the best mask for each trajectory frame
    lig_rec_lddt_total = lddt.lddt(
        lig_pred_coords,
        rec_pred_coords,
        lig_true_coords[lig_best_mask_id_per_traj],
        rec_true_coords[None].repeat(num_traj, 1, 1),
        lig_best_mask_per_traj[:, :, None],
        rec_true_mask[None, :, None].repeat(num_traj, 1, 1),
        per_residue=False,
        exclude_self=False
    )  # (Ntraj)

    return {
        'rec_rec_lddt_true_per_residue': rec_rec_lddt,
        'lig_rec_lddt_true_per_atom': lig_rec_lddt,
        'rec_rec_lddt_true_total': rec_rec_lddt_total,
        'lig_rec_lddt_true_total': lig_rec_lddt_total,
        'lig_best_mask_per_traj': lig_best_mask_per_traj,
        'lig_best_mask_id_per_traj': lig_best_mask_id_per_traj
    }


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


if __name__ == '__main__':
    pass
