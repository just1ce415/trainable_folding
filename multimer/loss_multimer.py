import numpy as np
import torch
import torch.nn as nn
from alphadock import residue_constants
from typing import Dict, Optional, Tuple, List
from multimer import rigid, all_atom_multimer
from multimer.utils.tensor_utils import masked_mean
import math

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def masked_msa_loss(out, batch):
    errors = softmax_cross_entropy(
        out['masked_msa'], torch.nn.functional.one_hot(batch['true_msa'], num_classes=23)
    )
    loss = errors * batch['bert_mask']
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = 1e-8 + torch.sum(scale * batch['bert_mask'], dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale
    return loss

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score

def lddt_loss(out, batch, config):
    pred_all_atom_pos = out['final_all_atom']
    true_all_atom_pos = batch['all_atom_positions']
    all_atom_mask = batch['all_atom_mask']
    score = lddt(pred_all_atom_pos[...,1,:], true_all_atom_pos[..., 1, :], all_atom_mask[...,1:2])
    score = score.detach()
    no_bins = config['predicted_lddt']['num_bins']
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )
    logits = out['predicted_lddt']['logits']
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    errors = torch.unsqueeze(errors, dim=-1)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
            1e-10 + torch.sum(all_atom_mask, dim=-1)
    )
    loss = loss * (
            (batch['resolution'] >= config['predicted_lddt']['min_resolution']) & (batch['resolution'] <= config['predicted_lddt']['max_resolution'])
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    # lddt for new residue
    new_res_lddt = score[:, -1] * 100

    del errors, score
    return loss, new_res_lddt

def distogram_loss(out, batch, config):
    logits = out['distogram']['logits']
    bin_edges = out['distogram']['bin_edges']
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']
    sq_breaks = bin_edges ** 2
    dists = torch.sum(
        (positions[..., None, :] - positions[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )
    true_bins = torch.sum(dists > sq_breaks, dim=-1)
    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, config['distogram']['num_bins']),
    )
    square_mask = mask[..., None] * mask[..., None, :]
    denom = 1e-6 + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)
    del positions, mask, dists, true_bins, errors, square_mask
    return mean

def experimentally_resolved_loss(out, batch, config):
    logits = out['experimentally_resolved']
    atom_exists = out['final_atom_mask'] * batch['seq_mask'][0][:,None]
    all_atom_mask = batch['all_atom_mask']
    errors = sigmoid_cross_entropy(logits, all_atom_mask)
    loss = torch.sum(errors * atom_exists, dim=-1)
    loss = loss / (1e-8 + torch.sum(atom_exists, dim=(-1, -2)))
    loss = torch.sum(loss, dim=-1)
    if(config['experimentally_resolved']['filter_by_resolution']):
        loss = loss * (
                (batch['resolution'] >= config['experimentally_resolved']['min_resolution']) & (batch['resolution'] <= config['experimentally_resolved']['max_resolution'])
        )
    loss = torch.mean(loss)

    return loss

def compute_atom14_gt(aatype, all_atom_positions, all_atom_mask, pred_pos):
    gt_positions = all_atom_multimer.atom37_to_atom14(all_atom_positions, aatype)
    gt_mask = all_atom_multimer.atom37_to_atom14(all_atom_mask, aatype)
    return gt_positions, gt_mask

def get_alt_atom14(aatype, residx_atom14_gt_positions, residx_atom14_gt_mask):
    restype_3 = [residue_constants.restype_1to3[res] for res in residue_constants.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=residx_atom14_gt_mask.dtype,
            device=residx_atom14_gt_mask.device,
        )
        for res in restype_3
    }
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            14, device=residx_atom14_gt_mask.device
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = residue_constants.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = residx_atom14_gt_mask.new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix
    
    renaming_matrices = torch.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[aatype].type(torch.float32)

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform
    )
    #protein["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform
    )
    #protein["atom14_alt_gt_exists"] = alternative_gt_mask
    restype_atom14_is_ambiguous = residx_atom14_gt_mask.new_zeros((22, 14))
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[residue_constants.restype_3to1[resname]]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name1
            )
            atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name2
            )
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    atom14_atom_is_ambiguous = restype_atom14_is_ambiguous[aatype]
    return alternative_gt_positions, alternative_gt_mask, atom14_atom_is_ambiguous

def compute_renamed_ground_truth(
    gt_positions,
    alt_gt_positions,
    atom_is_ambiguous,
    gt_mask,
    alt_gt_mask,
    atom14_pred_positions,
    eps=1e-10,
):
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = gt_positions
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = alt_gt_positions
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = gt_mask
    atom14_atom_is_ambiguous = atom_is_ambiguous
    #print(atom14_gt_exists.shape, atom14_atom_is_ambiguous.shape, atom14_gt_exists.shape)

    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * alt_gt_mask

    return alt_naming_is_better, renamed_atom14_gt_positions, renamed_atom14_gt_mask

def compute_fape(
    pred_frames: rigid.Rigid,
    target_frames: rigid.Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    l1_clamp_distance: float,
    length_scale=20.0,
    eps=1e-4,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
    mask = frames_mask[..., None] * positions_mask[..., None, :]

    if pair_mask is not None:
        mask = mask * pair_mask
    
    normalization_factor = torch.sum(mask, dim=(-1, -2))
    normed_error = torch.sum(normed_error, dim=(-2, -1)) / (eps + normalization_factor)

    return normed_error

def backbone_loss(
    gt_rigid: rigid.Rigid,
    gt_mask: torch.Tensor,
    traj: torch.Tensor,
    pair_mask: torch.Tensor,
    config
):
    pred_rigid = rigid.Rigid.from_tensor_7(traj)
    pred_rigid = rigid.Rigid(
        rigid.Rotation(rot_mats=pred_rigid.get_rots().get_rot_mats(), quats=None),
        pred_rigid.get_trans(),
    )
    fape = compute_fape(
            pred_rigid,
            gt_rigid,
            gt_mask,
            pred_rigid.get_trans(),
            gt_rigid.get_trans(),
            gt_mask,
            pair_mask,
            l1_clamp_distance=config['atom_clamp_distance'],
            length_scale=config['loss_unit_distance']
            )
    fape_loss = torch.mean(fape)
    return fape_loss
    
def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    config
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = rigid.Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = rigid.Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        None,
        l1_clamp_distance=config['atom_clamp_distance'],
        length_scale=config['loss_unit_distance'],
    )

    return fape

def get_renamed_chi_angles(aatype: torch.Tensor,
                           chi_angles: torch.Tensor,
                           alt_is_better: torch.Tensor
                           ) -> torch.Tensor:
    chi_angle_is_ambiguous = chi_angles.new_tensor(residue_constants.chi_pi_periodic)[aatype].type(torch.float32)
    alt_chi_angles = chi_angles + math.pi * chi_angle_is_ambiguous
    alt_chi_angles = alt_chi_angles - 2 * math.pi * (alt_chi_angles > np.pi).type(torch.float32)
    alt_is_better = alt_is_better[...,:, None]
    return (1. - alt_is_better) * chi_angles + alt_is_better * alt_chi_angles

def supervised_chi_loss(
    sequence_mask: torch.Tensor,
    target_chi_mask: torch.Tensor,
    aatype: torch.Tensor,
    target_chi_angles: torch.Tensor,
    pred_angles: torch.Tensor,
    unnormed_angles: torch.Tensor,
    config,
    eps=1e-6):
    chi_mask = target_chi_mask.type(torch.float32)
    pred_angles = pred_angles[:, :, 3:]
    residue_type_one_hot = torch.nn.functional.one_hot(aatype, residue_constants.restype_num + 1)[None].type(torch.float32)
    chi_pi_periodic = torch.einsum('ijk, kl->ijl', residue_type_one_hot,pred_angles.new_tensor(residue_constants.chi_pi_periodic))
    true_chi = target_chi_angles[None]
    sin_true_chi = torch.sin(true_chi)
    cos_true_chi = torch.cos(true_chi)
    sin_cos_true_chi = torch.stack([sin_true_chi, cos_true_chi], dim=-1)

    shifted_mask = (1.0 - 2.0 * chi_pi_periodic)[..., None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi
    sq_chi_error = torch.sum((sin_cos_true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (sin_cos_true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_loss = masked_mean(chi_mask[None], sq_chi_error, dim=(-1, -2, -3))
    angle_norm = torch.sqrt(
        torch.sum(unnormed_angles ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.)
    angle_norm_loss = masked_mean(sequence_mask[None, :, None], norm_error, dim=(-1,-2,-3))
    loss = config['chi_weight']*sq_chi_loss + config['angle_norm_weight']*angle_norm_loss
    return loss

def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    residue_index: torch.Tensor,
    aatype: torch.Tensor,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = ((residue_index[..., 1:] - residue_index[..., :-1]) == 1.0).type(torch.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[..., 1:] == residue_constants.restype_order["P"]).type(torch.float32)
    gt_length = (
        1.0 - next_is_proline
    ) * residue_constants.between_res_bond_length_c_n[0] 
    + next_is_proline * residue_constants.between_res_bond_length_c_n[1]
    gt_stddev = (
        1.0 - next_is_proline
    ) * residue_constants.between_res_bond_length_stddev_c_n[0] 
    + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }

def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2).long(), num_classes=14
    )
    #c_one_hot = c_one_hot.reshape(
    #    *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    #)
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0).long(), num_classes=14
    )
    #n_one_hot = n_one_hot.reshape(
     #   *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    #)
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = ((residue_index[:, None] + 1) == residue_index[None, :])
    neighbour_mask = neighbour_mask & (asym_id[:, None] == asym_id[None, :])
    neighbour_mask = neighbour_mask[..., None, None]
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx).long()
    #cys_sg_idx = cys_sg_idx.reshape(
    #    *((1,) * len(residue_index.shape[:-1])), 1
    #).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }

def find_structural_violations(
    aatype: torch.Tensor,
    residue_index: torch.Tensor,
    mask: torch.Tensor,
    pred_positions: torch.Tensor,
    asym_id: torch.Tensor,
    config
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=pred_positions,
        pred_atom_mask=mask.type(torch.float32),
        residue_index=residue_index.type(torch.float32),
        aatype=aatype,
        tolerance_factor_soft=config['violation_tolerance_factor'],
        tolerance_factor_hard=config['violation_tolerance_factor'],
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]
    
    residx_atom14_to_atom37 = aatype.new_tensor(all_atom_multimer._make_restype_atom14_to_atom37())[aatype]
    atomtype_radius = pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = mask * atomtype_radius[residx_atom14_to_atom37]


    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=pred_positions,
        atom14_atom_exists=mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        asym_id=asym_id,
        overlap_tolerance_soft=config['clash_overlap_tolerance'],
        overlap_tolerance_hard=config['clash_overlap_tolerance'],
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config['clash_overlap_tolerance'],
        bond_length_tolerance_factor=config['violation_tolerance_factor'],
    )
    atom14_dists_lower_bound = pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[aatype]
    atom14_dists_upper_bound = pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[aatype]
    residue_violations = within_residue_violations(
        atom14_pred_positions=pred_positions,
        atom14_atom_exists=mask,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # ()
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }

def structural_violation_loss(mask: torch.Tensor,
                              violations,
                              config
                              ):
    num_atoms = torch.sum(mask).type(torch.float32) + 1e-6
    between_residues = violations['between_residues']
    within_residues = violations['within_residues']
    return (config['structural_violation_loss_weight'] *
          (between_residues['bonds_c_n_loss_mean'] +
           between_residues['angles_ca_c_n_loss_mean']  +
           between_residues['angles_c_n_ca_loss_mean'] +
           torch.sum(between_residues['clashes_per_atom_loss_sum'] +
                   within_residues['per_atom_loss_sum']) / num_atoms
           ))

def tm_loss(
    logits,
    boundaries,
    predicted_rigid_tensor,
    backbone_rigid,
    backbone_rigid_mask,
    resolution,
    config,
    eps=1e-8,
):
    pred_affine = rigid.Rigid.from_tensor_7(predicted_rigid_tensor)
    min_resolution = config['predicted_aligned_error']['min_resolution']
    max_resolution = config['predicted_aligned_error']['max_resolution']
    num_bins = config['predicted_aligned_error']['num_bins']

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1
    )

    sq_diff = sq_diff.detach()

    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, num_bins)
    )

    square_mask = (
        backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the loss dimension
    loss = torch.mean(loss)

    return loss
def structure_loss(out, batch, config):
    aatype = batch['aatype'][0]
    all_atom_positions = batch['all_atom_positions'][0]
    all_atom_mask = batch['all_atom_mask'][0]
    seq_mask = batch['seq_mask'][0]
    residue_index = batch['residue_index'][0]
    n, ca, c = [residue_constants.atom_order[a] for a in ["N", "CA", "C"]]
    gt_rigid = rigid.Rigid.make_transform_from_reference(
        n_xyz=all_atom_positions[..., n, :],
        ca_xyz=all_atom_positions[..., ca, :],
        c_xyz=all_atom_positions[..., c, :],
        eps=1e-20,
        )
    gt_affine_mask = (all_atom_mask[:, n] * all_atom_mask[:, ca] * all_atom_mask[:, c]).float()
    chi_angles, chi_mask = all_atom_multimer.compute_chi_angles(
            all_atom_positions,
            all_atom_mask,
            aatype)
    pred_mask = all_atom_multimer.atom_14_mask(aatype.long())
    pred_mask = pred_mask * seq_mask[:, None]
    pred_positions = out['final_atom14_positions']
    gt_positions, gt_mask = compute_atom14_gt(aatype.long(), all_atom_positions, all_atom_mask, pred_positions)
    alt_gt_positions, alt_gt_mask, atom_is_ambiguous = get_alt_atom14(aatype.long(), gt_positions.type(torch.float32), gt_mask.type(torch.float32))
    alt_naming_is_better, gt_positions, gt_mask = compute_renamed_ground_truth(gt_positions, alt_gt_positions, atom_is_ambiguous, gt_mask, alt_gt_mask, pred_positions)
    asym_id = batch['asym_id'][0]
    intra_chain_mask = asym_id[:, None] == asym_id[None, :]
    traj = out['struct_out']['frames'][:,0,...]

    intra_chain_bb_loss = backbone_loss(gt_rigid, gt_affine_mask, traj, intra_chain_mask, config['structure_module']['intra_chain_fape'])
    interface_bb_loss = backbone_loss(gt_rigid, gt_affine_mask, traj, ~intra_chain_mask, config['structure_module']['interface_fape'])
    rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_group_exists, rigidgroups_group_is_ambiguous, rigidgroups_alt_gt_frames = all_atom_multimer.atom37_to_frames(aatype.long(), all_atom_positions, all_atom_mask)

    sc_loss = sidechain_loss(out['struct_out']['sc_frames'][:,0,...],out['struct_out']['atom_pos'][:,0,...],rigidgroups_gt_frames, rigidgroups_alt_gt_frames, rigidgroups_gt_exists, gt_positions, gt_mask, alt_naming_is_better, config['structure_module']['sidechain'])

    gt_chi_angles = get_renamed_chi_angles(aatype.long(), chi_angles, alt_naming_is_better)

    sup_chi_loss = supervised_chi_loss(seq_mask,
                                        chi_mask,
                                        aatype.long(),
                                        gt_chi_angles,
                                        out['struct_out']['angles_sin_cos'][:,0,...],
                                        out['struct_out']['unnormalized_angles_sin_cos'][:,0,...],
                                        config['structure_module'],)

    violations = find_structural_violations(
            aatype.long(),
            residue_index,
            pred_mask,
            pred_positions,
            asym_id,
            config['structure_module'])
    violation_loss = structural_violation_loss(pred_mask, violations, config['structure_module'])
    return 0.5*(intra_chain_bb_loss + interface_bb_loss) + 0.5*sc_loss + sup_chi_loss + violation_loss, gt_rigid, gt_affine_mask

def masked_msa_loss(out, batch):
    errors = softmax_cross_entropy(
        out['msa_head'], torch.nn.functional.one_hot(batch['true_msa'].long(), num_classes=23)
    )
    loss = errors * batch['bert_mask']
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = 1e-8 + torch.sum(scale * batch['bert_mask'], dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss
class MultimerLoss(nn.Module):
    def __init__(self, config):
        self.config = config
    def forward(self, out, batch):
        #masked_loss = masked_msa_loss(out, batch)
        lddt_l = lddt_loss(out, batch, self.config)
        #dist_loss = distogram_loss(out, batch, self.config)
        #exp_res_loss = experimentally_resolved_loss(out, batch, self.config)
        return lddt_l
