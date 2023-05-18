import torch
from typing import Dict, Optional, Tuple

def compute_plddt(logits):
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device='cuda:0')
  probs = torch.nn.functional.softmax(logits, dim=-1)
  predicted_lddt_ca = torch.sum(probs * bin_centers[None, ...], dim=-1)
  return predicted_lddt_ca * 100

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers

def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )

def compute_predicted_aligned_error(
    logits: torch.Tensor,
    breaks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=breaks,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }

def predicted_tm_score(
    logits: torch.Tensor,
    breaks: torch.Tensor,
    pair_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

    bin_centers = _calculate_bin_centers(breaks)

    if pair_weights is None:
        pair_weights = torch.ones((logits.shape[0], logits.shape[0]), dtype=float, device=logits.device)
    # here we implicitly assume that residue weights are binary 0/1 values
    num_res = int(torch.sum(torch.amax(pair_weights,-1)))
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8
    if d0 < 0.5:
        d0 = 0.02*num_res
    # Convert logits to probs.
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    # E_distances tm(distance).
    predicted_tm_term = torch.sum(probs * tm_per_bin, axis=-1)

    predicted_tm_term *= pair_weights

    normed_residue_mask = pair_weights / (1e-8 + torch.sum(
        pair_weights, dim=-1, keepdims=True))
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * torch.amax(pair_weights, dim=-1)
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def __get_contact_map(atom_positions, atom_masks, asym_ids, cutoff = 4.5):
    # first, get a rough contact map based on CA atoms
    ca_pos = atom_positions[:,1,:]
    ca_dist_map = torch.sum((ca_pos[None, :] - ca_pos[:, None])**2, dim=-1)
    filt_asym_ids = asym_ids[None,:] != asym_ids[:, None]
    prefilt_pairs = torch.where((ca_dist_map < 20*20) & filt_asym_ids)
    contact_map = torch.zeros((atom_positions.shape[0], atom_positions.shape[0]), device=atom_positions.device)

    # go through the identified pairs and refine distance using all atoms
    dist_sq = torch.sum((atom_positions[prefilt_pairs[0]][:, None, :, :]
                - atom_positions[prefilt_pairs[1]][:, :, None, :])**2, dim=-1)
    filt_atom_mask = (atom_masks[prefilt_pairs[0]][:, None, :] < 0.5) | (atom_masks[prefilt_pairs[1]][:, :, None] < 0.5)
    dist_sq += 100500 * filt_atom_mask

    if dist_sq.shape[0] != 0:
        dist_sq = torch.amin(dist_sq, dim=(-1,-2))
        final_pairs = (prefilt_pairs[0][torch.where(dist_sq < cutoff*cutoff)],
                       prefilt_pairs[1][torch.where(dist_sq < cutoff*cutoff)])
        contact_map[final_pairs] = 1.0

    #print('diff:', (contact_map - contact_map2).sum())

    return contact_map

def get_confidence_metrics(prediction_result, asym_ids, multimer_mode=True):
    """
    Post processes prediction_result to get confidence metrics.
    Populates 'plddt', 'ptm', 'iptm', 'ranking_confidence' and
    'multimer_confidence' fields.
    The main difference from the standard af2 function is the
    passing of gloabal ASYM_IDs variable to predict_tm_score()
    """
    confidence_metrics = {}
    if 'predicted_lddt' in prediction_result.keys():
        confidence_metrics['plddt'] = compute_plddt(
            prediction_result['predicted_lddt']['logits'])
    num_chains = len(set(asym_ids))
    if 'predicted_aligned_error' in prediction_result:
        confidence_metrics.update(compute_predicted_aligned_error(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks']))
        confidence_metrics['ptm'] = predicted_tm_score(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks'])
        # Save raw pae to dump it to file later
        confidence_metrics['raw_pae'] = prediction_result['predicted_aligned_error']

        if num_chains > 1: #multimer_mode:
            #print(global_var.ASYM_IDs)
            mask_interchain = asym_ids[:, None] != asym_ids[None, :]
            atom_positions = prediction_result['final_all_atom']
            atom_masks = prediction_result['final_atom_mask']
            ca_dist_mask = __get_contact_map(atom_positions,
                                             atom_masks,
                                             asym_ids)
            mask_interface =  ca_dist_mask * mask_interchain

            # Compute the ipTM only for the multimer model.
            confidence_metrics['iptm'] = predicted_tm_score(
                logits=prediction_result['predicted_aligned_error']['logits'],
                breaks=prediction_result['predicted_aligned_error']['breaks'],
                pair_weights = mask_interface)
            confidence_metrics['multimer_confidence'] = (
                0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])
            confidence_metrics['ranking_confidence'] = (
                0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

            asym_id_set = sorted(list(set(asym_ids)))
            interfaces = {}
            for i, chain1 in enumerate(asym_id_set):
                resi_i = (asym_ids==chain1)
                mask_ii = resi_i[:, None] * resi_i[None, :]
                for j, chain2 in enumerate(asym_id_set[i+1:]):
                    resi_j = (asym_ids==chain2)
                    mask_jj = resi_j[:, None] * resi_j[None, :]

                    resi_ij = resi_i | resi_j
                    mask_ij_full = resi_ij[:, None] * resi_ij[None, :]

                    mask_ij_interchain = mask_ij_full ^ (mask_ii | mask_jj)
                    mask_ij_interface  = ca_dist_mask * mask_ij_interchain

                    ptm_ij = predicted_tm_score(
                        logits=prediction_result['predicted_aligned_error']['logits'],
                        breaks=prediction_result['predicted_aligned_error']['breaks'],
                        pair_weights = mask_ij_full)
                    iptm_ij = predicted_tm_score(
                        logits=prediction_result['predicted_aligned_error']['logits'],
                        breaks=prediction_result['predicted_aligned_error']['breaks'],
                        pair_weights = mask_ij_interface)
                    multimer_conf_ij = 0.8*iptm_ij + 0.2*ptm_ij

                    interface_ij = {}
                    interface_ij['ptm'] = ptm_ij
                    interface_ij['iptm'] = iptm_ij
                    interface_ij['multimer_confidence'] = multimer_conf_ij
                    #print(chain1, chain2, ptm_ij, iptm_ij, multimer_conf_ij)
                    interfaces[(chain1, chain2)] = interface_ij
            confidence_metrics['interfaces'] = interfaces

    # Monomer models use mean pLDDT for model ranking.
    if (num_chains == 1) and ('plddt' in confidence_metrics.keys()):
        confidence_metrics['ranking_confidence'] = torch.mean(confidence_metrics['plddt'])

    return confidence_metrics 

