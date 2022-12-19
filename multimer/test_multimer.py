import sys
sys.path.insert(1, '../')
from alphadock import all_atom, residue_constants
import pickle
import numpy as np
import torch
import json, codecs
from multimer import modules_multimer, config_multimer, load_param_multimer, pipeline_multimer, pdb_to_template
import io
import os
from Bio.PDB import PDBParser
from multimer import rigid
from typing import Dict, Optional, Tuple

def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
        f.write(f'test.pred\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['aatype'][0].cpu(),
            out_dict['final_all_atom'].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')

def protein_to_pdb(aatype, atom_positions, residue_index, chain_index, atom_mask, b_factors):
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []
    residue_index = residue_index.astype(np.int32)
    chain_index = chain_index.astype(np.int32)
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
          f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
            atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
            residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                              chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

  # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'

def compute_plddt(logits):
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = torch.arange(0.5 * bin_width, 1.0, bin_width, device=logits.device)
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
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])
    bin_centers = _calculate_bin_centers(breaks)

    num_res = logits.shape[-2]
    clipped_num_res = max(num_res, 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    pair_mask = torch.ones((num_res, num_res), dtype=bool, device=logits.device)
    if interface:
        pair_mask *= asym_id[..., None] != asym_id[None, ...]

    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (1e-8 + torch.sum(
      pair_residue_weights, dim=-1, keepdim=True))
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)
    weighted = per_alignment * residue_weights
    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]

def get_confidence_metrics(
    prediction_result,
    multimer_mode: bool):
    confidence_metrics = {}
    confidence_metrics['plddt'] = compute_plddt(prediction_result['predicted_lddt']['logits'])
    if 'predicted_aligned_error' in prediction_result:
        confidence_metrics.update(compute_predicted_aligned_error(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks']))
        confidence_metrics['ptm'] = predicted_tm_score(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks'],
            asym_id=None)
        if multimer_mode:
        # Compute the ipTM only for the multimer model.
            confidence_metrics['iptm'] = predicted_tm_score(
                logits=prediction_result['predicted_aligned_error']['logits'],
                breaks=prediction_result['predicted_aligned_error']['breaks'],
                asym_id=prediction_result['predicted_aligned_error']['asym_id'],
                interface=True)
            confidence_metrics['ranking_confidence'] = (0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

    if not multimer_mode:
    # Monomer models use mean pLDDT for model ranking.
        confidence_metrics['ranking_confidence'] = torch.mean(confidence_metrics['plddt'])

    return confidence_metrics

def from_pdb_string(pdb_str: str) :
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    # if chain_id is not None:
    #     chain = model[chain_id]
    # else:
    #     chains = list(model.get_chains())
    #     if len(chains) != 1:
    #         raise ValueError(
    #             "Only single chain PDBs are supported when chain_id not specified. "
    #             f"Found {len(chains)} chains."
    #         )
    #     else:
    #         chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    for chain in model.get_chains():
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                f"PDB contains an insertion code at chain {chain.id} and residue "
                f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[
                    residue_constants.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            b_factors.append(res_b_factors)

    return atom_positions

# def add_template_feature(feats, template, template_chain, seq_length):
#     with open(template, "r") as fp:
#         pdb_string = fp.read()
#     protein_object_A = pdb_to_template.from_pdb_string(pdb_string, template_chain)
#     has_ca = protein_object_A.atom_mask[:, 0] == 1
#     template_aatype = np.expand_dims(np.concatenate((protein_object_A.aatype[has_ca], np.zeros((seq_length,),dtype=int)), axis=0), axis=0)
#     template_all_atom_pos = np.expand_dims(np.concatenate((protein_object_A.atom_positions[has_ca],  np.zeros((seq_length, 37, 3))), axis=0), axis=0)
#     template_all_atom_mask = np.expand_dims(np.concatenate((protein_object_A.atom_mask[has_ca], np.zeros((seq_length,37))), axis=0), axis=0)
#     feats['template_aatype'] =  torch.unsqueeze(torch.tensor(template_aatype, device='cuda:0'),0)
#     feats['template_all_atom_mask'] =  torch.unsqueeze(torch.tensor(template_all_atom_mask, device='cuda:0'),0)
#     feats['template_all_atom_positions'] =  torch.unsqueeze(torch.tensor(template_all_atom_pos, device='cuda:0'),0)

if __name__ == '__main__':
    #with open('../../af2_wrapper/multimer_feat_6A77.pkl', 'rb') as f:
    #    processed_feature_dict = pickle.load(f)
    #for k, v in processed_feature_dict.items():
    #    print(k, v)
    # for k, v in processed_feature_dict.items():
    #     print(k, v.shape)
    # print(processed_feature_dict['entity_id'])
    # print(processed_feature_dict['asym_id'])
    # print(processed_feature_dict['sym_id'])

    #with open('/home/thu/Downloads/5FWY_with_features/features.pkl', 'rb') as f:
    #   heteromer = pickle.load(f)
    #with open('./H1137_V6_v1/H1137_V6_v1_model_1_multimer_v2_0.pdb') as f:
    #    atom_pos = from_pdb_string(f.read())
    #atom_pos = torch.tensor(np.asarray(atom_pos), dtype=torch.double, device='cuda:0')
    #print(atom_pos.shape)
    #orig_rigids = rigid.Rigid.make_transform_from_reference(
    #    n_xyz=atom_pos[..., 0, :],
    #    ca_xyz=atom_pos[..., 1, :],
    #    c_xyz=atom_pos[..., 2, :],
    #)
    processed_feature_dict = pipeline_multimer.process('./test/6n31/6n31.json', config_multimer)
    #with open('6A77/model.000.07.pdb', "r") as fp:
    #    pdb_string = fp.read()
    #protein_object_A = pdb_to_template.from_pdb_string(pdb_string, 'A')
    #protein_object_H = pdb_to_template.from_pdb_string(pdb_string, 'H')
    #protein_object_L = pdb_to_template.from_pdb_string(pdb_string, 'L')
    #template_aatype = np.expand_dims(np.concatenate((protein_object_A.aatype, protein_object_H.aatype, protein_object_L.aatype), axis=0), axis=0)
    #template_all_atom_pos = np.expand_dims(np.concatenate((protein_object_A.atom_positions, protein_object_H.atom_positions, protein_object_L.atom_positions), axis=0), axis=0)
    #template_all_atom_mask = np.expand_dims(np.concatenate((protein_object_A.atom_mask, protein_object_H.atom_mask, protein_object_L.atom_mask), axis=0), axis=0)
    #asym_id_values_set = set(processed_feature_dict['asym_id'])
    #tot_chain_len = 0
    #origin_res_index = processed_feature_dict['residue_index'].copy()
    #origin_asym_id = processed_feature_dict['asym_id'].copy()
    #residue_index_modif = processed_feature_dict['residue_index'].copy()
    #for i in sorted(list(asym_id_values_set)):
    #    chain_len = np.sum(processed_feature_dict['asym_id'][:] == i)
    #    tot_chain_len += chain_len
    #    residue_index_modif[tot_chain_len:] += chain_len + 200
    #processed_feature_dict['residue_index'] = residue_index_modif

    #processed_feature_dict['template_aatype'] = template_aatype
    #processed_feature_dict['template_all_atom_mask'] = template_all_atom_mask
    #processed_feature_dict['template_all_atom_positions'] = template_all_atom_pos
    #processed_feature_dict['entity_id'] = np.ones(processed_feature_dict['entity_id'].shape)
    #processed_feature_dict['sym_id'] = np.ones(processed_feature_dict['sym_id'].shape)
    #processed_feature_dict['asym_id'] = np.ones(processed_feature_dict['asym_id'].shape)

    for k, v in processed_feature_dict.items():
        print(k, v.shape)

    #features_output_path = 'multimer_feat.pkl'
    #with open(features_output_path, 'wb') as f:
    #    pickle.dump(processed_feature_dict, f, protocol=4)
    feats = {k: torch.unsqueeze(torch.tensor(v, device='cuda:0'),0) for k,v in processed_feature_dict.items()}
    feats['msa_profile'] = modules_multimer.make_msa_profile(feats)
    feats = modules_multimer.sample_msa(feats, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_msa'])
    feats = modules_multimer.make_masked_msa(feats, config_multimer.config_multimer['model']['embeddings_and_evoformer']['masked_msa'])
    (feats['cluster_profile'], feats['cluster_deletion_mean']) = modules_multimer.nearest_neighbor_clusters(feats)
    feats['msa_feat'] = modules_multimer.create_msa_feat(feats)
    feats['extra_msa_feat'], feats['extra_msa_mask'] = modules_multimer.create_extra_msa_feature(feats, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_extra_msa'])

    model = modules_multimer.DockerIteration(config_multimer.config_multimer)
    load_param_multimer.import_jax_weights_(model)
    # add_template_feature(feats, './test/6n31/6n31.pdb', 'A', 10)
    #for name, param in model.named_parameters():
    #    print(name)
    #num_recycle = 4
    #with torch.no_grad():
    #    for recycle_iter in range(num_recycle):
    output = model(feats)
    #feats['asym_id'] = torch.unsqueeze(torch.tensor(origin_asym_id, device='cuda:0'),0)
    #feats['residue_index'] = torch.unsqueeze(torch.tensor(origin_res_index, device='cuda:0'),0)
    output['predicted_aligned_error']['asym_id'] = feats['asym_id'][0]
    confidences = get_confidence_metrics(output, True)
    out_converted = {}
    for k, v in confidences.items():
        if(k!="plddt" and k!="aligned_confidence_probs" and k!="predicted_aligned_error"):
            out_converted[k] = confidences[k].detach().cpu().numpy().tolist()
    out_json = out_converted
    #torch.save(get_confidence_metrics(output, True), './6A77/confidence_score.txt')
    json.dump(out_json, codecs.open('confidence_score_model1.txt', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
    plddt = confidences['plddt'].detach().cpu().numpy()
    plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
    
    pdb_out = protein_to_pdb(feats['aatype'][0].cpu().numpy(), output['final_all_atom'].detach().cpu().numpy(), feats['residue_index'][0].cpu().numpy() + 1, feats['asym_id'][0].cpu().numpy(), output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0])
    with open('model1.pdb', 'w') as f:
        f.write(pdb_out)
