import sys
sys.path.insert(1, '../')
from alphadock import all_atom, residue_constants
import numpy as np
import torch
import json, codecs
from multimer import modules_multimer, config_multimer, load_param_multimer, load_param_multimer_v3, pipeline_multimer, pdb_to_template, new_confidence_score
import io
from Bio.PDB import PDBParser
from typing import Dict, Optional, Tuple
import argparse


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

def protein_to_pdb(aatype, atom_positions, residue_index, chain_index, atom_mask, b_factors, out_mask=None):
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
        if out_mask is not None and out_mask[i] == 0:
            continue
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

def main(FLAGS):
    processed_feature_dict = {}
    # with np.load('/data/thu/colabfold_batch_multiseed/feat.npz') as data:
    #     for k in list(data.keys()):
    #         processed_feature_dict[k] = data[k]

    processed_feature_dict = pipeline_multimer.process(FLAGS.input_file, config_multimer, FLAGS.use_mock_template, FLAGS.allow_duplicate_msa)

    for k, v in processed_feature_dict.items():
        print(k, v.shape)
    if FLAGS.model_version != 'v3':
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['evoformer']['triangle_multiplication_incoming']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['evoformer']['triangle_multiplication_outgoing']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['extra_msa']['triangle_multiplication_incoming']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['extra_msa']['triangle_multiplication_outgoing']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['template']['template_pair_stack']['triangle_multiplication_incoming']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['template']['template_pair_stack']['triangle_multiplication_outgoing']['fuse_projection_weights'] = False
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_msa'] = 252
        config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_extra_msa'] = 1152
    if FLAGS.resample_msa:
        config_multimer.config_multimer['model']['resample_msa_in_recycling'] = True
    config_multimer.config_multimer['model']['embeddings_and_evoformer']['masked_msa']['replace_fraction'] = FLAGS.msa_replace_frac
    feats = {k: torch.unsqueeze(torch.tensor(v, device='cuda:0'), 0) for k, v in processed_feature_dict.items()}
    model = modules_multimer.DockerIteration(config_multimer.config_multimer).to('cuda:0')
    if FLAGS.model_version == 'v3':
        load_param_multimer_v3.import_jax_weights_(model, FLAGS.model_weight_path)
    else:
        load_param_multimer.import_jax_weights_(model, FLAGS.model_weight_path)
    output = model(feats, FLAGS.is_validation, FLAGS.msa_indices_prefix)
    output['predicted_aligned_error']['asym_id'] = feats['asym_id'][0]
    if FLAGS.use_new_score:
        with torch.no_grad():
            confidences = new_confidence_score.get_confidence_metrics(output, output['predicted_aligned_error']['asym_id'], True)
        no_score_out = ['plddt', 'aligned_confidence_probs', 'predicted_aligned_error', 'raw_pae', 'interfaces']
    else:
        confidences = get_confidence_metrics(output, True)
        no_score_out = ['plddt', 'aligned_confidence_probs', 'predicted_aligned_error', 'raw_pae', 'interfaces']
    out_converted = {}
    for k, v in confidences.items():
        if (k not in no_score_out):
            out_converted[k] = confidences[k].detach().cpu().numpy().tolist()
        if k == 'interfaces':
            out_converted[k] = {}
            for l,h in confidences[k].items():
                out_converted[k][str(l[0].item())+'_'+str(l[1].item())] = {}
                for h1, h2 in h.items():
                    out_converted[k][str(l[0].item())+'_'+str(l[1].item())][h1] = h2.detach().cpu().numpy().tolist()
    out_json = out_converted
    json.dump(out_json, codecs.open(FLAGS.output_dir+'confidence_score_model1.txt', 'w', encoding='utf-8'), separators=(',', ':'),
              sort_keys=True, indent=4)

    plddt = confidences['plddt'].detach().cpu().numpy()
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    pdb_out = protein_to_pdb(feats['aatype'][0].cpu().numpy(), output['final_all_atom'].detach().cpu().numpy(),
                             feats['residue_index'][0].cpu().numpy() + 1, feats['asym_id'][0].cpu().numpy(),
                             output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0])
    with open(FLAGS.output_dir+'model1.pdb', 'w') as f:
        f.write(pdb_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, help="Input json file")
    parser.add_argument('--use_mock_template', type=bool, default=True, help="Use mock template")
    parser.add_argument('--allow_duplicate_msa', type=bool, default=True, help="Allow duplicate msa or not")
    parser.add_argument('--resample_msa', type=bool, default=True, help="Allow resampling msa every recycle")
    parser.add_argument('--model_version', type=str, default='v3', help="Version of model, v1, v2 or v3")
    parser.add_argument('--msa_replace_frac', type=float, default=0.15, help="how much to mask msa for bert")
    parser.add_argument('--model_weight_path', type=str, help="Path to model weight")
    parser.add_argument('--msa_indices_prefix', type=str, help="Prefix to msa index files")
    parser.add_argument('--is_validation', type=bool, default=True, help="Run in validation mode")
    parser.add_argument('--output_dir', type=str, help="Output directory")
    parser.add_argument('--use_new_score', type=bool, default=True, help="Use new confidence scoring function")
    FLAGS = parser.parse_args()
    main(FLAGS)
