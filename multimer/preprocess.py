import sys
sys.path.insert(1, '../')

import argparse
import json
import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


from multimer import msa_pairing, pdb_to_template, pipeline_multimer, feature_processing


def process_single_chain_pdb(
        all_position,
        all_mask,
        renum_mask,
        resolution,
        description,
        sequence,
        a3m_file,
        is_homomer_or_monomer,
        mmcif_dir,
        hhr_file=None
):
    pdb_feat = pdb_to_template.make_pdb_features(all_position, all_mask, renum_mask, sequence, description, resolution)
    chain_feat = pdb_feat
    with open(a3m_file, "r") as fp:
        msa = pipeline_multimer.parse_a3m(fp.read())
    msa_feat = pipeline_multimer.make_msa_features((msa,))
    chain_feat.update(msa_feat)
    if hhr_file is not None:
        with open (hhr_file) as f:
            hhr = f.read()
        pdb_temp = pipeline_multimer.get_template_hits(output_string=hhr)
        templates_result = pipeline_multimer.get_templates(query_sequence=sequence, hits=pdb_temp, mmcif_dir=mmcif_dir)
        temp_feat = templates_result.features
        chain_feat.update(temp_feat)

    if not is_homomer_or_monomer:
        all_seq_features = pipeline_multimer.make_msa_features([msa])
        valid_feats = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int',
                        'msa_uniprot_accession_identifiers','msa_species_identifiers',)
        feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

        chain_feat.update(feats)
    return chain_feat


def process_unmerged_features(all_chain_features):
  """Postprocessing stage for per-chain features before merging."""
  num_chains = len(all_chain_features)
  for chain_features in all_chain_features.values():
    # Convert deletion matrices to float.
    chain_features['deletion_matrix'] = np.asarray(
        chain_features.pop('deletion_matrix_int'), dtype=np.float32)
    if 'deletion_matrix_int_all_seq' in chain_features:
      chain_features['deletion_matrix_all_seq'] = np.asarray(
          chain_features.pop('deletion_matrix_int_all_seq'), dtype=np.float32)

    chain_features['deletion_mean'] = np.mean(
        chain_features['deletion_matrix'], axis=0)

    # Add assembly_num_chains.
    chain_features['assembly_num_chains'] = np.asarray(num_chains)

  # Add entity_mask.
  for chain_features in all_chain_features.values():
    chain_features['entity_mask'] = (
        chain_features['entity_id'] != 0).astype(np.int32)


def pair_and_merge(all_chain_features, is_homomer):
  """Runs processing on features to augment, pair and merge.

  Args:
    all_chain_features: A MutableMap of dictionaries of features for each chain.

  Returns:
    A dictionary of features.
  """

  process_unmerged_features(all_chain_features)

  np_chains_list = list(all_chain_features.values())

  pair_msa_sequences = not is_homomer

  if pair_msa_sequences:
    np_chains_list = msa_pairing.create_paired_features(
        chains=np_chains_list)
    np_chains_list = msa_pairing.deduplicate_unpaired_sequences(np_chains_list)
  np_chains_list = feature_processing.crop_chains(
      np_chains_list,
      msa_crop_size=feature_processing.MSA_CROP_SIZE,
      pair_msa_sequences=pair_msa_sequences,
      max_templates=feature_processing.MAX_TEMPLATES)
  np_example = msa_pairing.merge_chain_features(
      np_chains_list=np_chains_list, pair_msa_sequences=pair_msa_sequences,
      max_templates=feature_processing.MAX_TEMPLATES)
  np_example = feature_processing.process_final(np_example)

  return np_example


def _preprocess_one(single_dataset):

    cif_path = single_dataset['cif_file']
    file_id = os.path.basename(cif_path)[:-4]
    chains = single_dataset['chains']
    resolution = single_dataset['resolution']
    antigen_chain = single_dataset['antigen_chain']

    sequences = []
    for chain in chains:
        sequences.append(single_dataset['sequences'][chain])
    #############
    is_homomer = len(set(sequences)) == 1
    all_chain_features = {}
    for chain in chains:
        sequence = single_dataset['sequences'][chain]
        #################
        if chain != antigen_chain:
            all_atom_positions, all_atom_mask, renum_mask = pdb_to_template.align_seq_pdb(
                single_dataset['renum_seq'][chain], single_dataset['pdb_file'], chain)
        else:
            all_atom_positions, all_atom_mask, renum_mask = pdb_to_template.make_antigen_features(sequence, single_dataset['cif_file'], chain)
        description = '_'.join([file_id, chain])
        #################
        a3m_file = os.path.join(pre_alignment_path, f'{file_id}_{chain}', 'mmseqs/aggregated.a3m')
        hhr_file = os.path.join(pre_alignment_path, f'{file_id}_{chain}', 'mmseqs/aggregated.hhr')
        chain_features = process_single_chain_pdb(all_atom_positions, all_atom_mask, renum_mask, resolution,
                                                  description, sequence, a3m_file, is_homomer, mmcif_dir,
                                                  hhr_file=hhr_file)
        chain_features = pipeline_multimer.convert_monomer_features(chain_features,
                                                                    chain_id=chain)
        all_chain_features[chain] = chain_features
    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = pair_and_merge(all_chain_features, is_homomer)
    np_example = pipeline_multimer.pad_msa(np_example, 512)
    np.savez(f'{preprocessed_data_dir}/{file_id}', **np_example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default=None)
    parser.add_argument("--pre_alignment_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--mmcif_dir", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    pre_alignment_path = args.pre_alignment_path
    preprocessed_data_dir = args.preprocessed_data_dir
    mmcif_dir = args.mmcif_dir
    with open(args.json_data_path) as f:
        json_data = json.load(f)

    pool = Pool(processes=args.n_jobs)
    for _ in tqdm(pool.imap_unordered(_preprocess_one, json_data), total=len(json_data)):
        pass
    pool.close()
    pool.join()
