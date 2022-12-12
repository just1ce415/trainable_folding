import sys
sys.path.insert(1, '../')

import argparse
import json
import os
import numpy as np
from openbabel import pybel
from multiprocessing import Pool
from tqdm import tqdm

from alphadock import residue_constants
from multimer import mmcif_parsing, msa_pairing, pipeline_multimer, feature_processing


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

def make_mmcif_features(
        mmcif_object: mmcif_parsing.MmcifObject,
        chain_id: str
):
    if(chain_id not in mmcif_object.chain_to_seqres):
        input_sequence = '~'  # TODO: remove it after we get a good cif file
    else:
        input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        pipeline_multimer.make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = pipeline_multimer._get_atom_positions(
        mmcif_object, chain_id, max_ca_ca_distance=15000.0
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats

def process_single_chain(
        mmcif_object,
        chain_id,
        a3m_file,
        is_homomer_or_monomer,
        mmcif_dir,
        hhr_file=None
):
    mmcif_feat = make_mmcif_features(mmcif_object, chain_id)
    chain_feat = mmcif_feat
    with open(a3m_file, "r") as fp:
        msa = pipeline_multimer.parse_a3m(fp.read())
    msa_feat = pipeline_multimer.make_msa_features((msa,))
    chain_feat.update(msa_feat)
    if hhr_file is not None:
        with open (hhr_file) as f:
            hhr = f.read()
        pdb_temp = pipeline_multimer.get_template_hits(output_string=hhr)
        templates_result = pipeline_multimer.get_templates(query_sequence=mmcif_object.chain_to_seqres[chain_id], hits=pdb_temp, mmcif_dir=mmcif_dir)
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


def crop_feature(features, crop_size):
    seq_len = features['seq_length']
    # start_crop = random.randint(0, seq_len - crop_size)
    start_crop = seq_len - crop_size
    feat_skip = {'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates', 'cluster_bias_mask'}
    feat_1 = {'aatype', 'residue_index', 'all_atom_positions', 'all_atom_mask', 'asym_id', 'sym_id', 'entity_id', 'deletion_mean', 'entity_mask', 'seq_mask'}
    for k in features.keys():
        if k not in feat_skip:
            if k in feat_1:
                features[k] = features[k][start_crop: start_crop+crop_size]
            else:
                features[k] = features[k][:, start_crop: start_crop+crop_size]
    features['seq_length'] = crop_size
    return features

def _preprocess_one(single_dataset):
    cif_path = single_dataset['cif_file']
    file_id = os.path.basename(cif_path)[:-4]
    sdf_path = single_dataset['sdf']
    sample_id = os.path.basename(sdf_path)[:-4]
    chains = single_dataset['chains']
    with open(cif_path, 'r') as f:
        mmcif_string = f.read()
    mmcif_obj = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string).mmcif_object
    sequences = []
    for c in mmcif_obj.chain_to_seqres.keys():
        sequences.append(mmcif_obj.chain_to_seqres[c])
    is_homomer = len(set(sequences)) == 1
    all_chain_features = {}
    for chain in chains:
        if (chain in mmcif_obj.chain_to_seqres):
            a3m_file = os.path.join(pre_alignment_path, f'{file_id}_{chain}', 'mmseqs/aggregated.a3m')
            # hhr_file = os.path.join(self.pre_align, f'{file_id}_{chain}', 'mmseqs/uniref.hhr')
        else:
            a3m_file = new_res_a3m_path
        hhr_file = None
        chain_features = process_single_chain(mmcif_obj, chain, a3m_file, is_homomer, hhr_file=hhr_file, mmcif_dir=mmcif_dir)
        chain_features = pipeline_multimer.convert_monomer_features(chain_features,
                                                                chain_id=chain)
        all_chain_features[chain] = chain_features
    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = pair_and_merge(all_chain_features, is_homomer)
    np_example = pipeline_multimer.pad_msa(np_example, 512)
    ######## Template##################
    atomtype_B = []
    coordinate_B = []
    for mol in pybel.readfile('sdf', sdf_path):
        for atom in mol:
            coordinate_B.append(atom.coords)
            atomtype_B.append(atom.type)

    assert atomtype_B == verification_atom_seq, f'Issue with {sample_id}, Atom seq {atomtype_B} does not match {verification_atom_seq}'

    atomtype_B[8] = 'N'
    atomtype_B[7] = 'CA'
    atomtype_B[5] = 'C'
    temp_coor_B = np.zeros((1, 37, 3))
    temp_mask_B = np.zeros((1, 37))
    for i in range(len(coordinate_B)):
        if (atomtype_B[i] in residue_constants.atom_types):
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][0] = coordinate_B[i][0]
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][1] = coordinate_B[i][1]
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][2] = coordinate_B[i][2]
            temp_mask_B[0][residue_constants.atom_order[atomtype_B[i]]] = 1
    np_example['all_atom_positions'][-1] = temp_coor_B[0]
    np_example['all_atom_mask'][-1] = temp_mask_B[0]
    np_example = crop_feature(np_example, 384)  # in this case it is fixed

    np.savez(f'{preprocessed_data_dir}/{sample_id}', **np_example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default=None)
    parser.add_argument("--pre_alignment_path", type=str, default=None)
    parser.add_argument("--preprocessed_data_dir", type=str, default=None)
    parser.add_argument("--mmcif_dir", type=str, default=None)
    parser.add_argument("--new_res_a3m_path", type=str, default=None)
    parser.add_argument("--verification_sdf", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    pre_alignment_path = args.pre_alignment_path
    preprocessed_data_dir = args.preprocessed_data_dir
    mmcif_dir = args.mmcif_dir
    new_res_a3m_path = args.new_res_a3m_path
    verification_sdf = args.verification_sdf

    verification_atom_seq = []
    for mol in pybel.readfile('sdf', verification_sdf):
        for atom in mol:
            verification_atom_seq.append(atom.type)

    json_data = json.load(open(args.json_data_path))

    pool = Pool(processes=args.n_jobs)
    for _ in tqdm(pool.imap_unordered(_preprocess_one, json_data), total=len(json_data)):
        pass
    pool.close()
    pool.join()
