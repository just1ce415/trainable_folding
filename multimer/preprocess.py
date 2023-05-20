import sys
sys.path.insert(1, '../')

import argparse
import json
import os
import requests
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


from multimer import mmcif_parsing, msa_pairing, pdb_to_template, pipeline_multimer, feature_processing


def find_mask_groups(lst):
    groups = []
    start = None
    for i, x in enumerate(lst):
        if x == 1:
            if start is None:
                start = i
        elif start is not None:
            groups.append((start, i - 1))
            start = None
    if start is not None:
        groups.append((start, len(lst) - 1))
    return groups

def make_mmcif_features(
        mmcif_object: mmcif_parsing.MmcifObject,
        chain_id: str
):
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
    np_chains_list = msa_pairing.create_paired_features(chains=np_chains_list)
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

def download_cif(pdb_id, output_file=None):
    if output_file is None:
        output_file = f"{pdb_id}.cif"

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)

    if response.status_code == 200:
        with open(output_file, "w") as f:
            f.write(response.text)

def crop_feature(features, crop_size, random=True):
    seq_len = features['seq_length']
    if seq_len <= crop_size:
        return features
    if random:
        start_crop = np.random.randint(0, seq_len - crop_size)
    else:
        start_crop = 0
    feat_skip = {
        'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates', 'cluster_bias_mask',
        'auth_chain_id', 'domain_name', 'sequence', 'release_date', 'is_distillation',
        'msa_species_identifiers', 'msa_species_identifiers_all_seq'
    }
    feat_1 = {
        'aatype', 'residue_index', 'all_atom_positions',
        'all_atom_mask', 'asym_id', 'sym_id', 'entity_id',
        'deletion_mean', 'entity_mask', 'seq_mask', 'loss_mask',
        'between_segment_residues'
    }
    for k in features.keys():
        if k not in feat_skip:
            if k in feat_1:
                # before: (seq_len, ...), after: (crop_size + 1, ...)
                features[k] = features[k][start_crop: start_crop+crop_size]
            else:
                # before: (n_msa, seq_len, ...), after: (n_msa, crop_size + 1, ...)
                try:
                    features[k] = features[k][:, start_crop: start_crop + crop_size]
                except:
                    print('ERROR', k, features[k].shape, start_crop, crop_size)
    features['seq_length'] = np.array([crop_size])
    return features

def _preprocess_one(single_dataset):
    pdb_id = single_dataset['pdb_id']
    sample_id = single_dataset['sample_id']
    protein_chain = single_dataset['protein_chain']
    peptide_chain = single_dataset['peptide_chain']
    deposition_date = single_dataset['deposition_date']

    cif_path = single_dataset.get('cif_file', -1)
    # download cif from pdb
    if cif_path == -1:
        cif_path = f'{preprocessed_data_dir}/cif_files/{pdb_id}.cif'
    if not os.path.exists(cif_path):
        cif_path = f'{preprocessed_data_dir}/cif_files/{pdb_id}.cif'
        os.makedirs(f'{preprocessed_data_dir}/cif_files/', exist_ok=True)
        download_cif(pdb_id, output_file=cif_path)
        single_dataset['cif_path'] = cif_path

    chains = [protein_chain, peptide_chain]

    with open(cif_path, 'r') as f:
        mmcif_string = f.read()
    mmcif_obj = mmcif_parsing.parse(file_id=pdb_id.upper(), mmcif_string=mmcif_string).mmcif_object

    if mmcif_obj is None:
        print(f'No mmcif object found for {pdb_id}')
        return None

    dataset = 'train'
    if deposition_date >= '2020-01-01':
        dataset = 'val'
    if deposition_date >= '2021-01-01':
        dataset = 'test'
    single_dataset['dataset'] = dataset

    is_homomer = len(set(chains)) == 1

    all_chain_features = {}

    # preprocess protein chain
    a3m_file = os.path.join(pre_alignment_path, f'{pdb_id.upper()}_{protein_chain}', 'mmseqs/aggregated.a3m')
    hhr_file = None
    if not os.path.exists(a3m_file):
        print(f'No a3m file found for {pdb_id}_{protein_chain}')
        return None

    chain_features = process_single_chain(mmcif_obj, protein_chain, a3m_file, is_homomer, hhr_file=hhr_file, mmcif_dir=mmcif_dir)
    chain_features = pipeline_multimer.convert_monomer_features(chain_features, chain_id=protein_chain)
    chain_features = crop_feature(chain_features, 182, random=False)
    all_chain_features[protein_chain] = chain_features

    # preprocess peptide chain
    a3m_file = f'{pre_alignment_path}/{pdb_id.upper()}_{peptide_chain}/dummy.a3m'
    hhr_file = None
    if not os.path.exists(a3m_file):
        os.makedirs(f'{pre_alignment_path}/{pdb_id.upper()}_{peptide_chain}/', exist_ok=True)
        with open(a3m_file, 'w') as f:
            f.write(f'>dummy_msa_{pdb_id.upper()}_{peptide_chain}\n{mmcif_obj.chain_to_seqres[peptide_chain]}\n')

    chain_features = process_single_chain(mmcif_obj, peptide_chain, a3m_file, is_homomer, hhr_file=hhr_file, mmcif_dir=mmcif_dir)
    chain_features = pipeline_multimer.convert_monomer_features(chain_features, chain_id=peptide_chain)
    all_chain_features[peptide_chain] = chain_features

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    try:
        np_example = pair_and_merge(all_chain_features, is_homomer)
    except:
        print(f'Issue with {pdb_id}')
        return
    np_example = pipeline_multimer.pad_msa(np_example, 512)

    np.savez(f'{preprocessed_data_dir}/npz_data/{sample_id}', **np_example)

    return single_dataset


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

    res = []

    pool = Pool(processes=args.n_jobs)
    for r in tqdm(pool.imap_unordered(_preprocess_one, json_data), total=len(json_data)):
        if r:
            res.append(r)
    pool.close()
    pool.join()

    train_data = [r for r in res if r['dataset'] == 'train']
    test_data = [r for r in res if r['dataset'] == 'test']
    val_data = [r for r in res if r['dataset'] == 'val']

    with open(f"{preprocessed_data_dir}/train.json", "w") as outfile:
        outfile.write(json.dumps(train_data, indent=4))

    with open(f"{preprocessed_data_dir}/val.json", "w") as outfile:
        outfile.write(json.dumps(val_data, indent=4))

    with open(f"{preprocessed_data_dir}/test.json", "w") as outfile:
        outfile.write(json.dumps(test_data, indent=4))

    print('Init:', len(json_data))
    print('Done:', len(res))
