import sys
sys.path.insert(1, '../')

import argparse
import json
import os
import Bio.PDB
import numpy as np
from openbabel import pybel
from multiprocessing import Pool
from tqdm import tqdm

from alphadock import residue_constants
from multimer import mmcif_parsing, msa_pairing, pipeline_multimer, feature_processing

import warnings
warnings.filterwarnings(action='ignore')

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
    # - 1 because we will always add the last element
    seq_len = features['seq_length'] - 1
    crop_size = crop_size - 1
    crop_size = min(seq_len, crop_size)
    # + 1 because np.random.randint doesn't include highest value
    start_crop = np.random.randint(0, seq_len - crop_size + 1)
    feat_skip = {'seq_length', 'resolution', 'num_alignments', 'assembly_num_chains', 'num_templates', 'cluster_bias_mask'}
    feat_1 = {
        'aatype', 'residue_index', 'all_atom_positions',
        'all_atom_mask', 'asym_id', 'sym_id', 'entity_id',
        'deletion_mean', 'entity_mask', 'seq_mask', 'loss_mask'
    }
    for k in features.keys():
        if k not in feat_skip:
            if k in feat_1:
                # before: (seq_len, ...), after: (crop_size + 1, ...)
                features[k] = np.concatenate((
                    features[k][start_crop: start_crop+crop_size],
                    features[k][-1:]
                ), axis=0)
            else:
                # before: (n_msa, seq_len, ...), after: (n_msa, crop_size + 1, ...)
                features[k] = np.concatenate((
                    features[k][:, start_crop: start_crop + crop_size],
                    features[k][:, -1:]
                ), axis=1)
    features['seq_length'] = crop_size + 1
    return features

def _preprocess_one(single_dataset):
    cif_path = single_dataset['cif_file']
    pdb_id = os.path.basename(cif_path)[:-4]
    sdf_path = single_dataset['sdf']
    sample_id = os.path.basename(sdf_path)[:-4]
    # chains = single_dataset['chains']
    sample_name = single_dataset['sdf'].split('/')[-1][:-4]
    chains = [sample_name[16:17], 'Z']

    parser = Bio.PDB.MMCIFParser()
    structure = parser.get_structure(pdb_id, cif_path)
    fragment_names = ['FMN', 'FAD', 'FAE', '9WY', 'RBF', 'LFN', 'C3F', 'FAS', 'CF2', 'CF4']
    n_fragments = sum([1 for res in structure[0][sample_name[16:17]] if res.get_resname() in fragment_names])

    if n_fragments != 1:
        return

    ### START READING COORDS FOR NEW RES

    atomtype_B = []
    coordinate_B = []
    for mol in pybel.readfile('sdf', sdf_path):
        for atom in mol:
            coordinate_B.append(atom.coords)
            atomtype_B.append(atom.type)

    if atomtype_B != verification_atom_seq:
        print(f'Issue with {sample_id}, Atom seq {atomtype_B} does not match {verification_atom_seq}')
        return


    atomtype_B[13] = 'N'
    atomtype_B[2] = 'CA'
    atomtype_B[4] = 'C'
    temp_coor_B = np.zeros((1, 37, 3))
    temp_mask_B = np.zeros((1, 37))
    for i in range(len(coordinate_B)):
        if (atomtype_B[i] in residue_constants.atom_types):
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][0] = coordinate_B[i][0]
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][1] = coordinate_B[i][1]
            temp_coor_B[0][residue_constants.atom_order[atomtype_B[i]]][2] = coordinate_B[i][2]
            temp_mask_B[0][residue_constants.atom_order[atomtype_B[i]]] = 1

    ### END READING COORDS FOR NEW RES

    with open(cif_path, 'r') as f:
        mmcif_string = f.read()
    mmcif_obj = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=mmcif_string).mmcif_object
    sequences = []

    release_date = mmcif_obj.header['release_date']
    single_dataset['release_date'] = release_date
    dataset = 'train'
    # if release_date >= '2020-01-01':
    #     dataset = 'val'
    if release_date >= '2021-01-01':
        dataset = 'test'
    single_dataset['dataset'] = dataset

    for c in mmcif_obj.chain_to_seqres.keys():
        sequences.append(mmcif_obj.chain_to_seqres[c])
    is_homomer = len(set(sequences)) == 1
    all_chain_features = {}
    for chain in chains:
        if (chain in mmcif_obj.chain_to_seqres):
            single_dataset['seq_len'] = len(mmcif_obj.chain_to_seqres[chain])
            a3m_file = os.path.join(pre_alignment_path, f'{pdb_id}_{chain}', 'mmseqs/aggregated.a3m')
        else:
            a3m_file = new_res_a3m_path
        hhr_file = None
        chain_features = process_single_chain(mmcif_obj, chain, a3m_file, is_homomer, hhr_file=hhr_file, mmcif_dir=mmcif_dir)
        chain_features = pipeline_multimer.convert_monomer_features(chain_features, chain_id=chain)

        all_chain_features[chain] = chain_features


    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    np_example = pair_and_merge(all_chain_features, is_homomer)
    np_example = pipeline_multimer.pad_msa(np_example, 512)

    np_example['all_atom_positions'][-1] = temp_coor_B[0]
    np_example['all_atom_mask'][-1] = temp_mask_B[0]
    # np_example = crop_feature(np_example, 384)  # in this case it is fixed

    distances = np.sqrt(
        np.sum(
            (
                np_example['all_atom_positions'][None, -1, ...]
                - np_example['all_atom_positions'][:, ...]
            )
            ** 2,
            axis=-1,
        )
    )

    close_atoms = (distances < 5.0) * np_example['all_atom_mask']
    if close_atoms.sum() < 2:
        return
    np_example['loss_mask'] = (close_atoms.sum(axis=1) > 0.0) * 1.0

    # np.savez(f'{preprocessed_data_dir}/{sample_id}', **np_example)

    return single_dataset


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
    json_data = [{**j, 'seed': i} for i, j in enumerate(json_data)]

    res = []

    pool = Pool(processes=args.n_jobs)
    for r in tqdm(pool.imap_unordered(_preprocess_one, json_data), total=len(json_data)):
        if r:
            res.append(r)
    pool.close()
    pool.join()

    _train_data = [r for r in res if r['dataset'] == 'train']
    _sorted_train_data = sorted(_train_data, key=lambda x: x['release_date'])
    train_data = _sorted_train_data[:-276]
    val_data = _sorted_train_data[-276:]
    test_data = [r for r in res if r['dataset'] == 'test']
    for i, r in enumerate(train_data):
        r['seed'] = i
        r['dataset'] = 'train'
    for i, r in enumerate(val_data):
        r['seed'] = i
        r['dataset'] = 'val'
    test_data = [r for r in res if r['dataset'] == 'test']

    os.makedirs(preprocessed_data_dir, exist_ok=True)

    with open(f"{preprocessed_data_dir}/train.json", "w") as outfile:
        outfile.write(json.dumps(train_data, indent=4))

    with open(f"{preprocessed_data_dir}/val.json", "w") as outfile:
        outfile.write(json.dumps(val_data, indent=4))

    with open(f"{preprocessed_data_dir}/test.json", "w") as outfile:
        outfile.write(json.dumps(test_data, indent=4))

    print('Init:', len(json_data))
    print('Done:', len(res))
