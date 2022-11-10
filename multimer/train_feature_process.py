import sys
sys.path.insert(1, '../')
import argparse
import json
import numpy as np
import os
import random
import torch
from torch import optim
from torch.utils.data import Dataset
from openbabel import pybel
from Bio import PDB
from multimer import mmcif_parsing, pipeline_multimer, feature_processing
from alphadock import residue_constants

from multimer import (
    msa_pairing,
    modules_multimer,
    config_multimer,
    load_param_multimer,
    pdb_to_template,
    test_multimer
)


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

def process_single_chain(mmcif_object, chain_id, a3m_file, is_homomer_or_monomer, hhr_file=None):
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
        templates_result = pipeline_multimer.get_templates(query_sequence=mmcif_object.chain_to_seqres[chain_id], hits=pdb_temp)
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

class MultimerDataset(Dataset):
    def __init__(self, json_data, pre_alignment_path):
        self.data = json_data
        self.pre_align = pre_alignment_path
    def process(self, idx):
        single_dataset = self.data[idx]
        cif_path = single_dataset['cif_file']
        file_id = os.path.basename(cif_path)[:-4]
        chains = single_dataset['chains']
        # TODO: make good cif file with both chains
        with open(cif_path, 'r') as f:
            mmcif_string = f.read()
        mmcif_obj = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string).mmcif_object
        sequences = []
        for c in mmcif_obj.chain_to_seqres.keys():
            sequences.append(mmcif_obj.chain_to_seqres[c])
        is_homomer = len(set(sequences))==1
        all_chain_features={}
        for chain in chains:
            a3m_file = os.path.join(self.pre_align, f'{file_id}_{chain}', 'mmseqs/uniref.a3m')
            hhr_file = os.path.join(self.pre_align, f'{file_id}_{chain}', 'mmseqs/uniref.hhr')
            if not os.path.isfile(hhr_file):
                hhr_file = None
            chain_features = process_single_chain(mmcif_obj, chain, a3m_file, is_homomer, hhr_file=hhr_file)
            chain_features = pipeline_multimer.convert_monomer_features(chain_features,
                                                chain_id=chain)
            all_chain_features[chain] = chain_features
        all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
        np_example = pair_and_merge(all_chain_features, is_homomer)
        np_example = pipeline_multimer.pad_msa(np_example, 512)
        ######## Template##################
        # TODO: change hard-coded names
        with open('./test/7epe/7epe_A.pdb', "r") as fp:
            pdb_string = fp.read()
        protein_object_A = pdb_to_template.from_pdb_string(pdb_string, 'A')
        atomtype_B = []
        coordinate_B = []
        for mol in pybel.readfile('sdf', './test/7epe/7epe_B.sdf'):
            for atom in mol:
                coordinate_B.append(atom.coords)
                atomtype_B.append(atom.type)
        atomtype_B[10] = 'N'
        atomtype_B[9] = 'CA'
        atomtype_B[7] = 'C'
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

        # has_ca = protein_object_A.atom_mask[:, 0] == 1
        #template_aatype = np.expand_dims(np_example['aatype'], axis=0, )
        #template_all_atom_pos = np.expand_dims(np_example['all_atom_positions'], axis=0)
        #template_all_atom_mask = np.expand_dims(np_example['all_atom_mask'], axis=0)
        #np_example['template_aatype'] = template_aatype
        #np_example['template_all_atom_mask'] = template_all_atom_mask
        #np_example['template_all_atom_positions'] = template_all_atom_pos
        ###################################
        np_example = crop_feature(np_example, 384)
        np_example = {k: torch.tensor(v, device='cuda:0') for k,v in np_example.items()}

        return np_example
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.process(idx)

from multimer import loss_multimer, all_atom_multimer
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str)
    parser.add_argument('--pre-alignment-path', type=str)
    args = parser.parse_args()

    with open(args.json_path) as f:
        json_data = json.load(f)
    mul_dataset = MultimerDataset(json_data, args.pre_alignment_path)
    mul_loader = torch.utils.data.DataLoader(mul_dataset, batch_size=1)
    model = modules_multimer.DockerIteration(config_multimer.config_multimer)
    # load_param_multimer.import_jax_weights_(model)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for ep in range(10):
        print('EPOCH: ', ep)
        print('USED MEMORY: ', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        for item in mul_loader:
            item['msa_profile'] = modules_multimer.make_msa_profile(item)
            item = modules_multimer.sample_msa(item, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_msa'])
            item = modules_multimer.make_masked_msa(item, config_multimer.config_multimer['model']['embeddings_and_evoformer']['masked_msa'])
            (item['cluster_profile'], item['cluster_deletion_mean']) = modules_multimer.nearest_neighbor_clusters(item)
            item['msa_feat'] = modules_multimer.create_msa_feat(item)
            item['extra_msa_feat'], item['extra_msa_mask'] = modules_multimer.create_extra_msa_feature(item, config_multimer.config_multimer['model']['embeddings_and_evoformer']['num_extra_msa'])
            item['pseudo_beta'], item['pseudo_beta_mask'] = modules_multimer.pseudo_beta_fn(item['aatype'], item['all_atom_positions'], item['all_atom_mask']) 
            #aatype_one_hot = torch.nn.functional.one_hot(item['template_aatype'], 22)

            #num_templates = item['template_aatype'].shape[1]
            #all_chi_angles = []
            #all_chi_masks = []
            #for i in range(num_templates):
            #    template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
            #        item['template_all_atom_positions'][0][i, :, :, :],
            #        item['template_all_atom_mask'][0][i, :, :],
            #        item['template_aatype'][0][i, :])
            #    all_chi_angles.append(template_chi_angles)
            #    all_chi_masks.append(template_chi_mask)
            #chi_angles = torch.stack(all_chi_angles, dim=0)
            #chi_angles = torch.unsqueeze(chi_angles, dim=0)
            #chi_mask = torch.stack(all_chi_masks, dim=0)
            #chi_mask = torch.unsqueeze(chi_mask, dim=0)

            #item['template_features'] = torch.cat(
            #    (aatype_one_hot, torch.sin(chi_angles) * chi_mask, torch.cos(chi_angles) * chi_mask, chi_mask),
            #    dim=-1).type(torch.float32)
            #item['template_masks'] = chi_mask[..., 0]

            optimizer.zero_grad()
            output, loss = model(item)
            print('LOSS: ', loss)
            #loss = loss_multimer.lddt_loss(output, item, config_multimer.config_multimer['model']['heads'])
            loss.backward()
            optimizer.step()

            if(ep == 0 or ep == 9):
                output['predicted_aligned_error']['asym_id'] = item['asym_id'][0]
                confidences = test_multimer.get_confidence_metrics(output, True)
                out_converted = {}
                for k, v in confidences.items():
                    if (k != "plddt" and k != "aligned_confidence_probs" and k != "predicted_aligned_error"):
                        out_converted[k] = confidences[k].detach().cpu().numpy().tolist()
                out_json = out_converted
                # torch.save(get_confidence_metrics(output, True), './6A77/confidence_score.txt')
                # json.dump(out_json, codecs.open('confidence_score_model1.txt', 'w', encoding='utf-8'),
                #           separators=(',', ':'), sort_keys=True, indent=4)

                plddt = confidences['plddt'].detach().cpu().numpy()
                plddt_b_factors = np.repeat(
                    plddt[..., None], residue_constants.atom_type_num, axis=-1
                )

                # output['final_atom_mask'][-1] = torch.tensor(temp_mask_B[0], device=output['final_atom_mask'].device)
                pdb_out = test_multimer.protein_to_pdb(item['aatype'][0].cpu().numpy(),
                                         output['final_all_atom'].detach().cpu().numpy(),
                                         item['residue_index'][0].cpu().numpy() + 1, item['asym_id'][0].cpu().numpy(),
                                         output['final_atom_mask'].cpu().numpy(), plddt_b_factors[0])
                file_out = "model_0.pdb" if ep == 0 else "model_9.pdb"
                with open(file_out, 'w') as f:
                    f.write(pdb_out)

            del output
            del loss
            del item
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print('ALLOCATED MEMORY: ', torch.cuda.memory_allocated(0) / 1024 / 1024)



