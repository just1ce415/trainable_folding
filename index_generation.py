import jax
import sys
sys.path.insert(1,'/data/thu/design_multiseed/')
import collections
import contextlib
import copy
import dataclasses
import json
import os
from typing import Mapping, MutableMapping, Sequence, Optional, Any, Union
import numpy as np
import sys
import time
from absl import logging
from absl import app
from absl import flags
import ml_collections
import haiku as hk
import jax.numpy as jnp
import tensorflow.compat.v1 as tf
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import feature_processing
from alphafold.data import msa_pairing
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import features
from alphafold.model import prng
from alphafold.model import modules_multimer
from alphafold.model import utils
from alphafold.data import parsers
from alphafold.data import msa_identifiers
import config as config

flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer_v1', 'multimer_v2', 'multimer_v3', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')

flags.DEFINE_string('fasta_dir', None, 'Path to fasta including all chain sequences')

flags.DEFINE_string('pre_align_dir', None, 'Path to prealignment including hhr, a3m')
flags.DEFINE_integer('num_prediction', 5, 'Number of runs per model parameter')
flags.DEFINE_bool('enable_single_seed', True, 'Choose single seed mode')

flags.DEFINE_bool('allow_msa_duplicate', True, 'Allow duplication in msa sequences')

flags.DEFINE_integer('single_seed', 100, 'Single seed for all predictions')
flags.DEFINE_integer('num_recycle', 4, 'Single seed for all predictions')
flags.DEFINE_string('msa_file_name', 'uniref.a3m', 'MSA file name')
FLAGS = flags.FLAGS

def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = 'be' if should_be_set else 'not be'
        raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')

@dataclasses.dataclass(frozen=True)
class _FastaChain:
  sequence: str
  description: str

def _make_chain_id_map(*,
                       sequences: Sequence[str],
                       descriptions: Sequence[str],
                       ) -> Mapping[str, _FastaChain]:
  """Makes a mapping from PDB-format chain ID to sequence and description."""
  if len(sequences) != len(descriptions):
    raise ValueError('sequences and descriptions must have equal length. '
                     f'Got {len(sequences)} != {len(descriptions)}.')
  if len(sequences) > protein.PDB_MAX_CHAINS:
    raise ValueError('Cannot process more chains than the PDB format supports. '
                     f'Got {len(sequences)} chains.')
  chain_id_map = {}
  for chain_id, sequence, description in zip(
      protein.PDB_CHAIN_IDS, sequences, descriptions):
    chain_id_map[chain_id] = _FastaChain(
        sequence=sequence, description=description)
  return chain_id_map

def make_msa_features_colab(msas: Sequence[parsers.Msa]) -> pipeline.FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(
          msa.descriptions[sequence_index])
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
  return features

def add_assembly_features(
    all_chain_features: MutableMapping[str, pipeline.FeatureDict],
    ) -> MutableMapping[str, pipeline.FeatureDict]:
  """Add features to distinguish between chains.

  Args:
    all_chain_features: A dictionary which maps chain_id to a dictionary of
      features for each chain.

  Returns:
    all_chain_features: A dictionary which maps strings of the form
      `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
      chains from a homodimer would have keys A_1 and A_2. Two chains from a
      heterodimer would have keys A_1 and B_1.
  """
  # Group the chains by sequence
  seq_to_entity_id = {}
  grouped_chains = collections.defaultdict(list)
  for chain_id, chain_features in all_chain_features.items():
    seq = str(chain_features['sequence'])
    if seq not in seq_to_entity_id:
      seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
    grouped_chains[seq_to_entity_id[seq]].append((chain_features, protein.PDB_CHAIN_IDS.index(chain_id)))

  new_all_chain_features = {}
  chain_id = 1
  for entity_id, group_chain_features in grouped_chains.items():
    for sym_id, (chain_features, chain_id) in enumerate(group_chain_features, start=1):
      new_all_chain_features[
          f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
      seq_length = chain_features['seq_length']
      chain_features['asym_id'] = chain_id * np.ones(seq_length)
      chain_features['sym_id'] = sym_id * np.ones(seq_length)
      chain_features['entity_id'] = entity_id * np.ones(seq_length)
      chain_id += 1
  return new_all_chain_features

def convert_monomer_features(
    monomer_features: pipeline.FeatureDict,
    chain_id: str) -> pipeline.FeatureDict:
  """Reshapes and modifies monomer features for multimer models."""
  converted = {}
  converted['auth_chain_id'] = np.asarray(chain_id, dtype=np.object_)
  unnecessary_leading_dim_feats = {
      'sequence', 'domain_name', 'num_alignments', 'seq_length'}
  for feature_name, feature in monomer_features.items():
    if feature_name in unnecessary_leading_dim_feats:
      # asarray ensures it's a np.ndarray.
      feature = np.asarray(feature[0], dtype=feature.dtype)
    elif feature_name == 'aatype':
      # The multimer model performs the one-hot operation itself.
      feature = np.argmax(feature, axis=-1).astype(np.int32)
    elif feature_name == 'template_aatype':
      feature = np.argmax(feature, axis=-1).astype(np.int32)
      new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
      feature = np.take(new_order_list, feature.astype(np.int32), axis=0)
    elif feature_name == 'template_all_atom_masks':
      feature_name = 'template_all_atom_mask'
    converted[feature_name] = feature
  return converted

def int_id_to_str_id(num: int) -> str:
  """Encodes a number as a string, using reverse spreadsheet style naming.

  Args:
    num: A positive integer.

  Returns:
    A string that encodes the positive integer using reverse spreadsheet style,
    naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
    usual way to encode chain IDs in mmCIF files.
  """
  if num <= 0:
    raise ValueError(f'Only positive integers allowed, got {num}.')

  num = num - 1  # 1-based indexing.
  output = []
  while num >= 0:
    output.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(output)

def pad_msa(np_example, min_num_seq):
  np_example = dict(np_example)
  num_seq = np_example['msa'].shape[0]
  if num_seq < min_num_seq:
    for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
      np_example[feat] = np.pad(
          np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
    np_example['cluster_bias_mask'] = np.pad(
        np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq),))
  return np_example

def pair_and_merge_colab(
    all_chain_features: MutableMapping[str, pipeline.FeatureDict]
    ) -> pipeline.FeatureDict:
  """Runs processing on features to augment, pair and merge.

  Args:
    all_chain_features: A MutableMap of dictionaries of features for each chain.

  Returns:
    A dictionary of features.
  """

  feature_processing.process_unmerged_features(all_chain_features)

  np_chains_list = list(all_chain_features.values())

  pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)

  if pair_msa_sequences:
    np_chains_list = msa_pairing.create_paired_features(
        chains=np_chains_list)
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

class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self):
      pass
  
  def _process_single_chain(
      self,
      chain_id: str,
      sequence: str,
      description: str,
      msa_output_dir: str,
      allow_msa_dup: bool) -> pipeline.FeatureDict:

      a3m_file = os.path.join(msa_output_dir,description,f'mmseqs/{FLAGS.msa_file_name}')
      # hhr_file = '/home/thu/Downloads/alphafold/msa/' + description + '.hhr'
      with open(a3m_file, "r") as fp:
          msa = parsers.parse_a3m(fp.read())
          data = {"msa": msa.sequences, "deletion_matrix": msa.deletion_matrix}
          sequence = msa.sequences[0]
          num_res = len(sequence)
          #print(sequence, num_res)
          msas, deletion_matrices = zip(*[
          (data["msa"], data["deletion_matrix"])])
          if allow_msa_dup:
              chain_features = {
                **pipeline.make_sequence_features(sequence=sequence, description="none",
                                              num_res=num_res),
                **make_msa_features_colab((msa,))
            }
          else:
              chain_features = {
                **pipeline.make_sequence_features(sequence=sequence, description="none",
                                              num_res=num_res),
                **pipeline.make_msa_features((msa,))
            }
      all_seq_features = pipeline.make_msa_features([msa])
      valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_uniprot_accession_identifiers',
        'msa_species_identifiers',
      )
      feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

      chain_features.update(feats)

      return chain_features

  def process(self,
              input_fasta_path: str,
              msa_output_dir: str,
              allow_msa_dup: bool = False) -> pipeline.FeatureDict:
    """Runs alignment tools on the input sequences and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

    chain_id_map = _make_chain_id_map(sequences=input_seqs,
                                      descriptions=input_descs)
    chain_id_map_path = os.path.join(msa_output_dir, 'chain_id_map.json')
    with open(chain_id_map_path, 'w') as f:
      chain_id_map_dict = {chain_id: dataclasses.asdict(fasta_chain)
                           for chain_id, fasta_chain in chain_id_map.items()}
      json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)

    all_chain_features = {}
    sequence_features = {}
    is_homomer_or_monomer = len(set(input_seqs)) == 1
    for chain_id, fasta_chain in chain_id_map.items():
      if fasta_chain.sequence in sequence_features:
        all_chain_features[chain_id] = copy.deepcopy(
            sequence_features[fasta_chain.sequence])
        continue
      chain_features = self._process_single_chain(
          chain_id=chain_id,
          sequence=fasta_chain.sequence,
          description=fasta_chain.description,
          msa_output_dir=msa_output_dir,
          allow_msa_dup=allow_msa_dup)

      chain_features = convert_monomer_features(chain_features,
                                                chain_id=chain_id)
      all_chain_features[chain_id] = chain_features
      sequence_features[fasta_chain.sequence] = chain_features

    all_chain_features = add_assembly_features(all_chain_features)

    if allow_msa_dup:
        np_example = pair_and_merge_colab(
            all_chain_features=all_chain_features,
        )
    else:
        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features,
        )

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pad_msa(np_example, 512)
    return np_example

def sample_msa(key, batch, max_seq, output_dir, index, recycle_idx):
  logits = (jnp.clip(jnp.sum(batch['msa_mask'], axis=-1), 0., 1.) - 1.) * 1e6
  if 'cluster_bias_mask' not in batch:
    cluster_bias_mask = jnp.pad(
        jnp.zeros(batch['msa'].shape[0] - 1), (1, 0), constant_values=1.)
  else:
    cluster_bias_mask = batch['cluster_bias_mask']

  logits += cluster_bias_mask * 1e6
  index_order = modules_multimer.gumbel_argsort_sample_idx(key.get(), logits)
  file_out = open(os.path.join(output_dir,'msa_indices_'+str(index)+'_'+str(recycle_idx)+'.txt'),'w')
  for i in index_order:
      file_out.write(str(i)+'\n')
  sel_idx = index_order[:max_seq]
  extra_idx = index_order[max_seq:]

  for k in ['msa', 'deletion_matrix', 'msa_mask', 'bert_mask']:
      if k in batch:
          batch['extra_' + k] = batch[k][np.array(extra_idx)]
          batch[k] = batch[k][sel_idx]
  return batch

def make_masked_msa(batch, key, config, out_dir, index, recycle_idx, epsilon=1e-6):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly.
  random_aa = jnp.array([0.05] * 20 + [0., 0.], dtype=jnp.float32)

  categorical_probs = (
      config.uniform_prob * random_aa +
      config.profile_prob * batch['msa_profile'] +
      config.same_prob * jax.nn.one_hot(batch['msa'], 22))

  # Put all remaining probability on [MASK] which is a new column.
  pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
  pad_shapes[-1][1] = 1
  mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
  assert mask_prob >= 0.
  categorical_probs = jnp.pad(
      categorical_probs, pad_shapes, constant_values=mask_prob)
  sh = batch['msa'].shape
  key, mask_subkey, gumbel_subkey = key.split(3)
  uniform = utils.padding_consistent_rng(jax.random.uniform)
  mask_position = uniform(mask_subkey.get(), sh) < config.replace_fraction
  mask_position *= batch['msa_mask']

  logits = jnp.log(categorical_probs + epsilon)
  bert_msa = modules_multimer.gumbel_max_sample(gumbel_subkey.get(), logits)
  bert_msa = jnp.where(mask_position,
                        jnp.argmax(bert_msa, axis=-1), batch['msa'])
  bert_msa *= batch['msa_mask']

  # Mix real and masked MSA.
  if 'bert_mask' in batch:
    batch['bert_mask'] *= mask_position.astype(jnp.float32)
  else:
    batch['bert_mask'] = mask_position.astype(jnp.float32)
  batch['true_msa'] = batch['msa']
  batch['msa'] = bert_msa
  #jax.debug.print("x: {}", batch['bert_mask'])
  file_out = os.path.join(out_dir,'bert_'+str(index)+'_'+str(recycle_idx)+'.npz')
  np.savez(file_out, bert_mask=batch['bert_mask'], bert_msa=bert_msa)

class EmbeddingsAndEvoformer(hk.Module):
  """Embeds the input data and runs Evoformer.

  Produces the MSA, single and pair representations.
  """

  def __init__(self, config, global_config, name='evoformer'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
  def __call__(self, batch, output_dir, index, recycle_idx, is_training, safe_key=None):

    c = self.config
    gc = self.global_config

    batch = dict(batch)
    dtype = jnp.bfloat16 if gc.bfloat16 else jnp.float32
    batch['msa_profile'] = modules_multimer.make_msa_profile(batch)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    safe_key, sample_key, mask_key = safe_key.split(3)
    batch = sample_msa(sample_key, batch, c.num_msa, output_dir, index, recycle_idx)
    make_masked_msa(batch, mask_key, c.masked_msa, output_dir, index, recycle_idx)


class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch,
               index,
               recycle_idx,
               output_dir,
               is_training,
               return_representations=False,
               safe_key=None):
    if is_training:
      num_ensemble = np.asarray(self.config.num_ensemble_train)
    else:
      num_ensemble = np.asarray(self.config.num_ensemble_eval)

    embedding_module = EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)

    safe_key, safe_subkey = safe_key.split()
    embedding_module(
          batch, output_dir,  index, recycle_idx, is_training, safe_key=safe_subkey)

class AlphaFold(hk.Module):
  """AlphaFold-Multimer model with recycling.
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config
  def __call__(
      self,
      batch,
      outdir,
      idx,
      num_recycle,
      is_training,
      return_representations=False,
      safe_key=None):

    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)
    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)
    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def apply_network(prev, safe_key, rec_idx):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          output_dir=outdir,
          index=idx,
          recycle_idx=rec_idx,
          is_training=is_training,
          safe_key=safe_key)
    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      prev['prev_pos'] = jnp.zeros(
          [num_res, residue_constants.atom_type_num, 3])
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_res, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_res, num_res, emb_config.pair_channel])
    if num_recycle >0:
        for i in range(num_recycle):
            safe_key1, safe_key2 = safe_key.split()
            apply_network(prev=prev, safe_key=safe_key2, rec_idx=i)
            safe_key = safe_key1
    else:
        num_recycle = 0
    apply_network(prev=prev, safe_key=safe_key, rec_idx=num_recycle)


class RunModel:
  """Container for JAX model."""

  def __init__(self,
               config: ml_collections.ConfigDict,
               params: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None):
    self.config = config
    self.params = params
    def _forward_fn(batch, outdir, idx, num_recycle):
        model = AlphaFold(self.config.model)
        return model(
            batch,
            outdir,
            idx,
            num_recycle,
            is_training=False)
    self.apply = hk.transform(_forward_fn).apply
  def process_features(
      self,
      raw_features: Union[tf.train.Example, features.FeatureDict],
      random_seed: int) -> features.FeatureDict:

    return raw_features
  def predict(self,
              feat: features.FeatureDict,
              random_seed: int,
              outdir: str,
              index: int,
              num_recycle:int
              ) -> Mapping[str, Any]:
    result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat, outdir, index, num_recycle)


def predict_structure(
    output_dir_base,
    model_runners,
    random_seed,
    features,
    num_predictions,
    model_preset,
    enable_single_seed,
    single_seed,
    num_recycles
    ):
    timings = {}
    output_dir = os.path.join(output_dir_base, "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
        logging.info('Running model %s', model_name)
        if enable_single_seed:
            model_random_seed = single_seed
        else:
            model_random_seed = int(random_seed[model_index % num_predictions])

        processed_feature_dict = model_runner.process_features(
         features, random_seed=model_random_seed)
        model_runner.predict(processed_feature_dict, random_seed=model_random_seed, outdir=output_dir_base, index=model_index, num_recycle=num_recycles)

def main(argv):
    dp = DataPipeline()
    multimer_feature = dp.process(FLAGS.fasta_dir,FLAGS.pre_align_dir,allow_msa_dup=FLAGS.allow_msa_duplicate)
    num_ensemble = 1
    model_runners = {}
    model_names = config.MODEL_PRESETS[FLAGS.model_preset]
    for model_name in model_names:
        model_config = config.model_config(model_name)
    model_config.model.num_ensemble_eval = num_ensemble

    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = RunModel(model_config, model_params)
    for i in range(FLAGS.num_prediction):
        model_runners[f'{model_name}_pred_{i}'] = model_runner
    #random_seed = FLAGS.random_seed
    #if random_seed is None:
    #    random_seed = random.randrange(sys.maxsize // len(model_names))
    random_seed = np.linspace(0, sys.maxsize-512, num=FLAGS.num_prediction)
    predict_structure(FLAGS.output_dir, model_runners, random_seed, multimer_feature, FLAGS.num_prediction, FLAGS.model_preset, FLAGS.enable_single_seed, FLAGS.single_seed, FLAGS.num_recycle)

if __name__ == '__main__':
    app.run(main)
