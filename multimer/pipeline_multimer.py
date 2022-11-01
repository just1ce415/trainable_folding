import collections
import contextlib
import copy
import dataclasses
import json
import os
import tempfile
from typing import Mapping, MutableMapping, Optional, Sequence, Union, Any, Set, Tuple, List, Dict, Iterable
from alphadock import residue_constants
import re
import string
import numpy as np
import datetime
import shutil
from absl import logging
from multimer import msa_pairing, feature_processing, mmcif_parsing, parsers

FeatureDict = MutableMapping[str, np.ndarray]
DeletionMatrix = Sequence[Sequence[int]]

_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot
    (?:tr|sp)
    \|
    # A primary accession number of the UniProtKB entry.
    (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})
    # Occasionally there is a _0 or _1 isoform suffix, which we ignore.
    (?:_\d)?
    \|
    # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic
    # protein ID code.
    (?:[A-Za-z0-9]+)
    _
    # A mnemonic species identification code.
    (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})
    # Small BFD uses a final value after an underscore, which we ignore.
    (?:_\d+)?
    $
    """,
    re.VERBOSE)

MSA_CROP_SIZE = 2048

TEMPLATE_FEATURES = {
    'template_aatype': np.float32,
    'template_all_atom_masks': np.float32,
    'template_all_atom_positions': np.float32,
    'template_domain_names': np.object,
    'template_sequence': np.object,
    'template_sum_probs': np.float32,
}

@dataclasses.dataclass(frozen=True)
class Msa:
  """Class representing a parsed MSA file."""
  sequences: Sequence[str]
  deletion_matrix: DeletionMatrix
  descriptions: Sequence[str]

  def __post_init__(self):
    if not (len(self.sequences) ==
            len(self.deletion_matrix) ==
            len(self.descriptions)):
      raise ValueError(
          'All fields for an MSA must have the same length. '
          f'Got {len(self.sequences)} sequences, '
          f'{len(self.deletion_matrix)} rows in the deletion matrix and '
          f'{len(self.descriptions)} descriptions.')

  def __len__(self):
    return len(self.sequences)

  def truncate(self, max_seqs: int):
    return Msa(sequences=self.sequences[:max_seqs],
               deletion_matrix=self.deletion_matrix[:max_seqs],
               descriptions=self.descriptions[:max_seqs])

@dataclasses.dataclass(frozen=True)
class TemplateHit:
  """Class representing a template hit."""
  index: int
  name: str
  aligned_cols: int
  sum_probs: Optional[float]
  query: str
  hit_sequence: str
  indices_query: List[int]
  indices_hit: List[int]

@dataclasses.dataclass(frozen=True)
class TemplateSearchResult:
  features: Mapping[str, Any]
  errors: Sequence[str]
  warnings: Sequence[str]

@dataclasses.dataclass(frozen=True)
class SingleHitResult:
  features: Optional[Mapping[str, Any]]
  error: Optional[str]
  warning: Optional[str]

def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
  """Parses FASTA string and returns list of strings with amino-acid sequences.

  Arguments:
    fasta_string: The string contents of a FASTA file.

  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines():
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line

  return sequences, descriptions


def parse_a3m(a3m_string: str) -> Msa:
  """Parses sequences and deletion matrix from a3m format alignment.

  Args:
    a3m_string: The string contents of a a3m file. The first sequence in the
      file should be the query sequence.

  Returns:
    A tuple of:
      * A list of sequences that have been aligned to the query. These
        might contain duplicates.
      * The deletion matrix for the alignment as a list of lists. The element
        at `deletion_matrix[i][j]` is the number of residues deleted from
        the aligned sequence i at residue position j.
      * A list of descriptions, one per sequence, from the a3m file.
  """
  sequences, descriptions = parse_fasta(a3m_string)
  deletion_matrix = []
  for msa_sequence in sequences:
    deletion_vec = []
    deletion_count = 0
    for j in msa_sequence:
      if j.islower():
        deletion_count += 1
      else:
        deletion_vec.append(deletion_count)
        deletion_count = 0
    deletion_matrix.append(deletion_vec)

  # Make the MSA matrix out of aligned (deletion-free) sequences.
  deletion_table = str.maketrans('', '', string.ascii_lowercase)
  aligned_sequences = [s.translate(deletion_table) for s in sequences]
  return Msa(sequences=aligned_sequences,
             deletion_matrix=deletion_matrix,
             descriptions=descriptions)


def get_template_hits(output_string: str) -> Sequence[parsers.TemplateHit]:
    return parsers.parse_hhr(output_string)

@dataclasses.dataclass(frozen=True)
class Identifiers:
  species_id: str = ''

def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
  """Gets species from an msa sequence identifier.

  The sequence identifier has the format specified by
  _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.
  An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`

  Args:
    msa_sequence_identifier: a sequence identifier.

  Returns:
    An `Identifiers` instance with species_id. These
    can be empty in the case where no identifier was found.
  """
  matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
  if matches:
    return Identifiers(
        species_id=matches.group('SpeciesIdentifier'))
  return Identifiers()


def _extract_sequence_identifier(description: str) -> Optional[str]:
  """Extracts sequence identifier from description. Returns None if no match."""
  split_description = description.split()
  if split_description:
    return split_description[0].partition('/')[0]
  else:
    return None


def get_identifiers(description: str) -> Identifiers:
  """Computes extra MSA features from the description."""
  sequence_identifier = _extract_sequence_identifier(description)
  if sequence_identifier is None:
    return Identifiers()
  else:
    return _parse_sequence_identifier(sequence_identifier)


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(msas: Sequence[Msa]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  species_ids = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = get_identifiers(
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


def add_assembly_features(
    all_chain_features: MutableMapping[str, FeatureDict],
    ) -> MutableMapping[str, FeatureDict]:
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
    grouped_chains[seq_to_entity_id[seq]].append(chain_features)

  new_all_chain_features = {}
  chain_id = 1
  for entity_id, group_chain_features in grouped_chains.items():
    for sym_id, chain_features in enumerate(group_chain_features, start=1):
      new_all_chain_features[
          f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
      seq_length = chain_features['seq_length']
      chain_features['asym_id'] = chain_id * np.ones(seq_length)
      chain_features['sym_id'] = sym_id * np.ones(seq_length)
      chain_features['entity_id'] = entity_id * np.ones(seq_length)
      chain_id += 1

  return new_all_chain_features


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


def convert_monomer_features(
    monomer_features: FeatureDict,
    chain_id: str) -> FeatureDict:
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



def process_single_chain(chain_id, sequence, description, a3m_file, is_homomer_or_monomer, hhr_file=None):
    with open(a3m_file, "r") as fp:
        msa = parse_a3m(fp.read())
    data = {"msa": msa.sequences, "deletion_matrix": msa.deletion_matrix}
    sequence = msa.sequences[0]
    num_res = len(sequence)
    msas, deletion_matrices = zip(*[(data["msa"], data["deletion_matrix"])])
    chain_features = make_sequence_features(sequence=sequence, description="none",num_res=num_res)
    chain_features.update(make_msa_features((msa,)))
    if hhr_file is not None:
        with open (hhr_file) as f:
            hhr = f.read()
        pdb_temp = get_template_hits(output_string=hhr)
        templates_result = get_templates(query_sequence=sequence, hits=pdb_temp)
        chain_features.update(templates_result.features)

    if not is_homomer_or_monomer:
        all_seq_features = make_msa_features([msa])
        valid_feats = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int',
                        'msa_uniprot_accession_identifiers','msa_species_identifiers',)
        feats = {f'{k}_all_seq': v for k, v in all_seq_features.items()
             if k in valid_feats}

        chain_features.update(feats)

    return chain_features

def _get_pdb_id_and_chain(hit: TemplateHit) -> Tuple[str, str]:
  """Returns PDB id and chain id for an HHSearch Hit."""
  # PDB ID: 4 letters. Chain ID: 1+ alphanumeric letters or "." if unknown.
  id_match = re.match(r'[a-zA-Z\d]{4}_[a-zA-Z0-9.]+', hit.name)
  if not id_match:
    raise ValueError(f'hit.name did not start with PDBID_chain: {hit.name}')
  pdb_id, chain_id = id_match.group(0).split('_')
  return pdb_id.lower(), chain_id

class Error(Exception):
  """Base class for exceptions."""

# Prefilter exceptions.
class PrefilterError(Exception):
  """A base class for template prefilter exceptions."""

class NoChainsError(Error):
  """An error indicating that template mmCIF didn't have any chains."""
class NoAtomDataInTemplateError(Error):
  """An error indicating that template mmCIF didn't contain atom positions."""


class TemplateAtomMaskAllZerosError(Error):
  """An error indicating that template mmCIF had all atom positions masked."""


class QueryToTemplateAlignError(Error):
  """An error indicating that the query can't be aligned to the template."""


class CaDistanceError(Error):
  """An error indicating that a CA atom distance exceeds a threshold."""
class MultipleChainsError(Error):
  """An error indicating that multiple chains were found for a given ID."""

class DateError(PrefilterError):
  """An error indicating that the hit date was after the max allowed date."""


class AlignRatioError(PrefilterError):
  """An error indicating that the hit align ratio to the query was too small."""


class DuplicateError(PrefilterError):
  """An error indicating that the hit was an exact subsequence of the query."""


class LengthError(PrefilterError):
  """An error indicating that the hit was too short."""

def _build_query_to_hit_index_mapping(
    hit_query_sequence: str,
    hit_sequence: str,
    indices_hit: Sequence[int],
    indices_query: Sequence[int],
    original_query_sequence: str) -> Mapping[int, int]:
  """Gets mapping from indices in original query sequence to indices in the hit.

  hit_query_sequence and hit_sequence are two aligned sequences containing gap
  characters. hit_query_sequence contains only the part of the original query
  sequence that matched the hit. When interpreting the indices from the .hhr, we
  need to correct for this to recover a mapping from original query sequence to
  the hit sequence.

  Args:
    hit_query_sequence: The portion of the query sequence that is in the .hhr
      hit
    hit_sequence: The portion of the hit sequence that is in the .hhr
    indices_hit: The indices for each aminoacid relative to the hit sequence
    indices_query: The indices for each aminoacid relative to the original query
      sequence
    original_query_sequence: String describing the original query sequence.

  Returns:
    Dictionary with indices in the original query sequence as keys and indices
    in the hit sequence as values.
  """
  # If the hit is empty (no aligned residues), return empty mapping
  if not hit_query_sequence:
    return {}

  # Remove gaps and find the offset of hit.query relative to original query.
  hhsearch_query_sequence = hit_query_sequence.replace('-', '')
  hit_sequence = hit_sequence.replace('-', '')
  hhsearch_query_offset = original_query_sequence.find(hhsearch_query_sequence)

  # Index of -1 used for gap characters. Subtract the min index ignoring gaps.
  min_idx = min(x for x in indices_hit if x > -1)
  fixed_indices_hit = [
      x - min_idx if x > -1 else -1 for x in indices_hit
  ]

  min_idx = min(x for x in indices_query if x > -1)
  fixed_indices_query = [x - min_idx if x > -1 else -1 for x in indices_query]

  # Zip the corrected indices, ignore case where both seqs have gap characters.
  mapping = {}
  for q_i, q_t in zip(fixed_indices_query, fixed_indices_hit):
    if q_t != -1 and q_i != -1:
      if (q_t >= len(hit_sequence) or
          q_i + hhsearch_query_offset >= len(original_query_sequence)):
        continue
      mapping[q_i + hhsearch_query_offset] = q_t

  return mapping

def _find_template_in_pdb(
    template_chain_id: str,
    template_sequence: str,
    mmcif_object: mmcif_parsing.MmcifObject) -> Tuple[str, str, int]:
  """Tries to find the template chain in the given pdb file.

  This method tries the three following things in order:
    1. Tries if there is an exact match in both the chain ID and the sequence.
       If yes, the chain sequence is returned. Otherwise:
    2. Tries if there is an exact match only in the sequence.
       If yes, the chain sequence is returned. Otherwise:
    3. Tries if there is a fuzzy match (X = wildcard) in the sequence.
       If yes, the chain sequence is returned.
  If none of these succeed, a SequenceNotInTemplateError is thrown.

  Args:
    template_chain_id: The template chain ID.
    template_sequence: The template chain sequence.
    mmcif_object: The PDB object to search for the template in.

  Returns:
    A tuple with:
    * The chain sequence that was found to match the template in the PDB object.
    * The ID of the chain that is being returned.
    * The offset where the template sequence starts in the chain sequence.

  Raises:
    SequenceNotInTemplateError: If no match is found after the steps described
      above.
  """
  # Try if there is an exact match in both the chain ID and the (sub)sequence.
  pdb_id = mmcif_object.file_id
  chain_sequence = mmcif_object.chain_to_seqres.get(template_chain_id)
  if chain_sequence and (template_sequence in chain_sequence):
    logging.info(
        'Found an exact template match %s_%s.', pdb_id, template_chain_id)
    mapping_offset = chain_sequence.find(template_sequence)
    return chain_sequence, template_chain_id, mapping_offset

  # Try if there is an exact match in the (sub)sequence only.
  for chain_id, chain_sequence in mmcif_object.chain_to_seqres.items():
    if chain_sequence and (template_sequence in chain_sequence):
      logging.info('Found a sequence-only match %s_%s.', pdb_id, chain_id)
      mapping_offset = chain_sequence.find(template_sequence)
      return chain_sequence, chain_id, mapping_offset

  # Return a chain sequence that fuzzy matches (X = wildcard) the template.
  # Make parentheses unnamed groups (?:_) to avoid the 100 named groups limit.
  regex = ['.' if aa == 'X' else '(?:%s|X)' % aa for aa in template_sequence]
  regex = re.compile(''.join(regex))
  for chain_id, chain_sequence in mmcif_object.chain_to_seqres.items():
    match = re.search(regex, chain_sequence)
    if match:
      logging.info('Found a fuzzy sequence-only match %s_%s.', pdb_id, chain_id)
      mapping_offset = match.start()
      return chain_sequence, chain_id, mapping_offset

  # No hits, raise an error.
  raise SequenceNotInTemplateError(
      'Could not find the template sequence in %s_%s. Template sequence: %s, '
      'chain_to_seqres: %s' % (pdb_id, template_chain_id, template_sequence,
                               mmcif_object.chain_to_seqres))


def _realign_pdb_template_to_query(
    old_template_sequence: str,
    template_chain_id: str,
    mmcif_object: mmcif_parsing.MmcifObject,
    old_mapping: Mapping[int, int],
    kalign_binary_path: str) -> Tuple[str, Mapping[int, int]]:
  """Aligns template from the mmcif_object to the query.

  In case PDB70 contains a different version of the template sequence, we need
  to perform a realignment to the actual sequence that is in the mmCIF file.
  This method performs such realignment, but returns the new sequence and
  mapping only if the sequence in the mmCIF file is 90% identical to the old
  sequence.

  Note that the old_template_sequence comes from the hit, and contains only that
  part of the chain that matches with the query while the new_template_sequence
  is the full chain.

  Args:
    old_template_sequence: The template sequence that was returned by the PDB
      template search (typically done using HHSearch).
    template_chain_id: The template chain id was returned by the PDB template
      search (typically done using HHSearch). This is used to find the right
      chain in the mmcif_object chain_to_seqres mapping.
    mmcif_object: A mmcif_object which holds the actual template data.
    old_mapping: A mapping from the query sequence to the template sequence.
      This mapping will be used to compute the new mapping from the query
      sequence to the actual mmcif_object template sequence by aligning the
      old_template_sequence and the actual template sequence.
    kalign_binary_path: The path to a kalign executable.

  Returns:
    A tuple (new_template_sequence, new_query_to_template_mapping) where:
    * new_template_sequence is the actual template sequence that was found in
      the mmcif_object.
    * new_query_to_template_mapping is the new mapping from the query to the
      actual template found in the mmcif_object.

  Raises:
    QueryToTemplateAlignError:
    * If there was an error thrown by the alignment tool.
    * Or if the actual template sequence differs by more than 10% from the
      old_template_sequence.
  """
  aligner = kalign.Kalign(binary_path=kalign_binary_path)
  new_template_sequence = mmcif_object.chain_to_seqres.get(
      template_chain_id, '')

  # Sometimes the template chain id is unknown. But if there is only a single
  # sequence within the mmcif_object, it is safe to assume it is that one.
  if not new_template_sequence:
    if len(mmcif_object.chain_to_seqres) == 1:
      logging.info('Could not find %s in %s, but there is only 1 sequence, so '
                   'using that one.',
                   template_chain_id,
                   mmcif_object.file_id)
      new_template_sequence = list(mmcif_object.chain_to_seqres.values())[0]
    else:
      raise QueryToTemplateAlignError(
          f'Could not find chain {template_chain_id} in {mmcif_object.file_id}. '
          'If there are no mmCIF parsing errors, it is possible it was not a '
          'protein chain.')

  try:
    parsed_a3m = parsers.parse_a3m(
        aligner.align([old_template_sequence, new_template_sequence]))
    old_aligned_template, new_aligned_template = parsed_a3m.sequences
  except Exception as e:
    raise QueryToTemplateAlignError(
        'Could not align old template %s to template %s (%s_%s). Error: %s' %
        (old_template_sequence, new_template_sequence, mmcif_object.file_id,
         template_chain_id, str(e)))

  logging.info('Old aligned template: %s\nNew aligned template: %s',
               old_aligned_template, new_aligned_template)

  old_to_new_template_mapping = {}
  old_template_index = -1
  new_template_index = -1
  num_same = 0
  for old_template_aa, new_template_aa in zip(
      old_aligned_template, new_aligned_template):
    if old_template_aa != '-':
      old_template_index += 1
    if new_template_aa != '-':
      new_template_index += 1
    if old_template_aa != '-' and new_template_aa != '-':
      old_to_new_template_mapping[old_template_index] = new_template_index
      if old_template_aa == new_template_aa:
        num_same += 1

  # Require at least 90 % sequence identity wrt to the shorter of the sequences.
  if float(num_same) / min(
      len(old_template_sequence), len(new_template_sequence)) < 0.9:
    raise QueryToTemplateAlignError(
        'Insufficient similarity of the sequence in the database: %s to the '
        'actual sequence in the mmCIF file %s_%s: %s. We require at least '
        '90 %% similarity wrt to the shorter of the sequences. This is not a '
        'problem unless you think this is a template that should be included.' %
        (old_template_sequence, mmcif_object.file_id, template_chain_id,
         new_template_sequence))

  new_query_to_template_mapping = {}
  for query_index, old_template_index in old_mapping.items():
    new_query_to_template_mapping[query_index] = (
        old_to_new_template_mapping.get(old_template_index, -1))

  new_template_sequence = new_template_sequence.replace('-', '')

  return new_template_sequence, new_query_to_template_mapping

def _get_atom_positions(
    mmcif_object: mmcif_parsing.MmcifObject,
    auth_chain_id: str,
    max_ca_ca_distance: float) -> Tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""
  if(not auth_chain_id in mmcif_object.chain_to_seqres):
      num_res = 1
      all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
      all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                    dtype=np.int64)
      return all_positions, all_positions_mask
  else:
      num_res = len(mmcif_object.chain_to_seqres[auth_chain_id])

  relevant_chains = [c for c in mmcif_object.structure.get_chains()
                     if c.id == auth_chain_id]
  if len(relevant_chains) != 1:
    raise MultipleChainsError(
        f'Expected exactly one chain in structure with id {auth_chain_id}.')
  chain = relevant_chains[0]

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)
  for res_index in range(num_res):
    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
    res_at_position = mmcif_object.seqres_to_structure[auth_chain_id][res_index]
    if not res_at_position.is_missing:
      res = chain[(res_at_position.hetflag,
                   res_at_position.position.residue_number,
                   res_at_position.position.insertion_code)]
      for atom in res.get_atoms():
        atom_name = atom.get_name()
        x, y, z = atom.get_coord()
        if atom_name in residue_constants.atom_order.keys():
          pos[residue_constants.atom_order[atom_name]] = [x, y, z]
          mask[residue_constants.atom_order[atom_name]] = 1.0
        elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
          # Put the coordinates of the selenium atom in the sulphur column.
          pos[residue_constants.atom_order['SD']] = [x, y, z]
          mask[residue_constants.atom_order['SD']] = 1.0

      # Fix naming errors in arginine residues where NH2 is incorrectly
      # assigned to be closer to CD than NH1.
      cd = residue_constants.atom_order['CD']
      nh1 = residue_constants.atom_order['NH1']
      nh2 = residue_constants.atom_order['NH2']
      if (res.get_resname() == 'ARG' and
          all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
          (np.linalg.norm(pos[nh1] - pos[cd]) >
           np.linalg.norm(pos[nh2] - pos[cd]))):
        pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
        mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

    all_positions[res_index] = pos
    all_positions_mask[res_index] = mask
  _check_residue_distances(
      all_positions, all_positions_mask, max_ca_ca_distance)
  return all_positions, all_positions_mask

def _extract_template_features(
    mmcif_object: mmcif_parsing.MmcifObject,
    pdb_id: str,
    mapping: Mapping[int, int],
    template_sequence: str,
    query_sequence: str,
    template_chain_id: str,
    kalign_binary_path: str) -> Tuple[Dict[str, Any], Optional[str]]:
  """Parses atom positions in the target structure and aligns with the query.

  Atoms for each residue in the template structure are indexed to coincide
  with their corresponding residue in the query sequence, according to the
  alignment mapping provided.

  Args:
    mmcif_object: mmcif_parsing.MmcifObject representing the template.
    pdb_id: PDB code for the template.
    mapping: Dictionary mapping indices in the query sequence to indices in
      the template sequence.
    template_sequence: String describing the amino acid sequence for the
      template protein.
    query_sequence: String describing the amino acid sequence for the query
      protein.
    template_chain_id: String ID describing which chain in the structure proto
      should be used.
    kalign_binary_path: The path to a kalign executable used for template
        realignment.

  Returns:
    A tuple with:
    * A dictionary containing the extra features derived from the template
      protein structure.
    * A warning message if the hit was realigned to the actual mmCIF sequence.
      Otherwise None.

  Raises:
    NoChainsError: If the mmcif object doesn't contain any chains.
    SequenceNotInTemplateError: If the given chain id / sequence can't
      be found in the mmcif object.
    QueryToTemplateAlignError: If the actual template in the mmCIF file
      can't be aligned to the query.
    NoAtomDataInTemplateError: If the mmcif object doesn't contain
      atom positions.
    TemplateAtomMaskAllZerosError: If the mmcif object doesn't have any
      unmasked residues.
  """
  if mmcif_object is None or not mmcif_object.chain_to_seqres:
    raise NoChainsError('No chains in PDB: %s_%s' % (pdb_id, template_chain_id))

  warning = None
  try:
    seqres, chain_id, mapping_offset = _find_template_in_pdb(
        template_chain_id=template_chain_id,
        template_sequence=template_sequence,
        mmcif_object=mmcif_object)
  except SequenceNotInTemplateError:
    # If PDB70 contains a different version of the template, we use the sequence
    # from the mmcif_object.
    chain_id = template_chain_id
    warning = (
        f'The exact sequence {template_sequence} was not found in '
        f'{pdb_id}_{chain_id}. Realigning the template to the actual sequence.')
    logging.warning(warning)
    # This throws an exception if it fails to realign the hit.
    seqres, mapping = _realign_pdb_template_to_query(
        old_template_sequence=template_sequence,
        template_chain_id=template_chain_id,
        mmcif_object=mmcif_object,
        old_mapping=mapping,
        kalign_binary_path=kalign_binary_path)
    logging.info('Sequence in %s_%s: %s successfully realigned to %s',
                 pdb_id, chain_id, template_sequence, seqres)
    # The template sequence changed.
    template_sequence = seqres
    # No mapping offset, the query is aligned to the actual sequence.
    mapping_offset = 0

  try:
    # Essentially set to infinity - we don't want to reject templates unless
    # they're really really bad.
    all_atom_positions, all_atom_mask = _get_atom_positions(
        mmcif_object, chain_id, max_ca_ca_distance=150.0)
  except (CaDistanceError, KeyError) as ex:
    raise NoAtomDataInTemplateError(
        'Could not get atom data (%s_%s): %s' % (pdb_id, chain_id, str(ex))
        ) from ex

  all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
  all_atom_masks = np.split(all_atom_mask, all_atom_mask.shape[0])

  output_templates_sequence = []
  templates_all_atom_positions = []
  templates_all_atom_masks = []

  for _ in query_sequence:
    # Residues in the query_sequence that are not in the template_sequence:
    templates_all_atom_positions.append(
        np.zeros((residue_constants.atom_type_num, 3)))
    templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
    output_templates_sequence.append('-')

  for k, v in mapping.items():
    template_index = v + mapping_offset
    templates_all_atom_positions[k] = all_atom_positions[template_index][0]
    templates_all_atom_masks[k] = all_atom_masks[template_index][0]
    output_templates_sequence[k] = template_sequence[v]

  # Alanine (AA with the lowest number of atoms) has 5 atoms (C, CA, CB, N, O).
  if np.sum(templates_all_atom_masks) < 5:
    raise TemplateAtomMaskAllZerosError(
        'Template all atom mask was all zeros: %s_%s. Residue range: %d-%d' %
        (pdb_id, chain_id, min(mapping.values()) + mapping_offset,
         max(mapping.values()) + mapping_offset))

  output_templates_sequence = ''.join(output_templates_sequence)

  templates_aatype = residue_constants.sequence_to_onehot(
      output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)

  return (
      {
          'template_all_atom_positions': np.array(templates_all_atom_positions),
          'template_all_atom_masks': np.array(templates_all_atom_masks),
          'template_sequence': output_templates_sequence.encode(),
          'template_aatype': np.array(templates_aatype),
          'template_domain_names': f'{pdb_id.lower()}_{chain_id}'.encode(),
      },
      warning)

def _is_after_cutoff(
    pdb_id: str,
    release_dates: Mapping[str, datetime.datetime],
    release_date_cutoff: Optional[datetime.datetime]) -> bool:
  """Checks if the template date is after the release date cutoff.

  Args:
    pdb_id: 4 letter pdb code.
    release_dates: Dictionary mapping PDB ids to their structure release dates.
    release_date_cutoff: Max release date that is valid for this query.

  Returns:
    True if the template release date is after the cutoff, False otherwise.
  """
  if release_date_cutoff is None:
    raise ValueError('The release_date_cutoff must not be None.')
  if pdb_id in release_dates:
    return release_dates[pdb_id] > release_date_cutoff
  else:
    # Since this is just a quick prefilter to reduce the number of mmCIF files
    # we need to parse, we don't have to worry about returning True here.
    return False

def _assess_hhsearch_hit(
    hit: parsers.TemplateHit,
    hit_pdb_code: str,
    query_sequence: str,
    release_dates: Mapping[str, datetime.datetime],
    release_date_cutoff: datetime.datetime,
    max_subsequence_ratio: float = 0.95,
    min_align_ratio: float = 0.1) -> bool:
  """Determines if template is valid (without parsing the template mmcif file).

  Args:
    hit: HhrHit for the template.
    hit_pdb_code: The 4 letter pdb code of the template hit. This might be
      different from the value in the actual hit since the original pdb might
      have become obsolete.
    query_sequence: Amino acid sequence of the query.
    release_dates: Dictionary mapping pdb codes to their structure release
      dates.
    release_date_cutoff: Max release date that is valid for this query.
    max_subsequence_ratio: Exclude any exact matches with this much overlap.
    min_align_ratio: Minimum overlap between the template and query.

  Returns:
    True if the hit passed the prefilter. Raises an exception otherwise.

  Raises:
    DateError: If the hit date was after the max allowed date.
    AlignRatioError: If the hit align ratio to the query was too small.
    DuplicateError: If the hit was an exact subsequence of the query.
    LengthError: If the hit was too short.
  """
  aligned_cols = hit.aligned_cols
  align_ratio = aligned_cols / len(query_sequence)

  template_sequence = hit.hit_sequence.replace('-', '')
  length_ratio = float(len(template_sequence)) / len(query_sequence)

  # Check whether the template is a large subsequence or duplicate of original
  # query. This can happen due to duplicate entries in the PDB database.
  duplicate = (template_sequence in query_sequence and
               length_ratio > max_subsequence_ratio)

  if _is_after_cutoff(hit_pdb_code, release_dates, release_date_cutoff):
    raise DateError(f'Date ({release_dates[hit_pdb_code]}) > max template date '
                    f'({release_date_cutoff}).')

  if align_ratio <= min_align_ratio:
    raise AlignRatioError('Proportion of residues aligned to query too small. '
                          f'Align ratio: {align_ratio}.')

  if duplicate:
    raise DuplicateError('Template is an exact subsequence of query with large '
                         f'coverage. Length ratio: {length_ratio}.')

  if len(template_sequence) < 10:
    raise LengthError(f'Template too short. Length: {len(template_sequence)}.')

  return True

def _check_residue_distances(all_positions: np.ndarray,
                             all_positions_mask: np.ndarray,
                             max_ca_ca_distance: float):
  """Checks if the distance between unmasked neighbor residues is ok."""
  ca_position = residue_constants.atom_order['CA']
  prev_is_unmasked = False
  prev_calpha = None
  for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
    this_is_unmasked = bool(mask[ca_position])
    if this_is_unmasked:
      this_calpha = coords[ca_position]
      if prev_is_unmasked:
        distance = np.linalg.norm(this_calpha - prev_calpha)
        if distance > max_ca_ca_distance:
          raise CaDistanceError(
              'The distance between residues %d and %d is %f > limit %f.' % (
                  i, i + 1, distance, max_ca_ca_distance))
      prev_calpha = this_calpha
    prev_is_unmasked = this_is_unmasked
    
def _process_single_hit(
    query_sequence: str,
    hit: TemplateHit,
    mmcif_dir: str,
    max_template_date: datetime.datetime,
    release_dates: Mapping[str, datetime.datetime],
    # obsolete_pdbs: Mapping[str, Optional[str]],
    kalign_binary_path: str,
    ) -> SingleHitResult:

    hit_pdb_code, hit_chain_id = _get_pdb_id_and_chain(hit)
    try:
        _assess_hhsearch_hit(
            hit=hit,
            hit_pdb_code=hit_pdb_code,
            query_sequence=query_sequence,
            release_dates=release_dates,
            release_date_cutoff=max_template_date)
    except PrefilterError as e:
        msg = f'hit {hit_pdb_code}_{hit_chain_id} did not pass prefilter: {str(e)}'
        logging.info(msg)

        return SingleHitResult(features=None, error=None, warning=None)
    
    mapping = _build_query_to_hit_index_mapping(
      hit.query, hit.hit_sequence, hit.indices_hit, hit.indices_query,
      query_sequence)

    template_sequence = hit.hit_sequence.replace('-', '')

    cif_path = os.path.join(mmcif_dir, hit_pdb_code + '.cif')
    logging.debug('Reading PDB entry from %s. Query: %s, template: %s', cif_path,
                query_sequence, template_sequence)

    with open(cif_path, 'r') as f:
        cif_string = f.read()

    parsing_result = mmcif_parsing.parse(
      file_id=hit_pdb_code, mmcif_string=cif_string)
    if parsing_result.mmcif_object is not None:
        hit_release_date = datetime.datetime.strptime(parsing_result.mmcif_object.header['release_date'], '%Y-%m-%d')
        if hit_release_date > max_template_date:
            error = ('Template %s date (%s) > max template date (%s).' %
               (hit_pdb_code, hit_release_date, max_template_date))
            logging.debug(error)
            return SingleHitResult(features=None, error=None, warning=None)

    try:
        features, realign_warning = _extract_template_features(
            mmcif_object=parsing_result.mmcif_object,
            pdb_id=hit_pdb_code,
            mapping=mapping,
            template_sequence=template_sequence,
            query_sequence=query_sequence,
            template_chain_id=hit_chain_id,
            kalign_binary_path=kalign_binary_path)
        if hit.sum_probs is None:
            features['template_sum_probs'] = [0]
        else:
            features['template_sum_probs'] = [hit.sum_probs]
        return SingleHitResult(
        features=features, error=None, warning=realign_warning)
    except (NoChainsError, NoAtomDataInTemplateError,
          TemplateAtomMaskAllZerosError) as e:
    # These 3 errors indicate missing mmCIF experimental data rather than a
    # problem with the template search, so turn them into warnings.
        warning = ('%s_%s (sum_probs: %s, rank: %s): feature extracting errors: '
               '%s, mmCIF parsing errors: %s'
               % (hit_pdb_code, hit_chain_id, hit.sum_probs, hit.index,
                  str(e), parsing_result.errors))
        return SingleHitResult(features=None, error=warning, warning=None)
    except Error as e:
        error = ('%s_%s (sum_probs: %.2f, rank: %d): feature extracting errors: '
             '%s, mmCIF parsing errors: %s'
             % (hit_pdb_code, hit_chain_id, hit.sum_probs, hit.index,
                str(e), parsing_result.errors))
        return SingleHitResult(features=None, error=error, warning=None)

def get_templates(
      query_sequence: str,
      hits: Sequence[TemplateHit],
      mmcif_dir='/pool-data/data/thu/mmcif_files/',
      max_hits=20,
      max_template_dates='2021-11-01',
      release_dates={},
      kalig_bin=shutil.which('kalign')) -> TemplateSearchResult:
    template_features = {}
    max_template_dates = datetime.datetime.strptime(max_template_dates, '%Y-%m-%d')
    for template_feature_name in TEMPLATE_FEATURES:
        template_features[template_feature_name] = []

    num_hits = 0
    errors = []
    warnings = []
    for hit in sorted(hits, key=lambda x: x.sum_probs, reverse=True):
      # We got all the templates we wanted, stop processing hits.
        if num_hits >= max_hits:
            break

        result = _process_single_hit(
          query_sequence=query_sequence,
          hit=hit,
          mmcif_dir=mmcif_dir,
          max_template_date=max_template_dates,
          release_dates=release_dates,
          kalign_binary_path=kalig_bin)
        if result.error:
            errors.append(result.error)
        if result.warning:
            warnings.append(result.warning)
        if result.features is None:
            logging.info('Skipped invalid hit %s, error: %s, warning: %s',
                     hit.name, result.error, result.warning)
        else:
        # Increment the hit counter, since we got features out of this hit.
            num_hits += 1
            for k in template_features:
                template_features[k].append(result.features[k])

    for name in template_features:
        if num_hits > 0:
            template_features[name] = np.stack(template_features[name], axis=0).astype(TEMPLATE_FEATURES[name])
        else:
            # Make sure the feature has correct dtype even if empty.
            template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])

    return TemplateSearchResult(
        features=template_features, errors=errors, warnings=warnings)

def process(input_file, config_multimer):
    f = open(input_file)
    inputs = json.load(f)
    sequences = []
    descriptions = []
    for chain in inputs.keys():
        sequences.append(inputs[chain]['sequence'])
        descriptions.append(inputs[chain]['description'])

    is_homomer_or_monomer = len(set(sequences)) == 1
    all_chain_features = {}
    sequence_features = {}
    
    for chain in inputs.keys():
        if(inputs[chain]['sequence'] in sequence_features):
            all_chain_features[chain] = copy.deepcopy(sequence_features[inputs[chain]['sequence']])
            continue
        if('hhr' in inputs[chain].keys()):
            hhr = inputs[chain]['hhr']
        else:
            hhr = None
        chain_features = process_single_chain(chain, inputs[chain]['sequence'], inputs[chain]['description'], inputs[chain]['a3m'],  is_homomer_or_monomer, hhr)
        chain_features = convert_monomer_features(chain_features, chain_id=chain)
        all_chain_features[chain] = chain_features
        sequence_features[inputs[chain]['sequence']] = chain_features
    
    all_chain_features = add_assembly_features(all_chain_features)
    np_example = feature_processing.pair_and_merge(all_chain_features=all_chain_features)
    np_example = pad_msa(np_example, 512)

    return np_example

    

