import sys 
sys.path.insert(1, '../')
import numpy as np
import dataclasses
import io
from Bio.PDB import PDBParser
from alphadock import residue_constants
from typing import Any, Mapping, Optional
from multimer import pipeline_multimer
from multimer import mmcif_parsing
import os

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    #residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]



def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                "Only single chain PDBs are supported when chain_id not specified. "
                f"Found {len(chains)} chains."
            )
        else:
            chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    #residue_index = []
    b_factors = []

    for res in chain:
        #if res.id[2] != " ":
        #    raise ValueError(
        #        f"PDB contains an insertion code at chain {chain.id} and residue "
        #        f"index {res.id[1]}. These are not supported."
        #    )
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
        #residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        #residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
    )

def map_seq_renumber(renum_seq_file):
    lines = open(renum_seq_file, 'r').readlines()
    seq_renum = {}
    renum_list = {}
    renum_mask = np.zeros(len(lines), dtype=np.float32)
    chain_type = lines[0][0]
    h1 = ['26', '27', '28', '29', '30', '31', '32']
    h2 = ['52', '53', '54', '55', '56']
    h3 = ['96', '97', '98', '99', '100', '101']
    l1 = ['26', '27', '28', '29', '30', '31', '32']
    l2 = ['50', '51', '52']
    l3 = ['91', '92', '93', '94', '95', '96']
    h_chain = h1 + h2 + h3
    l_chain = l1 + l2 + l3
    i=0
    for line in lines:
        line = line.strip()
        line = line.split()
        renumber = line[0][1:]
        residue = line[1]
        seq_renum[renumber] = residue
        renum_list[renumber] = i
        if(chain_type=='H'):
            for h in h_chain:
                if h in renumber:
                    renum_mask[i] = 1
        else:
            for l in l_chain:
                if l in renumber:
                    renum_mask[i] = 1
        i += 1
    return seq_renum, renum_list, renum_mask


def align_seq_pdb(renum_seq_file, renum_pdb_file, chain_id):
    map_seq_renum, renumber_list, renum_mask = map_seq_renumber(renum_seq_file)
    num_res = len(renumber_list)
    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    p = PDBParser()
    struc = p.get_structure("", renum_pdb_file)
    model = list(struc.get_models())[0]
    chain = model[chain_id]
    for res in chain:
        res_id = ''.join([str(res.id[1]), res.id[2]]).strip()
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        if res_id in renumber_list and res_shortname == map_seq_renum[res_id]:
            position = renumber_list[res_id]
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                all_atom_positions[position][residue_constants.atom_order[atom.name]] = atom.coord
                all_atom_mask[position][residue_constants.atom_order[atom.name]] = 1.0
    return all_atom_positions, all_atom_mask, renum_mask

def align_antigen_seq(ang_cif_seq, pdb_file, antigen_chain):
    with open(pdb_file, "r") as fp:
        pdb_string = fp.read()
    protein_object = from_pdb_string(pdb_string, antigen_chain)
    pdb_seq = _aatype_to_str_sequence(protein_object.aatype)
    aligner = kalign.Kalign(binary_path=shutil.which('kalign'))
    parsed_a3m = parsers.parse_a3m(aligner.align([ang_cif_seq , pdb_seq]))
    true_aligned_seq, pdb_aligned_seq = parsed_a3m.sequences
    true_to_pdb_seq_mapping = {}
    true_seq_index = -1
    pdb_seq_index = -1
    
    for true_seq_res, pdb_seq_res in zip(true_aligned_seq, pdb_aligned_seq):
        if true_seq_res != '-':
            true_seq_index += 1
        if pdb_seq_res != '-':
            pdb_seq_index += 1
        if true_seq_res != '-' and pdb_seq_res != '-':
            true_to_pdb_seq_mapping[true_seq_index] = pdb_seq_index
    num_res = len(ang_cif_seq)
    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    for i, j in true_to_pdb_seq_mapping.items():
        all_atom_positions[i] = protein_object.atom_positions[j]
        all_atom_mask[i] = 1.0
    renum_mask = np.zeros(num_res, dtype=np.float32)
    return all_atom_positions, all_atom_mask, renum_mask

def make_antigen_features(sequence, cif_file, chain_id):
    file_id = os.path.basename(cif_file)[:-4]
    with open(cif_file, 'r') as f:
        mmcif_string = f.read()
    mmcif_obj = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string).mmcif_object
    all_atom_positions, all_atom_mask = pipeline_multimer._get_atom_positions(
        mmcif_obj, chain_id, max_ca_ca_distance=15000.0
    )
    assert all_atom_positions.shape[0] == len(sequence)
    renum_mask = np.zeros(len(sequence), dtype=np.float32)
    return all_atom_positions, all_atom_mask, renum_mask


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def make_pdb_features(
    all_atom_positions,
    all_atom_mask,
    renum_mask,
    sequence: str,
    description: str,
    resolution: float
):
    pdb_feats = {}
    pdb_feats.update(
        pipeline_multimer.make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(sequence),
        )
    )

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([resolution]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(0.).astype(np.float32)
    pdb_feats["renum_mask"] = renum_mask

    return pdb_feats

# from multimer import kalign
# import shutil
# from multimer import mmcif_parsing
# if __name__ == '__main__':
   # with open('/storage/thu/antibodies_all_structures/all_structures/chothia/7kqg.pdb', "r") as fp:
   #     pdb_string = fp.read()
   # protein_object_A = from_pdb_string(pdb_string, 'A')
   # # protein_object_H = from_pdb_string(pdb_string, 'H')
   # # protein_object_L = from_pdb_string(pdb_string, 'L')
   # # template_aatype = np.concatenate((protein_object_A.aatype, protein_object_H.aatype, protein_object_L.aatype), axis=0)
   # # template_all_atom_pos = np.concatenate((protein_object_A.atom_positions, protein_object_H.atom_positions, protein_object_L.atom_positions), axis=0)
   # # template_all_atom_mask = np.concatenate((protein_object_A.atom_mask, protein_object_H.atom_mask, protein_object_L.atom_mask), axis=0)
   # true_seq = 'PLTTTPTKSYFANLKGTRTRGKLCPDCLNCTDLDVALGRPMCVGTTPSAKASILHEVKPVTSGCFPIMHDRTKIRQLPNLLRGYENIRLSTQNVIDAEKAPGGPYRLGTSGSCPNATSKSGFFATMAWAVPKDNNKNATNPLTVEVPYICTEGEDQITVWGFHSDDKTQMKNLYGDSNPQKFTSSANGVTTHYVSQIGSFPDQTEDGGLPQSGRIVVDYMMQKPGKTGTIVYQRGVLLPQKVWCASGRSKVIKGSLPLIGEADCLHEKYGGLNKSKPYYTGEHAKAIGNCPIWVKTPLK'
   # query_seq = _aatype_to_str_sequence(protein_object_A.aatype)
   # aligner = kalign.Kalign(binary_path=shutil.which('kalign'))
   # parsed_a3m = parsers.parse_a3m(aligner.align([true_seq , query_seq]))
   # old_aligned_template, new_aligned_template = parsed_a3m.sequences
   # old_to_new_template_mapping = {}
   # old_template_index = -1
   # new_template_index = -1
   # num_same = 0
   # for old_template_aa, new_template_aa in zip(
   #         old_aligned_template, new_aligned_template):
   #     if old_template_aa != '-':
   #         old_template_index += 1
   #     if new_template_aa != '-':
   #         new_template_index += 1
   #     if old_template_aa != '-' and new_template_aa != '-':
   #         old_to_new_template_mapping[old_template_index] = new_template_index
   #         if old_template_aa == new_template_aa:
   #             num_same += 1
   # print(old_to_new_template_mapping)
   # with open('/data/thu/af_database/pdb_mmcif/mmcif_files/7n0u.cif', 'r') as f:
   #     mmcif_string = f.read()
   # mmcif_obj = mmcif_parsing.parse(file_id="4hkx", mmcif_string=mmcif_string).mmcif_object
   # print(mmcif_obj.chain_to_seqres.keys())


