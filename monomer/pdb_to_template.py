import sys 
sys.path.insert(1, '../')
import numpy as np
import dataclasses
import io
from Bio.PDB import PDBParser
from alphadock import residue_constants
from typing import Any, Mapping, Optional

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
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    res_name: np.ndarray



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
    residue_index = []
    b_factors = []
    res_name = []

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
        res_name.append(res_shortname)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        res_name=np.array(res_name),
    )

#if __name__ == '__main__':
#    with open('6A77/6A77_coeff0_6_model_1_multimer_0.pdb', "r") as fp:
#        pdb_string = fp.read()
#    protein_object_A = from_pdb_string(pdb_string, 'A')
#    protein_object_H = from_pdb_string(pdb_string, 'H')
#    protein_object_L = from_pdb_string(pdb_string, 'L')
#    template_aatype = np.concatenate((protein_object_A.aatype, protein_object_H.aatype, protein_object_L.aatype), axis=0)
#    template_all_atom_pos = np.concatenate((protein_object_A.atom_positions, protein_object_H.atom_positions, protein_object_L.atom_positions), axis=0)
#    template_all_atom_mask = np.concatenate((protein_object_A.atom_mask, protein_object_H.atom_mask, protein_object_L.atom_mask), axis=0)
#    print(template_aatype.shape, template_all_atom_pos.shape, template_all_atom_mask.shape)
    
