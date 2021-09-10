import traceback
import random
import itertools
import numpy as np
import prody
from path import Path
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from functools import partial

import torch

import utils_loc
from amino_acids import residue_bonds_noh, FUNCTIONAL_GROUPS, ATOM_TYPES
from sblu.ft import read_rotations, FTRESULT_DTYPE, read_ftresults, apply_ftresult, apply_ftresults_atom_group
import dgl


DTYPE_FLOAT = np.float32


ELEMENTS = {x[1]: x[0] for x in enumerate(['I', 'S', 'F', 'N', 'C', 'CL', 'BR', 'O', 'P'])}

HYBRIDIZATIONS = {
    'S': 0,
    'SP': 1,
    'SP2': 2,
    'SP3': 3,
    'SP3D': 4,
    'SP3D2': 5
}

MAX_VALENCE = 7

MAX_HS = 3

MAX_DEGREE = 5

CHIRALITY = {
    'CHI_TETRAHEDRAL_CW': 0,
    'CHI_TETRAHEDRAL_CCW': 1
}


def atom_to_vector(atom):
    vec = [0] * len(ELEMENTS)
    vec[ELEMENTS[atom.GetSymbol().upper()]] = 1

    # chirality
    new_vec = [0] * (len(CHIRALITY) + 1)
    new_vec[CHIRALITY.get(str(atom.GetChiralTag()), len(CHIRALITY))] = 1
    vec += new_vec

    # formal charge
    fcharge = atom.GetFormalCharge()
    new_vec = [0] * 3
    if fcharge < 0:
        new_vec[0] = 1
    elif fcharge > 0:
        new_vec[1] = 1
    else:
        new_vec[2] = 1
    vec += new_vec

    # aromaticity
    new_vec = [0, 0]
    new_vec[int(atom.GetIsAromatic())] = 1
    vec += new_vec

    # degree
    new_vec = [0] * (MAX_DEGREE + 1)
    new_vec[int(min(atom.GetTotalDegree(), MAX_DEGREE))] = 1
    vec += new_vec

    # num Hs
    new_vec = [0] * (MAX_HS + 1)
    new_vec[int(min(atom.GetTotalNumHs(), MAX_HS))] = 1
    vec += new_vec

    # valence
    new_vec = [0] * (MAX_VALENCE + 1)
    new_vec[int(min(atom.GetTotalValence(), MAX_VALENCE))] = 1
    vec += new_vec

    # in ring flag
    new_vec = [0, 0]
    new_vec[int(atom.IsInRing())] = 1
    vec += new_vec

    # is ion
    new_vec = [0, 0]
    new_vec[int(atom.GetTotalDegree() == 0)] = 1
    vec += new_vec

    return np.array(vec, dtype=DTYPE_FLOAT)


BOND_TYPE = {'AROMATIC': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}


def bond_to_vector(bond):
    # bond type
    vec = [0] * len(BOND_TYPE)
    vec[BOND_TYPE[str(bond.GetBondType())]] = 1

    # in ring
    new_vec = [0] * 2
    new_vec[bond.IsInRing()] = 1
    vec += new_vec

    return np.array(vec, dtype=DTYPE_FLOAT)


def mol_to_features(mol):
    mol = Chem.RemoveHs(mol)
    # return
    # [num_atoms, c]
    # [num_atoms, num_atoms, c]
    pass


def template3d_to_features(rec_ag, mol3d, rec_map, mol_map):
    # return
    # [num_res, c]: aatype, torsions, has_coords
    # [num_rec, num_res, c]: distogram
    # [num_atoms, num_atoms, c]: distogram, bond_type
    # [num_rec, num_atoms, c]: distogram
    pass


def rec3d_to_features(ag, residue_confidence):
    # return
    # [num_res, c]: aatype, torsions, has_coords, confidence
    # [num_res, num_res, c]: distogram, resi
    pass


def template3d_to_CEP(rec_ag, mol3d, mol_map):
    # return [num_atoms, c]
    pass


