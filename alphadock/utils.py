import os
import subprocess
import json
import prody
import contextlib
import tempfile
import shutil
import numpy as np
from io import StringIO
from path import Path
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdFMCS

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment


def _get_rdkit_elements():
    pt = Chem.GetPeriodicTable()
    elements = []
    for i in range(1000):
        try:
            elements.append(pt.GetElementSymbol(i))
        except:
            break
    return elements


# TODO: This produces annoying error log, when it hits atom number which is not in the table.
#       Need to fix it somehow.
RDKIT_ELEMENTS = _get_rdkit_elements()


@contextlib.contextmanager
def isolated_filesystem(dir=None, remove=True):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    if dir is None:
        t = tempfile.mkdtemp(prefix='pocketdock-')
    else:
        t = dir
    os.chdir(t)
    try:
        yield t
    except Exception as e:
        #logger.error(f'Error occured, temporary files are in {t}')
        raise
    else:
        os.chdir(cwd)
        if remove:
            try:
                shutil.rmtree(t)
            except (OSError, IOError):
                pass
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def cwd(dir):
    pwd = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(pwd)


def tmp_file(**kwargs):
    handle, fname = tempfile.mkstemp(**kwargs)
    os.close(handle)
    return Path(fname)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def numpy_to_list(arr):
    return [x.item() for x in arr]


def rank_list(l):
    return zip(*sorted(enumerate(np.argsort(l)), key=lambda x: x[1]))


def safe_read_ag(ag) -> prody.Atomic:
    if isinstance(ag, prody.AtomGroup):
        return ag
    elif isinstance(ag, str):
        return prody.parsePDB(ag)
    else:
        raise RuntimeError(f"Can't read atom group, 'ag' has wrong type {type(ag)}")


def read_mol_block(mol_block, removeHs=True):
    if mol_block is None:
        raise RuntimeError(f'Mol block is empty')

    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=removeHs)
    san_res = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL, catchErrors=True)
    if san_res != 0:
        #if san_res == Chem.SANITIZE_PROPERTIES:
        #    logger.warning('Sanitization failed on SANITIZE_PROPERTIES, removing this flag and trying once again')
        #    san_res = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES, catchErrors=True)
        if san_res != 0:
            raise RuntimeError(f'Sanitization failed on {san_res}')

    return mol


def mol_to_ag(mol):
    return prody.parsePDBStream(StringIO(Chem.MolToPDBBlock(mol)))


def ag_to_mol_assign_bonds(ag, mol_template):
    output = StringIO()
    prody.writePDBStream(output, ag)
    ag_mol = AllChem.MolFromPDBBlock(output.getvalue())
    ag_mol = AllChem.AssignBondOrdersFromTemplate(mol_template, ag_mol)
    return ag_mol


def apply_prody_transform(coords, tr):
    return np.dot(coords, tr.getRotation().T) + tr.getTranslation()


def minimize_rmsd(mob_ag, ref_ag, mob_serials=None, ref_serials=None, mob_cset=None, ref_cset=None):
    if mob_serials is not None and ref_serials is not None:
        mob_sel = mob_ag.select('serial ' + ' '.join(map(str, mob_serials)))
        ref_sel = ref_ag.select('serial ' + ' '.join(map(str, ref_serials)))
        mob_s2i = dict(zip(mob_sel.getSerials(), mob_sel.getIndices()))
        ref_s2i = dict(zip(ref_sel.getSerials(), ref_sel.getIndices()))
        mob_ids = [mob_s2i[s] for s in mob_serials]
        ref_ids = [ref_s2i[s] for s in ref_serials]
    else:
        mob_ids = mob_ag.all.getIndices()
        ref_ids = ref_ag.all.getIndices()

    if mob_cset is not None:
        mob_crd = mob_ag.getCoordsets(mob_cset)[mob_ids]
    else:
        mob_crd = mob_ag.getCoords()[mob_ids]

    if ref_cset is not None:
        ref_crd = ref_ag.getCoordsets(ref_cset)[ref_ids]
    else:
        ref_crd = ref_ag.getCoords()[ref_ids]

    tr = prody.calcTransformation(mob_crd, ref_crd)
    rmsd_minimized = prody.calcRMSD(apply_prody_transform(mob_crd, tr), ref_crd)
    transformation = numpy_to_list(tr.getMatrix().flatten())
    return rmsd_minimized, transformation


def change_mol_coords(mol, new_coords, conf_ids=None):
    if len(new_coords.shape) == 2:
        new_coords = [new_coords]

    conf_ids = range(mol.GetNumConformers()) if conf_ids is None else conf_ids

    if len(conf_ids) != len(new_coords):
        raise RuntimeError('Number of coordinate sets is different from the number of conformers')

    for coords_id, conf_id in enumerate(conf_ids):
        conformer = mol.GetConformer(conf_id)
        new_coordset = new_coords[coords_id]

        if mol.GetNumAtoms() != new_coordset.shape[0]:
            raise ValueError(f'Number of atoms is different from the number of coordinates \
            ({mol.GetNumAtoms()} != {new_coordset.shape[0]})')

        for i in range(mol.GetNumAtoms()):
            x, y, z = new_coordset[i]
            conformer.SetAtomPosition(i, Point3D(x, y, z))


def apply_prody_transform_to_rdkit_mol(mol, tr):
    mol = deepcopy(mol)
    new_coords = apply_prody_transform(mol.GetConformer().GetPositions(), tr)
    change_mol_coords(mol, new_coords)
    return mol


def global_align(s1, s2):
    aln = Bio.pairwise2.align.globalds(s1, s2, matlist.blosum62, -14.0, -4.0)
    return aln


def calc_d2mat(crd1, crd2):
    return np.square(crd1[:, None, :] - crd2[None, :, :]).sum(2)


def calc_dmat(crd1, crd2):
    return np.sqrt(calc_d2mat(crd1, crd2))


def calc_mcs(mol1, mol2, mcs_flags=[], timeout=60):
    if 'aa' in mcs_flags:
        atomcompare = rdFMCS.AtomCompare.CompareAny
    elif 'ai' in mcs_flags:
        # CompareIsotopes matches based on the isotope label
        # isotope labels can be used to implement user-defined atom types
        atomcompare = rdFMCS.AtomCompare.CompareIsotopes
    else:
        atomcompare = rdFMCS.AtomCompare.CompareElements

    if 'ba' in mcs_flags:
        bondcompare = rdFMCS.BondCompare.CompareAny
    elif 'be' in mcs_flags:
        bondcompare = rdFMCS.BondCompare.CompareOrderExact
    else:
        bondcompare = rdFMCS.BondCompare.CompareOrder

    if 'v' in mcs_flags:
        matchvalences = True
    else:
        matchvalences = False

    if 'chiral' in mcs_flags:
        matchchiraltag = True
    else:
        matchchiraltag = False

    if 'r' in mcs_flags:
        ringmatchesringonly = True
    else:
        ringmatchesringonly = False

    if 'cr' in mcs_flags:
        completeringsonly = True
    else:
        completeringsonly = False

    maximizebonds = True

    mols = [mol1, mol2]
    try:
        mcs_result = rdFMCS.FindMCS(mols,
                                    timeout=timeout,
                                    atomCompare=atomcompare,
                                    bondCompare=bondcompare,
                                    matchValences=matchvalences,
                                    ringMatchesRingOnly=ringmatchesringonly,
                                    completeRingsOnly=completeringsonly,
                                    matchChiralTag=matchchiraltag,
                                    maximizeBonds=maximizebonds)
    except:
        # sometimes Boost (RDKit uses it) errors occur
        raise RuntimeError('MCS calculation failed')
    if mcs_result.canceled:
        raise RuntimeError('MCS calculation ran out of time')

    return mcs_result.smartsString, mcs_result.numAtoms, mcs_result.numBonds

