import prody
from io import StringIO
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdFMCS

from alphadock import utils


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
    new_coords = utils.apply_prody_transform(mol.GetConformer().GetPositions(), tr)
    change_mol_coords(mol, new_coords)
    return mol


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
