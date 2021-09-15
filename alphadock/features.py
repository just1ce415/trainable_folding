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
from collections import OrderedDict

from amino_acids import d_3aa
from config import DATA_DIR
import utils


DTYPE_FLOAT = np.float32

ELEMENTS = {x[1]: x[0] for x in enumerate(['I', 'S', 'F', 'N', 'C', 'CL', 'BR', 'O', 'P', 'X'])}

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

BOND_TYPE = {'AROMATIC': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3}

AATYPE = sorted(d_3aa.keys()) + ['X', '-']

REC_DISTOGRAM = {
    'min': 3.25,
    'max': 51,
    'num_bins': 39
}

LIG_DISTOGRAM = {
    'min': 3.25,
    'max': 50.75,
    'num_bins': 39
}

REC_LIG_DISTOGRAM = {
    'min': 3.25,
    'max': 50.75,
    'num_bins': 39
}

RESIGRAM_MAX = 32


def atom_to_vector(atom):
    vec = [0] * len(ELEMENTS)
    vec[ELEMENTS.get(atom.GetSymbol().upper(), ELEMENTS['X'])] = 1

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


def bond_to_vector(bond):
    # bond type
    vec = [0] * len(BOND_TYPE)
    vec[BOND_TYPE[str(bond.GetBondType())]] = 1

    # in ring
    new_vec = [0] * 2
    new_vec[bond.IsInRing()] = 1
    vec += new_vec

    return np.array(vec, dtype=DTYPE_FLOAT)


def mol_to_features(mol_rd):
    mol_rd = Chem.RemoveHs(mol_rd)
    feats = []
    for x in mol_rd.GetAtoms():
        atom_to_vector(x)


def cbeta_atom(residue):
    if residue is None:
        return None
    name = 'CA' if residue.getResname() == 'GLY' else 'CB'
    cbeta = residue.select('name ' + name)
    if cbeta is not None:
        return cbeta[0]
    return None


def calc_sidechain_torsions(residue):
    pass


def calc_bb_torsions(a, b, c):
    pass


def ag_to_features(rec_ag, ag_aln, tar_aln):
    feats_1d = OrderedDict(
        aatype=[],
        crd_mask=[],
        crd_beta_mask=[],
        crd_beta=[],
        torsions=[],
        torsions_alt=[],
        torsions_mask=[],
        resi=[],
    )

    residues = list(rec_ag.getHierView().iterResidues())
    assert len(residues) == len(ag_aln.replace('-', ''))

    ag_resi = 0
    tar_resi = 0
    for a, b in zip(ag_aln, tar_aln):
        if b != '-':
            feats_1d['aatype'].append(a)
            residue = None
            if a != '-':
                residue = residues[ag_resi]
            feats_1d['crd_mask'].append(residue is not None)
            cbeta = cbeta_atom(residue)
            feats_1d['crd_beta_mask'].append(cbeta is not None)
            feats_1d['crd_beta'].append(None if cbeta is None else cbeta.getCoords())
            feats_1d['resi'].append(tar_resi)
            tar_resi += 1

        if a != '-':
            ag_resi += 1

    return feats_1d


def rec_to_features(case_dict):
    # aatype, af_confidence, crd_mask, crd_beta_mask, torsions, torsions_alt, torsions_mask, res_index, sasa
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    af_seq = case_dict['alphafold']['seq']
    entity_seq = case_dict['entity_info']['pdbx_seq_one_letter_code_can']
    #assert af_seq == entity_seq

    rec_ag = prody.parsePDB(case_dir / 'AF_orig.pdb')
    #print(utils.global_align(af_seq, entity_seq))
    af_aln, ent_aln = utils.global_align(af_seq, entity_seq)[0][:2]
    return ag_to_features(rec_ag, af_aln, ent_aln)


def transform_hh_aln(tpl_seq, tar_seq, hhpred_dict):
    tpl_begin, tpl_end = hhpred_dict['target_range']
    tar_begin, tar_end = hhpred_dict['query_range']

    # make sure that sequence in hhpred db is the same as in pdb entity
    #print(hhpred_dict['query_aln'].replace('-', ''))
    #print(tar_seq[tar_begin:tar_end])
    assert hhpred_dict['query_aln'].replace('-', '') == tar_seq[tar_begin:tar_end]
    assert hhpred_dict['target_aln'].replace('-', '') == tpl_seq[tpl_begin:tpl_end]

    tpl_aln = tpl_seq[:tpl_begin] + '-'*len(tar_seq[:tar_begin]) + hhpred_dict['target_aln'] + '-'*len(tar_seq[tar_end:]) + tpl_seq[tpl_end:]
    tar_aln = '-'*len(tpl_seq[:tpl_begin]) + tar_seq[:tar_begin] + hhpred_dict['query_aln'] + tar_seq[tar_end:] + '-'*len(tpl_seq[tpl_end:])

    # sanity check
    assert len(tpl_aln) == len(tar_aln)
    return tpl_aln, tar_aln


def repatch_tpl_ag_to_ent_aln(ag_aln, ent_aln, hh_tpl_aln, hh_tar_aln):
    new_ag_aln = []
    new_hh_tar_aln = []
    ent_pos = 0
    for hh_pos, sym in enumerate(hh_tpl_aln):
        if sym != '-':
            new_ag_aln.append(ag_aln[ent_pos])
            new_hh_tar_aln.append(hh_tar_aln[hh_pos])
            ent_pos += 1
            while ent_pos < len(ent_aln) and ent_aln[ent_pos] == '-':
                new_ag_aln.append(ag_aln[ent_pos])
                new_hh_tar_aln.append('-')
                ent_pos += 1
        else:
            new_ag_aln.append(sym)
            new_hh_tar_aln.append(hh_tar_aln[hh_pos])

    return ''.join(new_ag_aln), ''.join(new_hh_tar_aln)


def hhpred_template_rec_to_features(tpl_dict, tar_dict, hhpred_dict):
    tpl_dir = DATA_DIR / 'cases' / tpl_dict['case_name']
    tpl_ag = prody.parsePDB(tpl_dir / 'rec_orig.pdb')

    tpl_ent_seq = tpl_dict['entity_info']['pdbx_seq_one_letter_code_can']
    tar_ent_seq = tar_dict['entity_info']['pdbx_seq_one_letter_code_can']

    tpl_hh_aln, tar_hh_aln = transform_hh_aln(tpl_ent_seq, tar_ent_seq, hhpred_dict)
    ent_aln = tpl_dict['entity_info']['entity_aln']
    pdb_aln = tpl_dict['entity_info']['pdb_aln']

    new_pdb_aln, new_tar_aln = repatch_tpl_ag_to_ent_aln(pdb_aln, ent_aln, tpl_hh_aln, tar_hh_aln)

    # sanity check
    assert tar_ent_seq == new_tar_aln.replace('-', '')
    return ag_to_features(tpl_ag, new_pdb_aln, new_tar_aln)


def dmat_to_distogram(dmat, dmin, dmax, num_bins, mask=None):
    shape = dmat.shape
    dmat = dmat.copy().flatten()
    dgram = np.zeros((len(dmat), num_bins))
    bin_size = (dmax - dmin) / num_bins
    dmat -= dmin
    bin_ids = (dmat // bin_size).astype(int)
    bin_ids[bin_ids < 0] = 0
    bin_ids[bin_ids >= num_bins] = num_bins - 1
    dgram[np.arange(len(dgram)), bin_ids] = 1.0
    dgram = dgram.reshape(*shape, num_bins)

    if mask is not None:
        mx, my = np.where(mask)
        dgram[mx, my] = 0

    return dgram


def rec_literal_to_numeric(rec_dict_literal):
    rec_dict = rec_dict_literal

    aatype = rec_dict['aatype']
    rec_dict['aatype'] = [np.zeros(len(AATYPE), dtype=DTYPE_FLOAT) for _ in aatype]
    for i, vec in enumerate(rec_dict['aatype']):
        vec[AATYPE.index(aatype[i])] = 1

    crd = np.stack([np.zeros(3) if x is None else x for x in rec_dict['crd_beta']])
    dmat = utils.calc_dmat(crd, crd)
    dgram = dmat_to_distogram(dmat, REC_DISTOGRAM['min'], REC_DISTOGRAM['max'], REC_DISTOGRAM['num_bins'])

    # cbeta_mask 2d
    crd_beta_mask = np.array(rec_dict['crd_beta_mask'])
    crd_beta_mask_2d = np.outer(crd_beta_mask, crd_beta_mask)

    # residue id 2d
    resi_1d = np.array(rec_dict['resi'])
    resi_2d = resi_1d[None, :] - resi_1d[:, None]
    resi_2d = dmat_to_distogram(resi_2d, -RESIGRAM_MAX, RESIGRAM_MAX + 1, RESIGRAM_MAX * 2 + 1)

    # mask distogram
    nob_x, nob_y = np.where(crd_beta_mask_2d)
    dgram[nob_x, nob_y] = 0.0

    return {
        'aatype': np.stack(rec_dict['aatype']),
        'crd_mask': np.array(rec_dict['crd_mask']),
        'crd_beta_mask': np.array(rec_dict['crd_beta_mask']),
        'crd_beta': crd,
        'resi_1d': resi_1d,
        'resi_2d': resi_2d,
        'distogram_2d': dgram,
        'crd_beta_mask_2d': crd_beta_mask_2d.astype(DTYPE_FLOAT)
    }


def target_rec_featurize(case_dict):
    rec_feats = rec_literal_to_numeric(rec_to_features(case_dict))
    rec_1d = np.concatenate([rec_feats['aatype'], rec_feats['crd_mask'][..., None], rec_feats['crd_beta_mask'][..., None]], axis=-1)
    rec_2d = np.concatenate([rec_feats['distogram_2d'], rec_feats['crd_beta_mask_2d'][..., None]], axis=-1)
    return {
        'rec_1d': rec_1d,
        'rec_2d': rec_2d,
        'relpos_2d': rec_feats['resi_2d']
    }


def ligand_featurize(mol, mol_3d=None):
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_to_vector(atom))
    bond_feats = []

    for bond in mol.GetBonds():
        vec = bond_to_vector(bond)
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feats.append([i, j, vec])
        bond_feats.append([j, i, vec])
    bonds_2d = np.zeros((len(atom_feats), len(atom_feats), 6))
    bonds_2d[[x[0] for x in bond_feats], [x[1] for x in bond_feats]] = [x[2] for x in bond_feats]

    out = OrderedDict(
        atom_feats=np.stack(atom_feats),
        bonds_2d=bonds_2d,
        coords=None,
        matches=None
    )

    if mol_3d is not None:
        matches = mol.GetSubstructMatches(mol_3d, uniquify=False)
        assert len(matches) > 0
        out['matches'] = matches
        out['coords'] = mol_3d.GetConformer(0).GetPositions()

    return out


def target_group_featurize(case_dict, group_dict):
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    group_dir = case_dir / group_dict['name']
    group_feats = []
    for lig_dict in group_dict['ligands']:
        mol = Chem.MolFromSmiles(lig_dict['smiles'])
        #mol_3d = Chem.MolFromMolFile(group_dir / lig_dict['sdf_id'] + '.mol', removeHs=True)
        group_feats.append(ligand_featurize(mol))

    group_1d = np.concatenate([x['atom_feats'] for x in group_feats], axis=0)
    group_2d = np.zeros(group_1d.shape[0], group_1d.shape[0], group_feats[0]['bonds_2d'])
    start = 0
    for lig_feats in group_feats:
        lig_size = lig_feats['atom_feats'].shape[0]
        group_2d[start:start+lig_size, start:start+lig_size] = lig_feats['bond_2d']
        start += lig_size

    return {
        'lig_1d': group_1d,
        'lig_2d': group_2d
    }


def hhpred_template_lig_to_features(mol_tpl: Chem.Mol, mol_tar: Chem.Mol, m_tpl, m_tar):
    num_atoms_tar = mol_tar.GetNumAtoms()
    feats = ligand_featurize(mol_tpl, mol_tpl)

    dilated_atom_feats = np.zeros((num_atoms_tar, feats['atom_feats'].shape[-1] + 1))
    dilated_atom_feats[:, -1] = 1
    dilated_atom_feats[list(m_tar), :-1] = feats['atom_feats'][list(m_tpl)]
    dilated_atom_feats[list(m_tar), -1] = 0

    # ideally we need to do smiles matching and then record here
    # which atoms don't have coords,
    # but thats too much work for now
    atom_present = np.zeros(num_atoms_tar)
    atom_present[list(m_tar)] = 1
    atom_present_2d = np.outer(atom_present, atom_present)

    dilated_crd = np.zeros((num_atoms_tar, 3))
    dilated_crd[list(m_tar)] = feats['coords'][list(m_tpl)]

    dmat = utils.calc_dmat(dilated_crd, dilated_crd)
    dgram = dmat_to_distogram(dmat, LIG_DISTOGRAM['min'], LIG_DISTOGRAM['max'], LIG_DISTOGRAM['num_bins'], mask=atom_present_2d < 1)

    dilated_bonds = np.zeros((num_atoms_tar, num_atoms_tar, feats['bonds_2d'].shape[-1]))
    tar_ids = list(zip(itertools.product(m_tar, m_tar)))
    tpl_ids = list(zip(itertools.product(m_tpl, m_tpl)))
    dilated_bonds[tar_ids[0], tar_ids[1]] = feats['bonds_2d'][tpl_ids[0], tpl_ids[1]]

    return {
        'atom_feats_1d': dilated_atom_feats,
        'atom_present_1d': atom_present,
        'atom_coords_1d': dilated_crd,
        'bond_feats_2d': dilated_bonds,
        'distogram_2d': dgram,
        'atom_present_2d': atom_present_2d,
        'match_tar': m_tar,
        'match_tpl': m_tpl
    }


def hh_template_featurize(tar_mol: Chem.Mol, tar_match, tar_case_dict, tar_group_dict, tar_ligand_id, temp_mol: Chem.Mol, temp_match, temp_dict, temp_case_dict):
    rec_literal = hhpred_template_rec_to_features(temp_case_dict, tar_case_dict, temp_dict['hhpred'])
    rec_feats = rec_literal_to_numeric(rec_literal)

    num_atoms_total = sum([x['num_heavy_atoms'] for x in tar_group_dict['ligands']])
    atom_begin = sum([x['num_heavy_atoms'] for x in tar_group_dict['ligands'][:tar_ligand_id]])
    atom_end = atom_begin + tar_group_dict['ligands'][tar_ligand_id]['num_heavy_atoms']
    lig_feats = hhpred_template_lig_to_features(temp_mol, tar_mol, temp_match, tar_match)

    #print(tar_group_dict)
    #print(tar_group_dict['ligands'][tar_ligand_id])
    #print(tar_mol.GetNumAtoms())
    #print(tar_mol.GetNumHeavyAtoms())

    rec_1d = np.concatenate([
        rec_feats['aatype'],
        rec_feats['crd_mask'][..., None],
        rec_feats['crd_beta_mask'][..., None]
    ], axis=-1)

    extra = np.tile(rec_feats['aatype'], (rec_feats['aatype'].shape[0], 1, 1))
    rr_2d = np.concatenate([
        rec_feats['distogram_2d'],
        rec_feats['crd_beta_mask_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=2)

    lig_1d = np.concatenate([lig_feats['atom_feats_1d'], lig_feats['atom_present_1d'][..., None]], axis=-1)

    extra = np.tile(lig_feats['atom_feats_1d'], (lig_feats['atom_feats_1d'].shape[0], 1, 1))
    ll_2d_local = np.concatenate([
        lig_feats['distogram_2d'],
        lig_feats['bond_feats_2d'],
        lig_feats['atom_present_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=-1)
    ll_2d = np.zeros((num_atoms_total, num_atoms_total, ll_2d_local.shape[-1]))
    ll_2d[atom_begin:atom_end, atom_begin:atom_end, :] = ll_2d_local

    rl_dmat = utils.calc_dmat(rec_feats['crd_beta'], lig_feats['atom_coords_1d'])
    rl_present_2d = np.outer(rec_feats['crd_beta_mask'], lig_feats['atom_present_1d'])
    rl_dgram = dmat_to_distogram(rl_dmat, REC_LIG_DISTOGRAM['min'], REC_LIG_DISTOGRAM['max'], REC_LIG_DISTOGRAM['num_bins'], mask=rl_present_2d < 1)
    rl_2d_local = np.concatenate([
        rl_dgram,
        rl_present_2d[..., None],
        np.tile(rec_feats['aatype'], (lig_feats['atom_feats_1d'].shape[0], 1, 1)).transpose([1, 0, 2]),
        np.tile(lig_feats['atom_feats_1d'], (rec_feats['aatype'].shape[0], 1, 1)),
    ], axis=-1)
    rl_2d = np.zeros((rl_2d_local.shape[0], num_atoms_total, rl_2d_local.shape[-1]))
    rl_2d[:, atom_begin:atom_end] = rl_2d_local

    lr_2d = rl_2d.transpose([1, 0, 2])

    out = {
        'lig_1d': lig_1d,
        'rec_1d': rec_1d,
        'll_2d': ll_2d,
        'rr_2d': rr_2d,
        'rl_2d': rl_2d,
        'lr_2d': lr_2d
    }
    return out


def match_mols(mol1, mol2, flags=['r']):
    smarts_str = utils.calc_mcs(mol1, mol2, mcs_flags=flags)[0]
    assert len(smarts_str) > 0
    smarts = Chem.MolFromSmarts(smarts_str)
    matches_tar = mol1.GetSubstructMatches(smarts, uniquify=True)
    matches_tpl = mol2.GetSubstructMatches(smarts, uniquify=True)
    return smarts_str, matches_tar, matches_tpl


def hh_templates_featurize_many(tar_case_dict, case_dicts):
    templates = []

    case_dir = DATA_DIR / 'cases' / tar_case_dict['case_name']
    for tar_group_dict in tar_case_dict['ligand_groups']:
        tar_group_dir = case_dir / tar_group_dict['name']
        for tar_ligand_id, tar_ligand_dict in enumerate(tar_group_dict['ligands']):
            temp_json = tar_group_dir / tar_ligand_dict['sdf_id'] + '.templates.json'
            if not temp_json.exists():
                continue
            tar_mol = Chem.MolFromSmiles(tar_ligand_dict['smiles'])
            assert tar_mol.GetNumAtoms() == tar_mol.GetNumHeavyAtoms()
            temp_list = utils.read_json(temp_json)
            for temp_hh_dict in temp_list:
                temp_case_dict = case_dicts[temp_hh_dict['hhpred']['hh_pdb']]
                temp_chemid = temp_hh_dict['lig_match']['ref_chemid']
                for temp_group_dict in temp_case_dict['ligand_groups']:
                    for temp_ligand_dict in temp_group_dict['ligands']:
                        if temp_chemid != temp_ligand_dict['chemid']:
                            continue
                        temp_mol = Chem.MolFromMolFile(DATA_DIR / 'cases' / temp_case_dict['case_name'] / temp_group_dict['name'] / temp_ligand_dict['sdf_id'] + '.mol', removeHs=True)
                        assert tar_mol.GetNumAtoms() == tar_mol.GetNumHeavyAtoms()
                        smarts, tar_matches, temp_matches = match_mols(tar_mol, temp_mol)
                        for tar_match in tar_matches:
                            for temp_match in temp_matches:
                                temp_feats = hh_template_featurize(tar_mol, tar_match, tar_case_dict, tar_group_dict, tar_ligand_id, temp_mol, temp_match, temp_hh_dict, temp_case_dict)
                                temp_feats['tar_match'] = tar_match
                                temp_feats['temp_match'] = temp_match
                                temp_feats['smarts'] = smarts
                                #temp_feats['smarts'] =
                                #temp_feats['smarts'] =
                                templates.append(temp_feats)
    return templates


def fragment_template_group_featurize(case_dict, group_dict):
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    group_dir = case_dir / group_dict['name']
    group_feats = []
    for lig_dict in group_dict['ligands']:
        mol = Chem.MolFromSmiles(lig_dict['smiles'])
        mol_3d = Chem.MolFromMolFile(group_dir / lig_dict['sdf_id'] + '.mol', removeHs=True)
        group_feats.append(ligand_featurize(mol, mol_3d))

    atom_feats_1d = np.concatenate([x['atom_feats'] for x in group_feats], axis=0)

    # combine matches
    matches = []
    for mgr in itertools.product(*[x['matches'] for x in group_feats]):
        match = []
        for lig_id, m in enumerate(mgr):
            prev_lig_size = 0 if lig_id == 0 else group_feats[lig_id-1]['atom_feats'].shape[0]
            match += [prev_lig_size + ai for ai in m]
        matches.append(match)

    # dilated coords (with missing atoms set to zero)
    coords = np.concatenate([x['coords'] for x in group_feats], axis=0)
    atom_coords_1d = np.zeros((atom_feats_1d.shape[0], 3))
    atom_coords_1d[matches[0]] = coords

    atom_present_1d = np.array([x in matches[0] for x in range(atom_feats_1d.shape[0])], dtype=float)
    atom_present_2d = np.outer(atom_present_1d, atom_present_1d)

    # fill bonds 2d
    bond_feats_2d = np.zeros((atom_feats_1d.shape[0], atom_feats_1d.shape[0], group_feats[0]['bonds_2d'].shape[-1]))
    start = 0
    for lig_feats in group_feats:
        lig_size = lig_feats['atom_feats'].shape[0]
        bond_feats_2d[start:start+lig_size, start:start+lig_size] = lig_feats['bonds_2d']
        start += lig_size

    # fill distogram
    dmat = utils.calc_dmat(atom_coords_1d, atom_coords_1d)
    dgram = dmat_to_distogram(dmat, LIG_DISTOGRAM['min'], LIG_DISTOGRAM['max'], LIG_DISTOGRAM['num_bins'], mask=atom_present_2d < 1)

    return {
        'atom_feats_1d': atom_feats_1d,
        'atom_coords_1d': atom_coords_1d,
        'atom_present_1d': atom_present_1d,
        'atom_present_2d': atom_present_2d,
        'bond_feats_2d': bond_feats_2d,
        'distogram_2d': dgram,
        'matches_smiles_to_3d': matches
    }


def fragment_template_featurize(temp_case_dict, group_dict):
    rec_literal = rec_to_features(temp_case_dict)
    rec_feats = rec_literal_to_numeric(rec_literal)

    rec_1d = np.concatenate([
        rec_feats['aatype'],
        rec_feats['crd_mask'][..., None],
        rec_feats['crd_beta_mask'][..., None]
    ], axis=-1)

    extra = np.tile(rec_feats['aatype'], (rec_feats['aatype'].shape[0], 1, 1))
    rr_2d = np.concatenate([
        rec_feats['distogram_2d'],
        rec_feats['crd_beta_mask_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=2)

    lig_feats = fragment_template_group_featurize(temp_case_dict, group_dict)
    lig_1d = np.concatenate([lig_feats['atom_feats_1d'], lig_feats['atom_present_1d'][..., None]], axis=-1)
    print(lig_feats['atom_present_1d'])

    extra = np.tile(lig_feats['atom_feats_1d'], (lig_feats['atom_feats_1d'].shape[0], 1, 1))
    ll_2d = np.concatenate([
        lig_feats['distogram_2d'],
        lig_feats['bond_feats_2d'],
        lig_feats['atom_present_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=-1)

    rl_dmat = utils.calc_dmat(rec_feats['crd_beta'], lig_feats['atom_coords_1d'])
    rl_present_2d = np.outer(rec_feats['crd_beta_mask'], lig_feats['atom_present_1d'])
    rl_dgram = dmat_to_distogram(rl_dmat, REC_LIG_DISTOGRAM['min'], REC_LIG_DISTOGRAM['max'], REC_LIG_DISTOGRAM['num_bins'], mask=rl_present_2d < 1)
    rl_2d = np.concatenate([
        rl_dgram,
        rl_present_2d[..., None],
        np.tile(rec_feats['aatype'], (lig_feats['atom_feats_1d'].shape[0], 1, 1)).transpose([1, 0, 2]),
        np.tile(lig_feats['atom_feats_1d'], (rec_feats['aatype'].shape[0], 1, 1)),
    ], axis=-1)

    lr_2d = rl_2d.transpose([1, 0, 2])

    out = {
        'lig_1d': lig_1d,
        'rec_1d': rec_1d,
        'll_2d': ll_2d,
        'rr_2d': rr_2d,
        'rl_2d': rl_2d,
        'lr_2d': lr_2d
    }
    return out


def example():
    from tqdm import tqdm
    print(DATA_DIR)
    #print(rec_featurize(utils.read_json(DATA_DIR / 'cases/10GS_A/case.json')))
    tar_case_dict = utils.read_json(DATA_DIR / 'cases/3HCF_A/case.json')
    tar_group_dict = tar_case_dict['ligand_groups'][1]
    tar_ligand_id = 1
    template_dict = utils.read_json(DATA_DIR / 'cases/3HCF_A/LT5_SAH/3hcf_SAH_1_A_3001__D___.templates.json')[0]
    template_case_dict = utils.read_json(DATA_DIR / 'cases/4EKG_A/case.json')
    template_mol_file = DATA_DIR / 'cases/4EKG_A/0QJ/4ekg_0QJ_1_A_500__B___.mol'
    #print(hh_template_featurize(tar_case_dict, tar_group_dict, tar_ligand_id, template_dict, template_case_dict, template_mol_file))

    cases = OrderedDict((x.dirname().basename(), utils.read_json(x)) for x in tqdm(sorted(DATA_DIR.glob('cases/*/case.json'))))
    for k, v in tqdm(list(cases.items())[115:]):
        print(len(hh_templates_featurize_many(v, cases)))


def example2():
    case_dict = utils.read_json(DATA_DIR / 'cases/3HCF_A/case.json')
    group_dict = case_dict['ligand_groups'][1]
    print(fragment_template_featurize(case_dict, group_dict))


if __name__ == '__main__':
    example2()
