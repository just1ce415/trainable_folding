import itertools
import numpy as np
import prody
from collections import OrderedDict, defaultdict
import typing
import torch
import time
from tqdm import tqdm
import traceback

from alphadock import utils
from alphadock import residue_constants
from alphadock import all_atom
from alphadock import r3
from alphadock import quat_affine
from alphadock.config import DATA_DIR, DTYPE_FLOAT, DTYPE_INT


ELEMENTS_ORDER = residue_constants.ELEMENTS_ORDER
ELEMENTS = residue_constants.ELEMENTS

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

AATYPE = residue_constants.restype_order
AATYPE_WITH_X = residue_constants.restype_order_with_x
AATYPE_WITH_X_AND_GAP = AATYPE_WITH_X.copy()
AATYPE_WITH_X_AND_GAP['-'] = len(AATYPE_WITH_X)

REC_DISTOGRAM = {
    'min': 3.25,
    'max': 51,
    'num_bins': 39
}

LIG_DISTOGRAM = {
    'min': 0.5,
    'max': 20,
    'num_bins': 39
}

REC_LIG_DISTOGRAM = {
    'min': 3.25,
    'max': 50.75,
    'num_bins': 39
}

RESIGRAM_MAX = 32

FRAGMENT_TEMPLATE_RADIUS = 10

LIG_EXTRA_DISTANCE = {
    'min': 3,
    'max': 12,
    'num_bins': 9
}

AFFINITY_LOG10 = {
    'min': 0,
    'max': 6,
    'num_bins': 6
}

SASA_BINS = {
    'min': 0,
    'max': 2.5,
    'num_bins': 10
}

AF_CONFIDENCE_BINS = {
    'min': 40,
    'max': 100,
    'num_bins': 6
}


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


def residue_to_atom14(residue: prody.Residue):
    ideal_names = residue_constants.restype_name_to_atom14_names.get(residue.getResname().upper() if residue is not None else 'UNK', 'UNK')
    coords = np.zeros((len(ideal_names), 3), dtype=DTYPE_FLOAT)
    has_coords = np.zeros(len(ideal_names), dtype=DTYPE_FLOAT)

    if residue is not None:
        residue = residue.heavy.copy()
        atom2crd = {a.getName(): a.getCoords() for a in residue}
        for i, a in enumerate(ideal_names):
            coords[i] = atom2crd.get(a, np.zeros(3))
            has_coords[i] = a in atom2crd
        res_names = [x.getName() for x in residue if x.getName() not in ['OXT']]
        assert set(res_names).issubset(set(ideal_names)), (ideal_names, res_names)

    return coords, has_coords


def ag_to_features(rec_ag, ag_aln, tar_aln, no_mismatch=False, residues_mask=None):
    # TODO: add sasa, torsions and AF confidence

    feats_1d = OrderedDict(
        seq_aatype=[],
        ag_aatype=[],
        crd_mask=[],
        crd_beta_mask=[],
        crd_beta=[],
        torsions=[],
        torsions_alt=[],
        torsions_mask=[],
        resi=[],
        atom14_coords=[],
        atom14_has_coords=[],
        has_frame=[],
        ag_resi=[]
    )

    residues = list(rec_ag.getHierView().iterResidues())
    assert len(residues) == len(ag_aln.replace('-', ''))

    #print(ag_aln)
    #print(tar_aln)

    ag_resi = 0
    tar_resi = 0
    for a, b in zip(ag_aln, tar_aln):
        assert a in AATYPE_WITH_X_AND_GAP
        assert b in AATYPE_WITH_X_AND_GAP
        if b != '-':
            if residues_mask is None or (residues_mask is not None and a != '-' and residues_mask[ag_resi] > 0.0):
                feats_1d['seq_aatype'].append(b)
                feats_1d['ag_aatype'].append(a)
                residue = None
                if a != '-':
                    if no_mismatch and a != b:
                        # TODO: decide what to do with mismatching residues when
                        #       the residues in the structure to do correspond to
                        #       the entry sequence. Treating them as missing for now
                        #print('Mismatch:', a, b)
                        residue = None
                    else:
                        residue = residues[ag_resi]

                try:
                    atom14_coords, atom14_has_coords = residue_to_atom14(residue)
                except AssertionError:
                    # 1BNN_A throws AssertionError
                    traceback.print_exc()
                    atom14_coords = np.zeros((14, 3), dtype=DTYPE_FLOAT)
                    atom14_has_coords = np.zeros(14, dtype=DTYPE_FLOAT)
                    residue = None

                feats_1d['crd_mask'].append(residue is not None)
                cbeta = cbeta_atom(residue)
                feats_1d['crd_beta_mask'].append(cbeta is not None)
                feats_1d['crd_beta'].append(None if cbeta is None else cbeta.getCoords())
                feats_1d['has_frame'].append((residue is not None) and ('CA' in residue) and ('C' in residue) and ('N' in residue))
                feats_1d['resi'].append(tar_resi)
                feats_1d['ag_resi'].append(min(ag_resi, len(residues)-1))

                feats_1d['atom14_coords'].append(atom14_coords)
                feats_1d['atom14_has_coords'].append(atom14_has_coords)
                #feats_1d['atom14_coords'].append(np.zeros((14, 3)))
                #feats_1d['atom14_has_coords'].append(np.zeros(14))

            tar_resi += 1

        if a != '-':
            ag_resi += 1

    return feats_1d


def rec_to_features(case_dict):
    # aatype, af_confidence, crd_mask, crd_beta_mask, torsions, torsions_alt, torsions_mask, res_index, sasa
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    af_seq = case_dict['alphafold']['seq']
    entity_seq = case_dict['entity_info']['pdbx_seq_one_letter_code_can']

    rec_ag = prody.parsePDB(case_dir / 'AF_orig.pdb')
    af_aln, ent_aln = utils.global_align(af_seq, entity_seq)[0][:2]
    #print(case_dict['case_name'] + '\n' + af_aln + '\n' + ent_aln + '\n' + 'mm_num: ' + str(sum([x != y and x != '-' and y != '-' for x, y in zip(af_aln, ent_aln)])))

    feats = ag_to_features(rec_ag, af_aln, ent_aln, no_mismatch=True)

    sasa = np.loadtxt(case_dir / 'AF_sasa.txt')
    assert len(sasa) == len(rec_ag)
    sasa_residues = [sasa[r.getIndices()].sum() for r in rec_ag.getHierView().iterResidues()]
    feats['sasa'] = np.array(sasa_residues)[feats['ag_resi']]
    feats['confidence'] = rec_ag.calpha.getBetas()[feats['ag_resi']]

    return feats


def _transform_hh_aln(tpl_seq, tar_seq, hhpred_dict):
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


def _repatch_tpl_ag_to_ent_aln(ag_aln, ent_aln, hh_tpl_aln, hh_tar_aln):
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

    tpl_hh_aln, tar_hh_aln = _transform_hh_aln(tpl_ent_seq, tar_ent_seq, hhpred_dict)
    ent_aln = tpl_dict['entity_info']['entity_aln']
    pdb_aln = tpl_dict['entity_info']['pdb_aln']

    new_pdb_aln, new_tar_aln = _repatch_tpl_ag_to_ent_aln(pdb_aln, ent_aln, tpl_hh_aln, tar_hh_aln)

    # sanity check
    assert tar_ent_seq == new_tar_aln.replace('-', '')
    return ag_to_features(tpl_ag, new_pdb_aln, new_tar_aln, no_mismatch=False)


def dmat_to_distogram(dmat, dmin, dmax, num_bins, mask=None):
    shape = dmat.shape
    dmat = dmat.copy().flatten()
    dgram = np.zeros((len(dmat), num_bins), dtype=DTYPE_FLOAT)
    bin_size = (dmax - dmin) / num_bins
    bin_ids = np.minimum(np.maximum(((dmat - dmin) // bin_size).astype(int), 0), num_bins - 1)
    dgram[np.arange(dgram.shape[0]), bin_ids] = 1.0
    dgram = dgram.reshape(*shape, num_bins)

    if mask is not None:
        mx, my = np.where(mask)
        dgram[mx, my] = 0

    return dgram


def one_hot_aatype(aa, alphabet):
    out = np.zeros(len(alphabet), dtype=DTYPE_FLOAT)
    out[alphabet[aa]] = 1
    return out


def rec_literal_to_numeric(rec_dict, seq_include_gap=False):
    seq_abc = AATYPE_WITH_X_AND_GAP if seq_include_gap else AATYPE_WITH_X
    seq_aatype_num = [seq_abc[x] for x in rec_dict['seq_aatype']]
    ag_aatype_num = [AATYPE_WITH_X_AND_GAP[x] for x in rec_dict['ag_aatype']]

    rec_dict['seq_aatype'] = [one_hot_aatype(x, seq_abc) for x in rec_dict['seq_aatype']]
    rec_dict['ag_aatype'] = [one_hot_aatype(x, AATYPE_WITH_X_AND_GAP) for x in rec_dict['ag_aatype']]
    crd_beta_mask = np.array(rec_dict['crd_beta_mask'])

    if 'sasa' in rec_dict:
        sasa = dmat_to_distogram(rec_dict['sasa'], SASA_BINS['min'], SASA_BINS['max'], SASA_BINS['num_bins']) * crd_beta_mask[:, None]

    if 'confidence' in rec_dict:
        confidence = dmat_to_distogram(rec_dict['confidence'], AF_CONFIDENCE_BINS['min'], AF_CONFIDENCE_BINS['max'], AF_CONFIDENCE_BINS['num_bins']) * crd_beta_mask[:, None]

    crd = np.stack([np.zeros(3) if x is None else x for x in rec_dict['crd_beta']])
    dmat = utils.calc_dmat(crd, crd)
    dgram = dmat_to_distogram(dmat, REC_DISTOGRAM['min'], REC_DISTOGRAM['max'], REC_DISTOGRAM['num_bins'])

    # cbeta_mask 2d
    crd_beta_mask_2d = np.outer(crd_beta_mask, crd_beta_mask)

    # residue id 2d
    resi_1d = np.array(rec_dict['resi'])
    resi_2d = resi_1d[None, :] - resi_1d[:, None]
    resi_2d = dmat_to_distogram(resi_2d, -RESIGRAM_MAX, RESIGRAM_MAX + 1, RESIGRAM_MAX * 2 + 1)

    # mask distogram
    dgram *= crd_beta_mask_2d[..., None]

    out_dict = {
        'seq_aatype': np.stack(rec_dict['seq_aatype']),
        'seq_aatype_num': np.array(seq_aatype_num, dtype=DTYPE_INT),
        'ag_aatype': np.stack(rec_dict['ag_aatype']),
        'ag_aatype_num': np.array(ag_aatype_num, dtype=DTYPE_INT),
        'crd_mask': np.array(rec_dict['crd_mask']),
        'crd_beta_mask': np.array(rec_dict['crd_beta_mask']),
        'crd_beta': crd,
        'resi_1d': resi_1d,
        'resi_2d': resi_2d,
        'distogram_2d': dgram,
        'dmat': dmat,
        'crd_beta_mask_2d': crd_beta_mask_2d.astype(DTYPE_FLOAT),
        'atom14_coords': np.stack(rec_dict['atom14_coords']),
        'atom14_has_coords': np.stack(rec_dict['atom14_has_coords']),
        'has_frame': np.array(rec_dict['has_frame'], dtype=DTYPE_INT),
    }
    if 'sasa' in rec_dict:
        out_dict['sasa'] = sasa

    if 'confidence' in rec_dict:
        out_dict['confidence'] = confidence

    return out_dict


def target_rec_featurize(case_dict, rot_mat=None):
    rec_feats = rec_literal_to_numeric(rec_to_features(case_dict), seq_include_gap=False)
    rec_1d = np.concatenate([rec_feats['seq_aatype'], rec_feats['crd_mask'][..., None], rec_feats['crd_beta_mask'][..., None], rec_feats['sasa'], rec_feats['confidence']], axis=-1)
    rec_2d = np.concatenate([rec_feats['distogram_2d'], rec_feats['crd_beta_mask_2d'][..., None]], axis=-1)

    renaming_mats = all_atom.RENAMING_MATRICES[rec_feats['seq_aatype_num']]  # (N, 14, 14)
    atom14_atom_is_ambiguous = (renaming_mats * np.eye(14)[None]).sum(1) == 0

    # apply random rotation
    if rot_mat is not None:
        rec_feats['atom14_coords'] = np.dot(rec_feats['atom14_coords'], rot_mat.T)

    # calculate atom37 representations and backbone frames
    rec_atom37_coords = all_atom.atom14_to_atom37(torch.from_numpy(rec_feats['atom14_coords']).float(), torch.from_numpy(rec_feats['seq_aatype_num']))
    rec_atom37_mask = all_atom.atom14_to_atom37(torch.from_numpy(rec_feats['atom14_has_coords']).float(), torch.from_numpy(rec_feats['seq_aatype_num']))
    rec_all_frames = all_atom.atom37_to_frames(torch.from_numpy(rec_feats['seq_aatype_num']), rec_atom37_coords.float(), rec_atom37_mask.float())
    # TODO: decide about placeholder
    rec_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(rec_feats['seq_aatype_num'][None]), rec_atom37_coords[None].float(), rec_atom37_mask[None].float(), placeholder_for_undefined=True)

    rec_bb_affine = r3.rigids_to_quataffine(r3.rigids_from_tensor_flat12(rec_all_frames['rigidgroups_gt_frames'][..., 0, :]))
    rec_bb_affine.quaternion = quat_affine.rot_to_quat(rec_bb_affine.rotation)
    rec_bb_affine = rec_bb_affine.to_tensor().numpy()
    rec_bb_affine_mask = rec_all_frames['rigidgroups_gt_exists'][..., 0].numpy()

    out = {
        'rec_1d': rec_1d.astype(DTYPE_FLOAT),
        'rec_2d': rec_2d.astype(DTYPE_FLOAT),
        'rec_relpos': rec_feats['resi_2d'].astype(DTYPE_FLOAT),
        'rec_atom14_coords': rec_feats['atom14_coords'].astype(DTYPE_FLOAT),
        'rec_atom14_has_coords': rec_feats['atom14_has_coords'].astype(DTYPE_FLOAT),
        'rec_atom37_coords': rec_atom37_coords.numpy().astype(DTYPE_FLOAT),
        'rec_atom37_has_coords': rec_atom37_mask.numpy().astype(DTYPE_FLOAT),

        'rec_aatype': rec_feats['seq_aatype_num'].astype(DTYPE_INT),
        'rec_index': rec_feats['resi_1d'].astype(DTYPE_INT),
        'rec_bb_affine': rec_bb_affine.astype(DTYPE_FLOAT),
        'rec_bb_affine_mask': rec_bb_affine_mask.astype(DTYPE_FLOAT),
        'rec_atom14_atom_is_ambiguous': atom14_atom_is_ambiguous.astype(DTYPE_FLOAT),
        'rec_atom14_atom_exists': residue_constants.restype_atom14_mask[rec_feats['seq_aatype_num']].astype(DTYPE_FLOAT),

        'rec_torsions_sin_cos': rec_torsions['torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'rec_torsions_sin_cos_alt': rec_torsions['alt_torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'rec_torsions_mask': rec_torsions['torsion_angles_mask'][0].numpy().astype(DTYPE_FLOAT),
    }
    out.update({k: v.numpy() for k, v in rec_all_frames.items()})

    return out


def ground_truth_featurize(case_dict, group_dict):
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    rec_ag = prody.parsePDB(case_dir / 'rec_orig.pdb')
    rec_dict = ag_to_features(
        rec_ag,
        case_dict['entity_info']['pdb_aln'],
        case_dict['entity_info']['entity_aln'],
        no_mismatch=True
    )

    rec_dict = rec_literal_to_numeric(rec_dict, seq_include_gap=False)

    #atom14_gt_positions_rigids = r3.Vecs(*[x.squeeze(-1) for x in np.split(rec_dict['rec_atom14_coords'], 3, axis=-1)])
    atom14_gt_positions = rec_dict['atom14_coords']  # (N, 14, 3)
    atom14_gt_exists = rec_dict['atom14_has_coords']  # (N, 14)
    renaming_mats = all_atom.RENAMING_MATRICES[rec_dict['seq_aatype_num']]  # (N, 14, 14)
    atom14_alt_gt_positions = np.sum(atom14_gt_positions[:, :, None, :] * renaming_mats[:, :, :, None], axis=1)
    atom14_alt_gt_exists = np.sum(rec_dict['atom14_has_coords'][:, :, None] * renaming_mats, axis=1)
    atom14_atom_is_ambiguous = (renaming_mats * np.eye(14)[None]).sum(1) == 0

    atom37_gt_positions = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_positions).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    atom37_gt_exists = all_atom.atom14_to_atom37(torch.from_numpy(atom14_gt_exists).float(), torch.from_numpy(rec_dict['seq_aatype_num']))
    gt_torsions = all_atom.atom37_to_torsion_angles(torch.from_numpy(rec_dict['seq_aatype_num'][None]), atom37_gt_positions[None].float(), atom37_gt_exists[None].float())

    rec_all_frames = all_atom.atom37_to_frames(torch.from_numpy(rec_dict['seq_aatype_num']), atom37_gt_positions.float(), atom37_gt_exists.float())
    rec_bb_affine = r3.rigids_to_quataffine(r3.rigids_from_tensor_flat12(rec_all_frames['rigidgroups_gt_frames'][..., 0, :]))
    rec_bb_affine.quaternion = quat_affine.rot_to_quat(rec_bb_affine.rotation)
    rec_bb_affine = rec_bb_affine.to_tensor().numpy()
    rec_bb_affine_mask = rec_all_frames['rigidgroups_gt_exists'][..., 0].numpy()

    # process ligand group
    group_dir = case_dir / group_dict['name']
    group_feats = []
    for lig_dict in group_dict['ligands']:
        lig_feats = np.load(DATA_DIR / 'featurized' / lig_dict['sdf_id'] + '.ligand_feats.npy', allow_pickle=True).item()
        group_feats.append(lig_feats)

    group_num_atoms = sum([x['atom_feats'].shape[0] for x in group_feats])
    lig_matches = []
    lig_coords = []
    lig_has_coords = []
    for match_combo in itertools.product(*[x['matches'] for x in group_feats]):
        group_match = []
        group_coords = np.zeros((group_num_atoms, 3))
        group_has_coords = np.zeros(group_num_atoms)

        lig_shift = 0
        for lig_id, lig_match in enumerate(match_combo):
            lig_match_shifted = [x + lig_shift for x in lig_match]
            group_match += lig_match_shifted
            group_coords[lig_match_shifted, :] = group_feats[lig_id]['coords']
            group_has_coords[group_match] = 1
            lig_shift += group_feats[lig_id]['atom_feats'].shape[0]

        lig_matches.append(group_match)
        lig_coords.append(group_coords)
        lig_has_coords.append(group_has_coords)

    #print(lig_matches)
    #print(group_dict['ligands'])
    aff_start = 0
    aff_end = 0
    aff_value = 0
    aff_label = 0
    aff_known = False
    for aff_lig_id, lig_dict in enumerate(group_dict['ligands']):
        aff_end += lig_dict['num_heavy_atoms']
        if lig_dict['affinity'] is not None and lig_dict['affinity']['unit'].upper() == 'NM':
            aff_value = np.log10(lig_dict['affinity']['value'])
            aff_bin_size = (AFFINITY_LOG10['max'] - AFFINITY_LOG10['min']) / AFFINITY_LOG10['num_bins']
            aff_label = np.minimum(np.abs(((aff_value - AFFINITY_LOG10['min']) // aff_bin_size).astype(int)), AFFINITY_LOG10['num_bins'] - 1)
            aff_known = True
            break
        aff_start += lig_dict['num_heavy_atoms']

    out = {
        'gt_aatype': rec_dict['seq_aatype_num'].astype(DTYPE_INT),  # same as for target
        'gt_atom14_coords': atom14_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords': atom14_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_coords_alt': atom14_alt_gt_positions.astype(DTYPE_FLOAT),  # (N_res, 14, 3)
        'gt_atom14_has_coords_alt': atom14_alt_gt_exists.astype(DTYPE_FLOAT),  # (N_res, 14)
        'gt_atom14_atom_is_ambiguous': atom14_atom_is_ambiguous.astype(DTYPE_FLOAT),

        'gt_torsions_sin_cos': gt_torsions['torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_sin_cos_alt': gt_torsions['alt_torsion_angles_sin_cos'][0].numpy().astype(DTYPE_FLOAT),
        'gt_torsions_mask': gt_torsions['torsion_angles_mask'][0].numpy().astype(DTYPE_FLOAT),

        'gt_rigidgroups_gt_frames': rec_all_frames['rigidgroups_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)
        'gt_rigidgroups_alt_gt_frames': rec_all_frames['rigidgroups_alt_gt_frames'].numpy().astype(DTYPE_FLOAT),  # (..., 8, 12)
        'gt_rigidgroups_gt_exists': rec_all_frames['rigidgroups_gt_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        #'gt_rigidgroups_group_exists': rec_all_frames['rigidgroups_group_exists'].numpy().astype(DTYPE_FLOAT),  # (..., 8)
        'gt_rigidgroups_group_is_ambiguous': rec_all_frames['rigidgroups_group_is_ambiguous'].numpy().astype(DTYPE_FLOAT),  # (..., 8)

        'gt_bb_affine': rec_bb_affine.astype(DTYPE_FLOAT),
        'gt_bb_affine_mask': rec_bb_affine_mask.astype(DTYPE_FLOAT),

        'gt_residue_index': np.arange(len(rec_dict['seq_aatype_num']), dtype=DTYPE_INT),  # (N_res)
        'gt_has_frame': rec_dict['has_frame'].astype(DTYPE_FLOAT),  # (N_res)
        'gt_lig_coords': np.stack(lig_coords).astype(DTYPE_FLOAT),  # (N_symm, N_atoms, 3)
        'gt_lig_has_coords': np.stack(lig_has_coords).astype(DTYPE_FLOAT),  # (N_symm, N_atoms)
    }

    if aff_known:
        out.update({
            'gt_affinity_label': aff_label.astype(DTYPE_INT),
            'gt_affinity_value': aff_value.astype(DTYPE_FLOAT),
            'gt_affinity_known': np.array(aff_known).astype(DTYPE_INT),
            'gt_affinity_start': np.array(aff_start).astype(DTYPE_INT),
            'gt_affinity_end': np.array(aff_end).astype(DTYPE_INT),
            'gt_affinity_lig_id': np.array(aff_lig_id).astype(DTYPE_INT)
        })

    return out


def target_group_featurize(case_dict, group_dict):
    #case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    #group_dir = case_dir / group_dict['name']
    group_feats = []
    for lig_dict in group_dict['ligands']:
        lig_feats = np.load(DATA_DIR / 'featurized' / lig_dict['sdf_id'] + '.ligand_feats.npy', allow_pickle=True).item()
        group_feats.append(lig_feats)

    group_1d = np.concatenate([x['atom_feats'] for x in group_feats], axis=0)
    group_2d = np.zeros((group_1d.shape[0], group_1d.shape[0], group_feats[0]['bonds_2d'].shape[-1]))
    start = 0
    start_list = []
    end_list = []
    for lig_feats in group_feats:
        start_list.append(start)
        lig_size = lig_feats['atom_feats'].shape[0]
        group_2d[start:start+lig_size, start:start+lig_size] = lig_feats['bonds_2d']
        start += lig_size
        end_list.append(start)

    return {
        'lig_1d': group_1d.astype(DTYPE_FLOAT),
        'lig_2d': group_2d.astype(DTYPE_FLOAT),
        'lig_bonded_2d': (group_2d.sum(-1) > 0).astype(DTYPE_FLOAT),
        'lig_starts': np.array(start_list, dtype=DTYPE_INT),
        'lig_ends': np.array(end_list, dtype=DTYPE_INT)
    }


def hhpred_template_lig_to_features(tpl_sdf_id, tar_num_atoms, m_tpl, m_tar):
    feats = np.load(DATA_DIR / 'featurized' / tpl_sdf_id + '.ligand_feats.npy', allow_pickle=True).item()

    coords = np.zeros((feats['atom_feats'].shape[0], 3))
    coords[list(feats['matches'][0])] = feats['coords']
    atom_present = np.zeros(feats['atom_feats'].shape[0])
    atom_present[list(feats['matches'][0])] = 1

    # shape + 1 because of a new symbol gap "-"
    dilated_atom_feats = np.zeros((tar_num_atoms, feats['atom_feats'].shape[-1] + 1))
    dilated_atom_feats[:, -1] = 1
    dilated_atom_feats[list(m_tar), :-1] = feats['atom_feats'][list(m_tpl)]
    dilated_atom_feats[list(m_tar), -1] = 0

    dilated_atom_present = np.zeros(tar_num_atoms)
    dilated_atom_present[list(m_tar)] = atom_present[list(m_tpl)]
    dilated_atom_present_2d = np.outer(dilated_atom_present, dilated_atom_present)

    dilated_crd = np.zeros((tar_num_atoms, 3))
    dilated_crd[list(m_tar)] = coords[list(m_tpl)]

    dmat = utils.calc_dmat(dilated_crd, dilated_crd)
    dgram = dmat_to_distogram(dmat, LIG_DISTOGRAM['min'], LIG_DISTOGRAM['max'], LIG_DISTOGRAM['num_bins'], mask=dilated_atom_present_2d < 1)

    dilated_bonds = np.zeros((tar_num_atoms, tar_num_atoms, feats['bonds_2d'].shape[-1]))
    tar_ids = list(zip(itertools.product(m_tar, m_tar)))
    tpl_ids = list(zip(itertools.product(m_tpl, m_tpl)))
    dilated_bonds[tar_ids[0], tar_ids[1]] = feats['bonds_2d'][tpl_ids[0], tpl_ids[1]]

    return {
        'atom_feats_1d': dilated_atom_feats,
        'atom_present_1d': dilated_atom_present,
        'atom_coords_1d': dilated_crd,
        'bond_feats_2d': dilated_bonds,
        'distogram_2d': dgram * dilated_atom_present_2d[..., None],
        'atom_present_2d': dilated_atom_present_2d,
        'match_tar': m_tar,
        'match_tpl': m_tpl
    }


def hh_template_featurize(tar_case_dict, tar_group_dict, tar_ligand_id, temp_dict, temp_case_dict, temp_sdf_id, tar_match, temp_match):
    #print(tar_case_dict['case_name'], tar_group_dict['name'], tar_ligand_id, temp_dict['lig_match']['ref_chemid'], temp_sdf_id, tar_match, temp_match)
    rec_literal = hhpred_template_rec_to_features(temp_case_dict, tar_case_dict, temp_dict['hhpred'])
    rec_feats = rec_literal_to_numeric(rec_literal, seq_include_gap=True)

    num_atoms_total = sum([x['num_heavy_atoms'] for x in tar_group_dict['ligands']])
    tar_atom_begin = sum([x['num_heavy_atoms'] for x in tar_group_dict['ligands'][:tar_ligand_id]])
    tar_match = [x + tar_atom_begin for x in tar_match]
    lig_feats = hhpred_template_lig_to_features(temp_sdf_id, num_atoms_total, temp_match, tar_match)

    rec_1d = np.concatenate([
        rec_feats['ag_aatype'],
        rec_feats['crd_mask'][..., None],
        rec_feats['crd_beta_mask'][..., None]
    ], axis=-1)

    extra = np.tile(rec_feats['ag_aatype'], (rec_feats['ag_aatype'].shape[0], 1, 1))
    rr_2d = np.concatenate([
        rec_feats['distogram_2d'],
        rec_feats['crd_beta_mask_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=2)

    lig_1d = np.concatenate([lig_feats['atom_feats_1d'], lig_feats['atom_present_1d'][..., None]], axis=-1)

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
        rl_dgram * rl_present_2d[..., None],
        rl_present_2d[..., None],
        np.tile(rec_feats['ag_aatype'], (lig_feats['atom_feats_1d'].shape[0], 1, 1)).transpose([1, 0, 2]),
        np.tile(lig_feats['atom_feats_1d'], (rec_feats['ag_aatype'].shape[0], 1, 1)),
    ], axis=-1)

    lr_2d = rl_2d.transpose([1, 0, 2])

    out = {
        'lig_1d': lig_1d.astype(DTYPE_FLOAT),
        'rec_1d': rec_1d.astype(DTYPE_FLOAT),
        'll_2d': ll_2d.astype(DTYPE_FLOAT),
        'rr_2d': rr_2d.astype(DTYPE_FLOAT),
        'rl_2d': rl_2d.astype(DTYPE_FLOAT),
        'lr_2d': lr_2d.astype(DTYPE_FLOAT)
    }
    return out


def fragment_template_group_featurize(case_dict, group_dict):
    case_dir = DATA_DIR / 'cases' / case_dict['case_name']
    group_dir = case_dir / group_dict['name']
    group_feats = []
    for lig_dict in group_dict['ligands']:
        lig_feats = np.load(DATA_DIR / 'featurized' / lig_dict['sdf_id'] + '.ligand_feats.npy', allow_pickle=True).item()
        group_feats.append(lig_feats)

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
        'distogram_2d': dgram * atom_present_2d[..., None],
        'matches_smiles_to_3d': matches
    }


def fragment_template_featurize(temp_case_dict, group_dict):
    lig_feats = fragment_template_group_featurize(temp_case_dict, group_dict)
    lig_1d = np.concatenate([lig_feats['atom_feats_1d'], lig_feats['atom_present_1d'][..., None]], axis=-1)

    extra = np.tile(lig_feats['atom_feats_1d'], (lig_feats['atom_feats_1d'].shape[0], 1, 1))
    ll_2d = np.concatenate([
        lig_feats['distogram_2d'],
        lig_feats['bond_feats_2d'],
        lig_feats['atom_present_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=-1)

    #start = time.time()
    rec_ag = prody.parsePDB(DATA_DIR / 'cases' / temp_case_dict['case_name'] / 'rec_orig.pdb')

    # select residues within FRAGMENT_TEMPLATE_RADIUS from the ligand
    residues = list(rec_ag.getHierView().iterResidues())
    residues_coords = np.stack([x['CA'].getCoords() if x['CA'] is not None else np.zeros(3) for x in residues])
    residues_has_coords = np.array([x['CA'] is not None for x in residues], dtype=DTYPE_INT)
    rl_ca_dmat = utils.calc_dmat(residues_coords, lig_feats['atom_coords_1d'])
    residues_mask = np.any(rl_ca_dmat <= FRAGMENT_TEMPLATE_RADIUS, axis=1) * residues_has_coords

    #print(time.time() - start); start = time.time()
    rec_literal = ag_to_features(rec_ag, temp_case_dict['entity_info']['pdb_aln'], temp_case_dict['entity_info']['entity_aln'], residues_mask=residues_mask)
    #print(time.time() - start); start = time.time()
    rec_feats = rec_literal_to_numeric(rec_literal)
    #print(time.time() - start); start = time.time()

    rec_1d = np.concatenate([
        rec_feats['seq_aatype'],
        rec_feats['crd_mask'][..., None],
        rec_feats['crd_beta_mask'][..., None]
    ], axis=-1)

    extra = np.tile(rec_feats['seq_aatype'], (rec_feats['seq_aatype'].shape[0], 1, 1))
    rr_2d = np.concatenate([
        rec_feats['distogram_2d'],
        rec_feats['crd_beta_mask_2d'][..., None],
        extra,
        extra.transpose([1, 0, 2]),
    ], axis=2)
    #print(time.time() - start); start = time.time()

    rl_cb_dmat = utils.calc_dmat(rec_feats['crd_beta'], lig_feats['atom_coords_1d'])
    rl_present_2d = np.outer(rec_feats['crd_beta_mask'], lig_feats['atom_present_1d'])
    rl_dgram = dmat_to_distogram(rl_cb_dmat, REC_LIG_DISTOGRAM['min'], REC_LIG_DISTOGRAM['max'], REC_LIG_DISTOGRAM['num_bins'], mask=rl_present_2d < 1)
    rl_2d = np.concatenate([
        rl_dgram * rl_present_2d[..., None],
        rl_present_2d[..., None],
        np.tile(rec_feats['seq_aatype'], (lig_feats['atom_feats_1d'].shape[0], 1, 1)).transpose([1, 0, 2]),
        np.tile(lig_feats['atom_feats_1d'], (rec_feats['seq_aatype'].shape[0], 1, 1)),
    ], axis=-1)

    lr_2d = rl_2d.transpose([1, 0, 2])

    #if FRAGMENT_TEMPLATE_RADIUS is not None:
    #    close_residue_mask = np.any(rl_dmat <= FRAGMENT_TEMPLATE_RADIUS, axis=1)
    #    rec_1d = rec_1d[close_residue_mask]
    #    rr_2d = rr_2d[close_residue_mask]
    #    rr_2d = rr_2d[:, close_residue_mask]
    #    rl_2d = rl_2d[close_residue_mask]
    #    lr_2d = lr_2d[:, close_residue_mask]
    #print(time.time() - start); start = time.time()

    out = {
        'lig_1d': lig_1d.astype(DTYPE_FLOAT),
        'rec_1d': rec_1d.astype(DTYPE_FLOAT),
        'll_2d': ll_2d.astype(DTYPE_FLOAT),
        'rr_2d': rr_2d.astype(DTYPE_FLOAT),
        'rl_2d': rl_2d.astype(DTYPE_FLOAT),
        'lr_2d': lr_2d.astype(DTYPE_FLOAT)
    }
    return out


def stack_with_padding(arrays: typing.List[np.ndarray]):
    padded_shape = np.stack([x.shape for x in arrays]).max(0)
    padded_arrays = []
    for array in arrays:
        padding = [(0, y - x) for x, y in zip(array.shape, padded_shape)]
        padded_arrays.append(np.pad(array, padding))
    return np.stack(padded_arrays)


def fragment_template_list_featurize(tpl_case_dicts, tpl_group_dicts, mappings):
    frag_feats_list = []

    for tpl_case_dict, tpl_group_dict, mapping in zip(tpl_case_dicts, tpl_group_dicts, mappings):
        #frag_feats = np.load(f"{DATA_DIR}/featurized/{tpl_case_dict['case_name']}.{tpl_group_dict['name']}.fragment_feats.npy", allow_pickle=True).item()
        frag_feats = fragment_template_featurize(tpl_case_dict, tpl_group_dict)
        frag_feats['fragment_mapping'] = np.array(mapping).astype(DTYPE_INT)

        num_res = len(frag_feats['rec_1d'])
        num_atoms = len(frag_feats['lig_1d'])
        frag_feats['num_res'] = num_res
        frag_feats['num_atoms'] = num_atoms

        frag_feats['ll_2d_mask'] = np.ones((num_atoms, num_atoms)).astype(DTYPE_FLOAT)
        frag_feats['rr_2d_mask'] = np.ones((num_res, num_res)).astype(DTYPE_FLOAT)
        frag_feats['rl_2d_mask'] = np.ones((num_res, num_atoms)).astype(DTYPE_FLOAT)
        frag_feats['lr_2d_mask'] = np.ones((num_atoms, num_res)).astype(DTYPE_FLOAT)

        frag_feats_list.append(frag_feats)

    #start = time.time()
    if len(frag_feats_list) > 0:
        output = defaultdict(list)
        for frag_feats in frag_feats_list:
            for k, v in frag_feats.items():
                output[k].append(v)
        #print(time.time() - start); start = time.time()
        output = {k: stack_with_padding(v) if isinstance(v[0], np.ndarray) else np.array(v, dtype=DTYPE_INT) for k, v in output.items()}
        #print(time.time() - start); start = time.time()
    else:
        output = None

    return output


def fragment_extra_featurize(tpl_case_dict, tpl_group_dict, mapping):
    lig_feats = fragment_template_group_featurize(tpl_case_dict, tpl_group_dict)

    rec_file = DATA_DIR / 'featurized' / tpl_case_dict['case_name'] + '.' + tpl_group_dict['name'] + '.pocket_ca.pdb'
    if not rec_file.exists():
        rec_file = DATA_DIR / 'cases' / tpl_case_dict['case_name'] / 'rec_orig.pdb'
    rec_ag = prody.parsePDB(rec_file)

    rec_coords = rec_ag.calpha.getCoords()
    rec_aatype = np.array([AATYPE_WITH_X.get(x, AATYPE_WITH_X['X']) for x in rec_ag.calpha.getSequence()], dtype=np.int)
    lr_dmat = utils.calc_dmat(lig_feats['atom_coords_1d'], rec_coords)

    close_mask = np.any(lr_dmat < LIG_EXTRA_DISTANCE['max'], axis=0)
    rec_aatype = rec_aatype[close_mask]
    lr_dmat = lr_dmat[:, close_mask]
    rec_aatype_onehot = np.zeros((rec_aatype.shape[0], len(AATYPE_WITH_X)))
    rec_aatype_onehot[range(rec_aatype.shape[0]), rec_aatype] = 1.0
    lr_dram = dmat_to_distogram(lr_dmat, LIG_EXTRA_DISTANCE['min'], LIG_EXTRA_DISTANCE['max'], LIG_EXTRA_DISTANCE['num_bins'])

    counts = np.matmul(lr_dram.swapaxes(1, 2), rec_aatype_onehot[None]).reshape(lr_dram.shape[0], -1)
    #counts = np.ones((lig_feats['atom_feats_1d'].shape[0], 236 - lig_feats['atom_feats_1d'].shape[-1]))

    out = np.zeros((len(mapping), lig_feats['atom_feats_1d'].shape[-1] + 1 + counts.shape[1] + 1), dtype=DTYPE_FLOAT)
    mapping = np.array(mapping, dtype=np.int)
    tpl_ids = mapping[mapping > -1]
    out[mapping > -1, :lig_feats['atom_feats_1d'].shape[-1]] = lig_feats['atom_feats_1d'][tpl_ids]
    out[mapping > -1, lig_feats['atom_feats_1d'].shape[-1]] = lig_feats['atom_present_1d'][tpl_ids]
    out[mapping > -1, lig_feats['atom_feats_1d'].shape[-1]+1:out.shape[-1]-1] = counts[tpl_ids]
    out *= (mapping > -1).astype(DTYPE_FLOAT)[:, None]
    out[mapping == -1, -1] = 1  # gap (unmapped target atoms)
    return out  # (Natoms, Nfeats)


def fragment_extra_featurize_mockup_(tpl_case_dict, tpl_group_dict, mapping):
    lig_feats = fragment_template_group_featurize(tpl_case_dict, tpl_group_dict)
    return np.zeros((len(mapping), 238), dtype=DTYPE_FLOAT)


def fragment_extra_list_featurize(tpl_case_dicts, tpl_group_dicts, mappings):
    feats_list = []

    for tpl_case_dict, tpl_group_dict, mapping in zip(tpl_case_dicts, tpl_group_dicts, mappings):
        lig_feats = fragment_extra_featurize(tpl_case_dict, tpl_group_dict, mapping)
        feats_list.append(lig_feats)

    return np.stack(feats_list)


def example2():
    case_dict = utils.read_json(DATA_DIR / 'cases/3HCF_A/case.json')
    group_dict = case_dict['ligand_groups'][1]
    #print(fragment_template_featurize(case_dict, group_dict))
    print([(k, v.shape) for k, v in target_rec_featurize(case_dict).items()])
    print([(k, v.shape) for k, v in target_group_featurize(case_dict, group_dict).items()])
    print([(k, v.shape) for k, v in list(fragment_template_featurize(case_dict, group_dict).items())[:]])


def example3():
    pass


if __name__ == '__main__':
    example2()
