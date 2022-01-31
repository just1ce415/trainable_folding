import contextlib
import os

import gc
import numpy as np
import prody
import torch
from path import Path
from rdkit import Chem
from sblu.ft import read_rotations, read_ftresults, apply_ftresults_atom_group
from tqdm import tqdm
from rdkit.Chem import AllChem
from io import StringIO

import sys
sys.path.insert(0, '/home/ignatovmg/projects/fft_affinity/trainings/run14_summit')

import lig_to_json
import utils_loc
from dataset import make_lig_graph, DTYPE_FLOAT, DTYPE_INT
from rec_feats import make_protein_grids, ag_to_tokens_old_like
from model import NetSE3


torch.set_num_threads(1)


def mol_to_ag(mol):
    return prody.parsePDBStream(StringIO(Chem.MolToPDBBlock(mol)))


def _cmp_equal_elements(ag, mol_rdkit):
    ag_e = [x.upper() for x in ag.getElements()]
    mol_e = [x.GetSymbol().upper() for x in mol_rdkit.GetAtoms()]
    assert all([x == y for x, y in zip(ag_e, mol_e)]), \
        f'Elements are different:\n{ag_e}\n{mol_e}'


from mol_grid import GridMaker, calc_sasa


def make_surface_mask(rec_ag, sasa=None, **box_kwargs):
    if sasa is None:
        sasa = calc_sasa(rec_ag, normalize=False)
    return GridMaker(atom_radius=7, config=lambda x: [x], mode='sphere', **box_kwargs).make_grids(rec_ag, weights=sasa)


def _load_model(checkpoint, device):
    model = NetSE3()
    pth = torch.load(checkpoint)
    print('Loading saved model from', checkpoint)
    model.load_state_dict(pth['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def _to_tensor(x, device):
    return torch.tensor(x, device=device, requires_grad=False)


def _grid_to_tensor(grid, device):
    return _to_tensor(grid.astype(np.float32), device)


def _assert_equal_elements_ag_ag(ag1, ag2):
    ag_e = [x.upper() for x in ag1.getElements()]
    mol_e = [x.upper() for x in ag2.getElements()]
    assert all([x == y for x, y in zip(ag_e, mol_e)]), \
        f'Elements are different:\n{ag_e}\n{mol_e}'


def _assert_equal_elements_ag_rd(ag, mol_rdkit):
    ag_e = [x.upper() for x in ag.getElements()]
    mol_e = [x.GetSymbol().upper() for x in mol_rdkit.GetAtoms()]
    assert all([x == y for x, y in zip(ag_e, mol_e)]), \
        f'Elements are different:\n{ag_e}\n{mol_e}'


def _select_top_energy(energy_grid, rec_grid, lig_grid, ntop=1, radius=3):
    cor_shape = list(energy_grid.shape[-3:])
    lig_shape = np.array(lig_grid.grid.shape[1:])
    energies = []
    tvs = []

    energy_grid = energy_grid.flatten()
    for min_idx in energy_grid.argsort():
        if len(tvs) >= ntop:
            break

        min_idx = min_idx.item()
        min_idx3d = np.unravel_index(min_idx, cor_shape)

        #if mask_grid[min_idx3d] > 0:
        #    continue
        #mask_grid[
        #    min(0, min_idx3d[0]-radius):min_idx3d[0]+radius,
        #    min(0, min_idx3d[1]-radius):min_idx3d[1]+radius,
        #    min(0, min_idx3d[2]-radius):min_idx3d[2]+radius
        #] = 1

        min_energy = energy_grid[min_idx].item()
        tv = rec_grid.origin - lig_grid.origin + ((1 - lig_shape) + min_idx3d) * lig_grid.delta

        dists = [np.power(tv - x, 2).sum() for x in tvs]
        if len(dists) > 0 and any([x < radius**2 for x in dists]):
            continue

        energies.append(min_energy)
        tvs.append(tv)

    return energies, tvs


def _select_top_energy_square(energy_grid, rec_grid, lig_grid, ntop=1, radius=3):
    cor_shape = list(energy_grid.shape[-3:])
    lig_shape = np.array(lig_grid.grid.shape[1:])
    energies = []
    tvs = []
    rad_cells = (radius / lig_grid.delta).astype(int)
    print(rad_cells)

    energy_grid = energy_grid[0, 0]
    while len(tvs) < ntop:
        min_idx = torch.argmin(energy_grid).item()
        min_idx3d = np.unravel_index(min_idx, cor_shape)

        min_energy = energy_grid[tuple(min_idx3d)].item()
        tv = rec_grid.origin - lig_grid.origin + ((1 - lig_shape) + min_idx3d) * lig_grid.delta

        energies.append(min_energy)
        tvs.append(tv)

        energy_grid[
        min(0, min_idx3d[0]-rad_cells[0]):min_idx3d[0]+rad_cells[0],
        min(0, min_idx3d[1]-rad_cells[1]):min_idx3d[1]+rad_cells[1],
        min(0, min_idx3d[2]-rad_cells[2]):min_idx3d[2]+rad_cells[2]
        ] = 10000

    return energies, tvs


def _write_ft_results(fname, ft_results):
    with open(fname, 'w') as f:
        for x in ft_results:
            f.write(f'{x[0]:<4d} {x[1][0]:<8f} {x[1][1]:<8f} {x[1][2]:<8f} {x[2]:<8f}\n')


def _read_ft_results(fname):
    ft_results = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.split()
            ft_results.append([int(line[0]), (float(line[1]), float(line[2]), float(line[3])), float(line[4])])
    return ft_results


def _get_ag_dims(ag):
    crd = ag.getCoords()
    return crd.max(0) - crd.min(0)


def _calc_symm_rmsd(ag, ref_cset_id, symmetries):
    csets = ag.getCoordsets()
    ref_crd = csets[ref_cset_id]
    rmsds = []
    ele = ag.getElements()
    for m in symmetries:
        ref_atoms = list(symmetries[0])
        mob_atoms = list(m)
        assert all(ele[i] == ele[j] for i, j in zip(ref_atoms, mob_atoms)), f'Elements are different for symmetry {m}'

        rmsd = np.power(ref_crd[None, ref_atoms, :] - csets[:, mob_atoms, :], 2).sum((1, 2)) / ref_crd.shape[0]
        rmsds.append(rmsd)
    rmsds = np.sqrt(np.stack(rmsds).min(0))
    return rmsds


def _boltzman_clustering(ag, energies, symmetries, radius, min_size=1, max_clusters=None):
    energies = np.array(energies)
    unused = np.array(range(len(energies)))
    assert ag.numCoordsets() == len(energies), (ag.numCoordsets(), len(energies))

    clusters = []
    while len(unused) > 0:
        center = unused[np.argmin(energies[unused])]
        rmsd = _calc_symm_rmsd(ag, center, symmetries)
        members = unused[np.where(rmsd[unused] < radius)[0]]
        unused = unused[~np.isin(unused, members)]
        if len(members) >= min_size:
            clusters.append((center.item(), members.tolist()))
        if max_clusters is not None and len(clusters) == max_clusters:
            break

    return clusters


def _dock_single_rotation(args):
    net, rot_id, rot_mat, lig_ag, lig_rd, device, rec_grid, rec_tokens, sasa_grid, tr_per_rot = args
    lig_ag_rotated = lig_ag.copy()
    lig_coords = lig_ag.getCoords()
    lig_coords = np.dot(lig_coords - lig_coords.mean(0), rot_mat.T) + lig_coords.mean(0)
    lig_ag_rotated._setCoords(lig_coords, overwrite=True)
    G, lig_grid, lig_pairwise = make_lig_graph(lig_to_json.lig_to_dict(lig_rd), lig_ag_rotated, box_kwargs={'cell': 1.0, 'padding': 7.0, 'mode': 'point'})
    sample = {
        'id': _to_tensor([123], device),  # to make model save grids for rot_id = 0
        'rec_grid_shape': _to_tensor(np.array(rec_grid.grid.shape[1:]).astype(DTYPE_INT), device)[None],
        'rec_grid_origin': _to_tensor(rec_grid.origin.astype(DTYPE_FLOAT), device)[None],
        'rec_grid_delta': _to_tensor(rec_grid.delta.astype(DTYPE_FLOAT), device)[None],
        'rec_coords': _to_tensor(np.array(rec_tokens['coords']).astype(DTYPE_FLOAT), device)[None],
        'rec_atom_feats': _to_tensor(np.array(rec_tokens['features']).astype(DTYPE_FLOAT), device)[None],
        'sasa_grid': _grid_to_tensor(sasa_grid.grid.astype(DTYPE_FLOAT), device)[None],
        'node_features': _to_tensor(G['f'], device)[None],
        'lig_grid_shape': _to_tensor(np.array(lig_grid.grid.shape[1:]).astype(DTYPE_INT), device)[None],
        'lig_grid_origin': _to_tensor(G['grid_origin'][0], device)[None],
        'lig_grid_delta': _to_tensor(G['grid_delta'][0], device)[None],
        'lig_pairwise': _to_tensor(lig_pairwise, device)[None],
        'lig_coords': _to_tensor(G['x'], device)[None],
    }

    # get energy grid batch
    energy_grid = net(sample)

    # select top poses for each rotation
    min_energies, tvs = _select_top_energy(energy_grid.detach(), rec_grid, lig_grid, ntop=tr_per_rot)
    ft_results_local = []
    for min_energy, tv in zip(min_energies, tvs):
        ft_results_local.append((rot_id, tuple(tv), min_energy))

    for x in sample.values():
        del x

    # important, otherwise memory overfills
    del energy_grid
    if device != 'cpu':
        torch.cuda.empty_cache()
    gc.collect()

    return ft_results_local


def _sample_ligand(
        mol_file,
        lig_rd_target,
        cluster_radius,
        num_confs=100,
        frac=0.5,
        num_threads=32,
        seed=123,
        align=True,
        assign_chirality=True
):
    lig_rd = Chem.MolFromMolFile(mol_file, removeHs=False)
    if assign_chirality:
        Chem.AssignAtomChiralTagsFromStructure(lig_rd)
    lig_rd = AllChem.AddHs(lig_rd)
    AllChem.EmbedMultipleConfs(lig_rd, numConfs=num_confs, numThreads=num_threads, randomSeed=seed)
    energies = AllChem.MMFFOptimizeMoleculeConfs(lig_rd, numThreads=num_threads, maxIters=1000)

    # take best energy half of the conformers
    low_e_confs = np.argsort([x[1] for x in energies])
    low_e_confs = [x for x in low_e_confs if energies[x][0] == 0]
    assert len(low_e_confs) != 0, 'No conformations converged'
    low_e_confs = low_e_confs[:max(1, int(len(low_e_confs) * frac))]
    energies = [energies[x][1] for x in low_e_confs]

    conf_coords = np.stack([lig_rd.GetConformer(x).GetPositions() for x in range(lig_rd.GetNumConformers())])
    conf_coords = conf_coords[low_e_confs]
    lig_ag = mol_to_ag(lig_rd)
    lig_ag._setCoords(conf_coords, overwrite=True)
    prody.alignCoordsets(lig_ag.heavy)

    lig_rd = Chem.RemoveHs(lig_rd)
    symms = lig_rd.GetSubstructMatches(lig_rd)
    _cmp_equal_elements(lig_ag[list(symms[0])], lig_rd)
    clusters = _boltzman_clustering(lig_ag, energies, symms, cluster_radius)
    lig_ag._setCoords(conf_coords[[x for x, _ in clusters]], overwrite=True)

    lig_ag_target = mol_to_ag(lig_rd_target)
    lig_ag_final = lig_ag_target.copy()
    matches = lig_rd.GetSubstructMatches(lig_rd_target, uniquify=True)
    assert len(matches) > 0
    lig_ag_final._setCoords(lig_ag.getCoordsets()[:, list(matches[0])], overwrite=True)

    rmsds = None
    if align:
        lig_ag_final._setCoords(np.concatenate([lig_ag_target.getCoordsets(), lig_ag_final.getCoordsets()]), overwrite=True)
        lig_ag_final.setACSIndex(0)
        prody.alignCoordsets(lig_ag_final.heavy)
        rmsds = prody.calcRMSD(lig_ag_final.heavy)[1:]
        lig_ag_final._setCoords(lig_ag_final.getCoordsets()[1:], overwrite=True)

    return lig_ag_final, clusters, energies, rmsds


def _get_ligand_confs(lig_mol, lig_rd_target, symms, rots, cluster_radius=2.0):
    assert len(rots.shape) == 3 and list(rots.shape[1:]) == [3, 3], rots.shape

    lig_ag, clusters, energies, rmsds = _sample_ligand(lig_mol, lig_rd_target, cluster_radius, num_confs=100, num_threads=16)
    csets = lig_ag.getCoordsets()
    rotated_csets = np.dot(csets - csets.mean(1, keepdims=True), rots.transpose([0, 2, 1])) + csets.mean(1, keepdims=True)[:, None]
    assert list(rotated_csets.shape) == [csets.shape[0], csets.shape[1], rots.shape[0], 3], rotated_csets.shape
    rotated_csets = rotated_csets.transpose([0, 2, 1, 3]).reshape(-1, csets.shape[1], 3)
    lig_ag._setCoords(rotated_csets, overwrite=True)

    clusters = _boltzman_clustering(lig_ag, np.repeat([energies[x] for x, _ in clusters], rots.shape[0]), symms, cluster_radius)
    lig_ag._setCoords(rotated_csets[[x for x, _ in clusters]], overwrite=True)
    return lig_ag


def _select_AF_aln(af_ag, pdb_ag):
    af_aln_ag, res, (af_aln, pdb_aln) = utils_loc.align(af_ag, pdb_ag)
    res_list = list(af_aln_ag.getHierView().iterResidues())
    assert len(res_list) == len(af_aln.replace('-', '')), (len(res_list), len(af_aln.replace('-', '')))

    aligned_ids = []
    af_id = -1
    for cursor, (af_aa, pdb_aa) in enumerate(zip(af_aln, pdb_aln)):
        if af_aa != '-':
            af_id += 1
            if pdb_aa != '-':
                aligned_ids.append(af_id)

    af_slice = res_list[aligned_ids[0]].copy()
    for resi in aligned_ids[1:]:
        af_slice += res_list[resi].copy()
    return af_slice


def _calc_rmsd(ag, ref_ag, ref_matches, symmetries):
    rmsd = []
    for match in ref_matches:
        assert np.all(ag.getElements()[list(symmetries[0])] == ref_ag.getElements()[list(match)])
        rmsd.append(prody.calcRMSD(ref_ag.getCoords()[list(match)], ag.getCoordsets()[:, list(symmetries[0])]))
    rmsd = np.stack(rmsd).min(0)
    return rmsd


def _dock_with_sampling(
        rec_pdb,
        frag_mol,
        rot_file,
        checkpoint,
        device='cuda:0',
        num_rots=500,
        tr_per_rot=5,
        num_poses=2500,
        crys_mol=None,
        clus_radius=3
):
    rec_ag = prody.parsePDB(rec_pdb).protein.copy()
    lig_rd_orig = Chem.MolFromMolFile(frag_mol, removeHs=True)
    lig_ag = mol_to_ag(lig_rd_orig)
    lig_rd_noh = Chem.MolFromMolFile(frag_mol, removeHs=True)
    assert len(lig_ag.heavy) == lig_rd_noh.GetNumAtoms(), (len(lig_ag.heavy), lig_rd_noh.GetNumAtoms())
    symmetries = lig_rd_orig.GetSubstructMatches(lig_rd_noh, uniquify=False)
    sasa = None #calc_sasa(rec_ag, normalize=False)

    if crys_mol is None:
        lig_ag_crys = lig_ag.copy()
        crys_matches = symmetries
    else:
        lig_rd_crys = Chem.MolFromMolFile(crys_mol, removeHs=False)
        lig_ag_crys = mol_to_ag(lig_rd_crys)
        crys_matches = lig_rd_crys.GetSubstructMatches(lig_rd_noh, uniquify=False)
        # print(crys_matches)
        #_assert_equal_elements(lig_ag.heavy, lig_ag_crys.heavy)

    assert len(crys_matches) != 0

    prody.writePDB('rec.pdb', rec_ag)
    prody.writePDB('lig.pdb', lig_ag)

    rots = read_rotations(rot_file, num_rots)
    rots = np.insert(rots, 0, np.eye(3, 3), axis=0)[:num_rots]
    lig_ag_sampled = _get_ligand_confs(frag_mol, lig_rd_orig, symmetries, rots)
    prody.writePDB('lig_sampled.pdb', lig_ag_sampled)
    #lig_ag_sampled = prody.parsePDB('lig_sampled.pdb')
    np.savetxt('rmsd_lig_sampled.txt', _calc_rmsd(lig_ag_sampled, lig_ag_crys, crys_matches, symmetries))

    with torch.no_grad():
        rec_grid = make_protein_grids(rec_ag, cell=1.0, padding=7, mode='point')
        sasa_grid = make_surface_mask(rec_ag, sasa=sasa, box_shape=rec_grid.grid.shape[1:], box_origin=rec_grid.origin, cell=1.0)
        rec_tokens = ag_to_tokens_old_like(rec_ag)
        net = _load_model(checkpoint, device)

        lig_blank = lig_ag_sampled.copy()
        dock_poses = []
        for conf_id, conf_coords in tqdm(list(enumerate(lig_ag_sampled.getCoordsets()))):
            lig_blank._setCoords(conf_coords)
            ft_results_conf = _dock_single_rotation([net, 0, np.eye(3, 3), lig_blank, lig_rd_orig, device, rec_grid, rec_tokens, sasa_grid, tr_per_rot])
            for ft_id, (_, tv, E) in enumerate(ft_results_conf):
                dock_poses.append((conf_id, ft_id, conf_coords + np.array(tv)[None], E))

    dock_poses_all = sorted(dock_poses, key=lambda x: x[-1])[:num_poses]

    for num_tr in [1, 3, 5]:
        dock_poses = [x for x in dock_poses_all if x[1] < num_tr]
        lig_ft_ag = lig_ag_sampled.copy()
        lig_ft_ag._setCoords(np.stack([x[2] for x in dock_poses]), overwrite=True)
        prody.writePDB(f'lig_ft_tr{num_tr}.pdb', lig_ft_ag)

        clusters = _boltzman_clustering(lig_ft_ag, [x[-1] for x in dock_poses], symmetries, clus_radius)
        utils_loc.write_json(clusters, f'clusters_bzman_tr{num_tr}.json')

        lig_clus_ag = lig_ag.copy()
        lig_clus_ag._setCoords(lig_ft_ag.getCoordsets()[[x[0] for x in clusters]], overwrite=True)
        prody.writePDB(f'lig_clus_bzman_tr{num_tr}.pdb', lig_clus_ag)

        rmsd = _calc_rmsd(lig_ft_ag, lig_ag_crys, crys_matches, symmetries)
        np.savetxt(f'rmsd_tr{num_tr}.txt', rmsd)
        np.savetxt(f'rmsd_clus_bzman_tr{num_tr}.txt', rmsd[[x[0] for x in clusters]])


@contextlib.contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


import traceback


def dock_cases():
    cases = utils_loc.read_json('data/train_split/train_12k_cleaned.json')['cases'] + utils_loc.read_json('data/train_split/valid_12k_cleaned.json')['cases']
    oldpwd = Path.getcwd()
    checkpoint = Path('/home/ignatovmg/projects/fft_affinity/trainings/run14_summit/runs/run14_summit_tune_one_evo/epoch_119_loss_3.131.pth').abspath()
    for case in cases:
        case_dir = Path('data/cases/' + case['case_name']).abspath()
        rec_pdb = case_dir / 'AF_aln.pdb'
        crys_pdb = case_dir / 'rec_orig.pdb'
        for lig_mol in sorted(case_dir.glob(case['group_name'] + '/*_.mol')):
            wdir = (Path('data/decoys').mkdir_p().abspath() / lig_mol.basename().stripext()).mkdir_p()
            print(wdir)
            with cwd(wdir):
                if Path('rmsd_tr1.txt').exists():
                    continue
                Path(lig_mol).copy('lig_input.mol')
                Path(rec_pdb).copy('AF_aln.pdb')
                Path(crys_pdb).copy('rec_crys.pdb')
                prody.writePDB('AF_slice.pdb', _select_AF_aln(prody.parsePDB('AF_aln.pdb'), prody.parsePDB('rec_crys.pdb')))
                try:
                    _dock_with_sampling(
                        'AF_slice.pdb',
                        'lig_input.mol',
                        oldpwd / '/home/ignatovmg/projects/fft_affinity/prms/rot70k.0.0.6.jm.mol2',
                        checkpoint,
                        clus_radius=4,
                        device='cuda:0',
                        crys_mol='lig_input.mol',
                        num_rots=10
                    )
                except:
                    traceback.print_exc()


if __name__ == '__main__':
    dock_cases()
