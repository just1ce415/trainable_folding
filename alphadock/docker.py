import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import functools

import modules
import structure
import r3
import quat_affine
import all_atom
import utils
import lddt


class DockerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InputEmbedder = modules.InputEmbedder(config['InputEmbedder'], global_config)
        self.Evoformer = nn.ModuleList([modules.EvoformerIteration(config['Evoformer']['EvoformerIteration'], global_config) for x in range(config['Evoformer']['num_iter'])])
        self.EvoformerExtractSingleLig = nn.Linear(global_config['rep_1d']['num_c'], global_config['num_single_c'])
        self.EvoformerExtractSingleRec = nn.Linear(global_config['rep_1d']['num_c'], global_config['num_single_c'])
        self.StructureModule = structure.StructureModule(config['StructureModule'], global_config)

        self.config = config
        self.global_config = global_config

    def forward(self, input):
        x = self.InputEmbedder(input)

        for evo_iter in self.Evoformer:
            if self.config['Evoformer']['EvoformerIteration']['checkpoint']:
                x = checkpoint(lambda a, b, c: evo_iter({'l1d': a, 'r1d': b, 'pair': c}), x['l1d'], x['r1d'], x['pair'])
            else:
                x = evo_iter(x)

        pair = x['pair']
        rec_single = self.EvoformerExtractSingleRec(x['r1d'])
        lig_single = self.EvoformerExtractSingleLig(x['l1d'][:, 0])
        struct_out = self.StructureModule({
            'r1d': rec_single,
            'l1d': lig_single,
            'pair': pair,
            'rec_bb_affine': input['target']['rec_bb_affine'],
            'rec_bb_affine_mask': input['target']['rec_bb_affine_mask'],
        })

        assert struct_out['rec_T'].shape[0] == 1
        final_all_atom = all_atom.backbone_affine_and_torsions_to_all_atom(
            struct_out['rec_T'][0][-1].clone(),
            struct_out['rec_torsions'][0][-1],
            input['target']['rec_aatype'][0]
        )
        print({k: v.shape for k, v in struct_out.items()})

        #print()
        return loss(input, struct_out, final_all_atom, self.global_config)


def loss(batch, struct_out, final_all_atom, config):
    # process predictions
    rec_traj = struct_out['rec_T'][0]  # (N_traj, N_res, 7)
    lig_traj = struct_out['lig_T'][0]  # (N_traj, N_atoms, 7)
    rec_traj[..., -3:] = rec_traj[..., -3:] * config['position_scale']
    lig_traj[..., -3:] = lig_traj[..., -3:] * config['position_scale']
    num_traj = rec_traj.shape[0]

    rec_final_pred_frames = final_all_atom['frames']  # r3.Rigids (N, 8)
    rec_final_atom14_pred_coords_vecs = final_all_atom['atom_pos']  # r3.Vecs (N, 14)
    rec_final_atom14_pred_coords_tensor = r3.vecs_to_tensor(final_all_atom['atom_pos'])  # Tensor (N, 14, 3)
    rec_final_atom14_pred_mask = batch['target']['rec_atom14_atom_exists'][0]  # Tensor (N, 14)
    #rec_final_atom37_pred_coords = all_atom.atom14_to_atom37(rec_final_atom14_pred_coords, batch['target']['rec_aatype'][0])
    #rec_final_atom37_pred_mask = all_atom.atom14_to_atom37(batch['target']['rec_atom14_atom_exists'][0], batch['target']['rec_aatype'][0])

    lig_final_pred_coords_tensor = lig_traj[-1, :, -3:]  #   Tensor (Natoms, 3)
    #lig_final_pred_coords_vecs = r3.vecs_from_tensor(lig_final_pred_coords_tensor)   # Vecs (Natoms)

    # process ground truth
    rec_gt_atom37_coords = all_atom.atom14_to_atom37(batch['ground_truth']['gt_atom14_coords'][0], batch['ground_truth']['gt_aatype'][0])
    rec_gt_atom37_mask = all_atom.atom14_to_atom37(batch['ground_truth']['gt_atom14_has_coords'][0], batch['ground_truth']['gt_aatype'][0])
    gt_all_frames = all_atom.atom37_to_frames(batch['ground_truth']['gt_aatype'][0], rec_gt_atom37_coords, rec_gt_atom37_mask)

    # (N_traj, N_res)
    gt_bb_frames = r3.rigids_from_tensor_flat12(gt_all_frames['rigidgroups_gt_frames'][:, 0, :].tile(num_traj, 1, 1))
    gt_bb_mask = gt_all_frames['rigidgroups_gt_exists'][:, 0].tile(num_traj, 1)

    num_lig_symm = batch['ground_truth']['gt_lig_coords'].shape[1]
    lig_gt_mask = batch['ground_truth']['gt_lig_has_coords'][0]
    lig_gt_coords = batch['ground_truth']['gt_lig_coords'][0]

    pred_bb_frames = r3.rigids_from_quataffine(quat_affine.QuatAffine.from_tensor(rec_traj))

    # TODO: disable clamping for 10% batches
    loss_bb_rec_rec = all_atom.frame_aligned_point_error(
        pred_bb_frames,
        gt_bb_frames,
        gt_bb_mask,
        pred_bb_frames.trans,
        gt_bb_frames.trans,
        gt_bb_mask,
        config['loss']['fape_loss_unit_distance'],
        config['loss']['fape_clamp_distance']
    )

    loss_bb_rec_lig = all_atom.frame_aligned_point_error(
        r3.apply_tree_rigids(lambda x: x.repeat_interleave(num_lig_symm, dim=0), pred_bb_frames),
        r3.apply_tree_rigids(lambda x: x.repeat_interleave(num_lig_symm, dim=0), gt_bb_frames),
        gt_bb_mask.repeat_interleave(num_lig_symm, dim=0),
        r3.vecs_from_tensor(lig_traj[:, :, -3:].repeat_interleave(num_lig_symm, dim=0)),
        r3.vecs_from_tensor(lig_gt_coords.tile(num_traj, 1, 1)),
        lig_gt_mask.tile(num_traj, 1),
        config['loss']['fape_loss_unit_distance'],
        config['loss']['fape_clamp_distance']
    )
    loss_bb_rec_lig = loss_bb_rec_lig.reshape(num_traj, num_lig_symm)

    # compute all atom FAPE
    renamed = all_atom.compute_renamed_ground_truth(
        dict(
            atom14_gt_positions=batch['ground_truth']['gt_atom14_coords'][0],
            atom14_alt_gt_positions=batch['ground_truth']['gt_atom14_coords_alt'][0],
            atom14_atom_is_ambiguous=batch['ground_truth']['gt_atom14_atom_is_ambiguous'][0],
            atom14_gt_exists=batch['ground_truth']['gt_atom14_has_coords'][0],
            atom14_alt_gt_exists=batch['ground_truth']['gt_atom14_has_coords_alt'][0],
            atom14_pred_positions=rec_final_atom14_pred_coords_tensor,
            atom14_atom_exists=rec_final_atom14_pred_mask
        ),
        rec_final_atom14_pred_coords_tensor)

    alt_naming_is_better = renamed['alt_naming_is_better']

    # (N, 8, 12)
    renamed_gt_frames_flat12 = (
        (1. - alt_naming_is_better[:, None, None])
        * gt_all_frames['rigidgroups_gt_frames']
        + alt_naming_is_better[:, None, None]
        * gt_all_frames['rigidgroups_alt_gt_frames']
    )

    renamed_gt_frames_flat = r3.rigids_from_tensor_flat12(renamed_gt_frames_flat12.reshape(-1, 12))
    renamed_gt_frames_mask_flat = gt_all_frames['rigidgroups_gt_exists'].flatten()
    renamed_gt_coords_flat = r3.vecs_from_tensor(renamed['renamed_atom14_gt_positions'].reshape(-1, 3))
    renamed_gt_coords_mask_flat = renamed['renamed_atom14_gt_exists'].flatten()
    rec_final_pred_frames_flat = r3.apply_tree_rigids(lambda x: x.flatten(), rec_final_pred_frames)
    rec_final_atom14_pred_coords_flat = r3.apply_tree_vecs(lambda x: x.flatten(), rec_final_atom14_pred_coords_vecs)

    # Compute frame_aligned_point_error score for the final layer.
    loss_aa_rec_rec = all_atom.frame_aligned_point_error(
        rec_final_pred_frames_flat,
        renamed_gt_frames_flat,
        renamed_gt_frames_mask_flat,
        rec_final_atom14_pred_coords_flat,
        renamed_gt_coords_flat,
        renamed_gt_coords_mask_flat,
        config['loss']['fape_loss_unit_distance'],
        config['loss']['fape_clamp_distance']
    )

    loss_aa_rec_lig = all_atom.frame_aligned_point_error(
        r3.apply_tree_rigids(lambda x: x.tile(num_lig_symm, 1), rec_final_pred_frames_flat),
        r3.apply_tree_rigids(lambda x: x.tile(num_lig_symm, 1), renamed_gt_frames_flat),
        renamed_gt_frames_mask_flat.tile(num_lig_symm, 1),
        r3.vecs_from_tensor(lig_final_pred_coords_tensor.tile(num_lig_symm, 1, 1)),
        r3.vecs_from_tensor(lig_gt_coords),
        lig_gt_mask,
        config['loss']['fape_loss_unit_distance'],
        config['loss']['fape_clamp_distance']
    )

    loss_chi = torsion_loss(batch, struct_out)

    loss_lddt = lddt_loss(batch, struct_out, config)

    loss_total = loss_bb_rec_rec.mean() * config['loss']['loss_bb_rec_rec_weight'] + \
        loss_bb_rec_lig.min(-1).values.mean() * config['loss']['loss_bb_rec_lig_weight'] + \
        loss_aa_rec_rec * config['loss']['loss_aa_rec_rec_weight'] + \
        loss_aa_rec_lig.min() * config['loss']['loss_aa_rec_lig_weight'] + \
        loss_chi['chi_loss'].mean() * config['loss']['loss_chi_value_weight'] + \
        loss_chi['norm_loss'].mean() * config['loss']['loss_chi_norm_weight'] + \
        loss_lddt['rec_rec_lddt_loss'] * config['loss']['loss_rec_rec_lddt_weight'] + \
        loss_lddt['lig_rec_lddt_loss'] * config['loss']['loss_lig_rec_lddt_weight']

    return {
        'loss_total': loss_total,
        'loss_torsions': loss_chi,
        'loss_lddt': loss_lddt,
        'loss_fape': {
            'loss_bb_rec_rec': loss_bb_rec_rec,
            'loss_bb_rec_lig': loss_bb_rec_lig,
            'loss_aa_rec_rec': loss_aa_rec_rec,
            'loss_aa_rec_lig': loss_aa_rec_lig,
        }
    }


def torsion_loss(batch, struct_out):
    eps = 1e-6

    pred_torsions_unnorm = struct_out['rec_torsions'][0]  # (Ntraj, Nres, 7, 2) - unnormalized
    gt_torsions = batch['ground_truth']['gt_torsions_sin_cos'][0]   # (Nres, 7, 2)
    gt_torsions_alt = batch['ground_truth']['gt_torsions_sin_cos_alt'][0]   # (Nres, 7, 2)
    gt_torsions_mask = batch['ground_truth']['gt_torsions_mask'][0]   # (Nres, 7)

    pred_torsions_norm = all_atom.l2_normalize(pred_torsions_unnorm, axis=-1)
    chi_squared = utils.squared_difference(pred_torsions_norm, gt_torsions[None]).sum(-1)
    chi_squared_alt = utils.squared_difference(pred_torsions_norm, gt_torsions_alt[None]).sum(-1)
    chi_loss = (torch.minimum(chi_squared, chi_squared_alt) * gt_torsions_mask[None]).sum((1, 2)) / gt_torsions_mask.sum()

    norm_loss = torch.abs(torch.sqrt(torch.sum(torch.square(pred_torsions_unnorm), dim=-1) + eps) - 1.0)
    norm_loss *= gt_torsions_mask[None]
    norm_loss = norm_loss.sum((1, 2)) / gt_torsions_mask.sum()

    return {
        'chi_loss': chi_loss,
        'norm_loss': norm_loss
    }


def lddt_loss(batch, struct_out, config):
    """
    """

    predicted_rec_lddt = struct_out['rec_lddt'][0]
    predicted_lig_lddt = struct_out['lig_lddt'][0]
    rec_pred_coords = struct_out['rec_T'][0, :, :, -3:] * config['position_scale']   # (Ntraj, Nres, 3)
    lig_pred_coords = struct_out['lig_T'][0, :, :, -3:] * config['position_scale']   # (Ntraj, Natoms, 3)
    rec_true_coords = batch['ground_truth']['gt_atom14_coords'][0, :, 1]   # (Nres, 3)
    rec_true_mask = batch['ground_truth']['gt_atom14_has_coords'][0, :, 1]   # (Nres)
    lig_true_coords = batch['ground_truth']['gt_lig_coords'][0]   #  (Nsymm, Natoms, 3)
    lig_true_mask = batch['ground_truth']['gt_lig_has_coords'][0]   #  (Nsymm, Natoms)

    num_traj = rec_pred_coords.shape[0]
    num_symm = lig_true_coords.shape[0]

    rec_rec_lddt = lddt.lddt(
        rec_pred_coords,
        rec_pred_coords,
        rec_true_coords[None],
        rec_true_coords[None],
        rec_true_mask[None, :, None],
        rec_true_mask[None, :, None],
        per_residue=True,
        exclude_self=True
    )   # (Ntraj, Nres)

    # compute lddt for each ligand symm and select the best symm id for each traj
    lig_rec_lddt_for_mask_selection = lddt.lddt(
        lig_pred_coords.repeat_interleave(num_symm, dim=0),
        rec_pred_coords.repeat_interleave(num_symm, dim=0),
        lig_true_coords.tile(num_traj, 1, 1),
        rec_true_coords[None].tile(num_traj * num_symm, 1, 1),
        lig_true_mask[:, :, None].tile(num_traj, 1, 1),
        rec_true_mask[None, :, None].tile(num_traj * num_symm, 1, 1),
        per_residue=False,
        exclude_self=False
    )  # (Ntraj * Nsymm)

    lig_best_mask_id_per_traj = lig_rec_lddt_for_mask_selection.reshape(num_traj, num_symm).max(-1).indices
    lig_best_mask_per_traj = lig_true_mask[lig_best_mask_id_per_traj]

    # compute lddt using the best mask for each trajectory frame
    lig_rec_lddt = lddt.lddt(
        lig_pred_coords,
        rec_pred_coords,
        lig_true_coords[lig_best_mask_id_per_traj],
        rec_true_coords[None].tile(num_traj, 1, 1),
        lig_best_mask_per_traj[:, :, None],
        rec_true_mask[None, :, None].tile(num_traj, 1, 1),
        per_residue=True,
        exclude_self=False
    )  # (Ntraj, Natoms)

    # TODO: decide on the proper range here
    # (Ntraj, Nres)
    rec_rec_lddt_label = torch.minimum(
        torch.div(rec_rec_lddt * 100, config['loss']['lddt_rec_bin_size'], rounding_mode='floor'),
        torch.full_like(rec_rec_lddt, config['loss']['lddt_rec_num_bins'] - 1)
    )

    # (Ntraj, Natoms)
    lig_rec_lddt_label = torch.minimum(
        torch.div(lig_rec_lddt * 100, config['loss']['lddt_lig_bin_size'], rounding_mode='floor'),
        torch.full_like(lig_rec_lddt, config['loss']['lddt_lig_num_bins'] - 1)
    )

    # set masked residues and atoms to -100
    rec_rec_lddt_label = rec_rec_lddt_label * rec_true_mask[None] + (rec_true_mask[None] - 1) * 100
    lig_rec_lddt_label = lig_rec_lddt_label * lig_best_mask_per_traj + (lig_best_mask_per_traj - 1) * 100

    # calc mean loss
    rec_rec_loss = F.cross_entropy(predicted_rec_lddt.flatten(end_dim=1), rec_rec_lddt_label.flatten().to(dtype=torch.long), ignore_index=-100, reduction='mean')
    lig_rec_loss = F.cross_entropy(predicted_lig_lddt.flatten(end_dim=1), lig_rec_lddt_label.flatten().to(dtype=torch.long), ignore_index=-100, reduction='mean')

    return {
        'rec_rec_lddt_loss': rec_rec_loss,
        'lig_rec_lddt_loss': lig_rec_loss,
        'rec_rec_lddt_true': rec_rec_lddt,
        'lig_rec_lddt_true': lig_rec_lddt,
        'lig_best_mask_per_traj': lig_best_mask_per_traj,
        'lig_best_mask_id_per_traj': lig_best_mask_id_per_traj
    }


def example3():
    from config import config, DATA_DIR
    with torch.no_grad():
        model = DockerIteration(config, config).cuda()

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)

        from dataset import DockingDataset
        ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
        #print(ds[0])
        item = ds[0]

        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None].cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

        #print(item['fragments']['rr_2d'])
        model(item)

        #print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in model(item).items()})


def example4():
    from config import config, DATA_DIR

    #with torch.autograd.set_detect_anomaly(True):
    model = DockerIteration(config, config).cuda()
    model.train()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num params:', pytorch_total_params)

    from dataset import DockingDataset
    ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
    item = ds[0]

    for k1, v1 in item.items():
        print(k1)
        for k2, v2 in v1.items():
            v1[k2] = torch.as_tensor(v2)[None].cuda()
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    #print(item['fragments']['rr_2d'])
    out = model(item)
    out['loss_total'].backward()

    #print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in model(item).items()})


if __name__ == '__main__':
    example4()
