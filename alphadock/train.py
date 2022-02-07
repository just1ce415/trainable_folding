import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import sys
from copy import deepcopy
from path import Path
import math
from tqdm import tqdm
import traceback

from alphadock import docker
from alphadock import config
from alphadock import dataset
from alphadock import all_atom
from alphadock import utils

import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


config_diff = {
    'Evoformer': {
        'num_iter': 4,
        'device': 'cuda:0'
    },
    'InputEmbedder': {
        'device': 'cuda:0',
        'TemplatePairStack': {
            'num_iter': 2,
            'device': 'cuda:0'
        },
        'TemplatePointwiseAttention': {
            'device': 'cuda:0',
            'attention_num_c': 64,
            'num_heads': 4
        },
        'FragExtraStack': {
            'num_iter': 4,
            'device': 'cuda:0'
        }
    },
    'StructureModule': {
        'num_iter': 8,
        'device': 'cuda:0',
    },
    'loss': {
        'loss_violation_weight': 0.0
    }
}


if False:
    config_diff = {
        'Evoformer': {
            'num_iter': 1,
            'device': 'cuda:0'
        },
        'InputEmbedder': {
            'device': 'cuda:0',
            'TemplatePairStack': {
                'num_iter': 2,
                'checkpoint': True,
                'device': 'cuda:0'
            },
            'TemplatePointwiseAttention': {
                'device': 'cuda:0',
                'attention_num_c': 64,
                'num_heads': 4
            },
            'FragExtraStack': {
                'num_iter': 4,
                'device': 'cuda:0'
            }
        },
        'StructureModule': {
            'num_iter': 2,
            'device': 'cuda:0',
        }
    }


config_summit = utils.merge_dicts(deepcopy(config.config), config_diff)
log_dir = Path('.').mkdir_p()
train_json = 'train_split/train_lig_sampled.json'
valid_json = 'train_split/valid_lig_sampled.json'
pdb_log_interval = 100
global_step = 0

LEARNING_RATE = 0.001 / 128  # same as AF, lr=0.001 for 128 batch size
SCHEDULER_PATIENCE = 20
SCHEDULER_FACTOR = 1. / 3
SCHEDULER_MIN_LR = 1e-6 / 128  # same as AF
CLIP_GRADIENT = True
CLIP_GRADIENT_VALUE = 0.1
USE_AMP = True
USE_AMP_SCALER = False

HOROVOD = True
HOROVOD_RANK = 0
if HOROVOD:
    import horovod.torch as hvd
    if __name__ == '__main__':
        hvd.init()
        HOROVOD_RANK = hvd.rank()


def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
        f.write(f'HEADER {out_pdb.basename().stripext()}.pred\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['target']['rec_aatype'][0].cpu(),
            out_dict['final_all_atom']['atom_pos_tensor'].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )
        all_atom.ligand_to_pdb_stream(
            f,
            input_dict['target']['lig_atom_types'][0].cpu(),
            out_dict['struct_out']['lig_T'][0, -1, :, -3:].detach().cpu(),
            resname='LIG',
            resnum=1,
            chain='B',
            serial_start=serial
        )
        f.write(f'HEADER {out_pdb.basename().stripext()}.crys\n')
        serial = all_atom.atom14_to_pdb_stream(
            f,
            input_dict['ground_truth']['gt_aatype'][0].cpu(),
            input_dict['ground_truth']['gt_atom14_coords'][0].detach().cpu(),
            chain='A',
            serial_start=1,
            resnum_start=1
        )
        all_atom.ligand_to_pdb_stream(
            f,
            input_dict['target']['lig_atom_types'][0].cpu(),
            input_dict['ground_truth']['gt_lig_coords'][0, 0, :, :].detach().cpu(),
            resname='LIG',
            resnum=1,
            chain='B',
            serial_start=serial
        )
        f.write(f'HEADER {out_pdb.basename().stripext()}.init\n')
        all_atom.ligand_to_pdb_stream(
            f,
            input_dict['target']['lig_atom_types'][0].cpu(),
            input_dict['target']['lig_init_coords'][0, :, :].detach().cpu(),
            resname='LIG',
            resnum=1,
            chain='B'
        )


def report_step(input, output, epoch, local_step, dataset, global_stats, train=True):
    stage = 'Train' if train else 'Valid'
    stats = {}

    if len(output) > 0:
        loss = output['loss']['loss_total'].item()
        stats['Loss_Total'] = output['loss']['loss_total'].item()
        stats['LDDT_Rec_Final'] = output['loss']['lddt_values']['rec_rec_lddt_true_total'][-1].item()
        stats['LDDT_Lig_Final'] = output['loss']['lddt_values']['lig_rec_lddt_true_total'][-1].item()
        stats['LDDT_Rec_MeanTraj'] = output['loss']['lddt_values']['rec_rec_lddt_true_total'].mean().item()
        stats['LDDT_Lig_MeanTraj'] = output['loss']['lddt_values']['lig_rec_lddt_true_total'].mean().item()
        stats['Loss_LDDT_Rec'] = output['loss']['lddt_loss_rec_rec'].item()
        stats['Loss_LDDT_Lig'] = output['loss']['lddt_loss_lig_rec'].item()
        stats['Loss_Torsions'] = output['loss']['loss_torsions']['chi_loss'].mean().item()
        stats['Loss_Norm'] = output['loss']['loss_torsions']['norm_loss'].mean().item()
        stats['Loss_FAPE_BB_Rec_Rec_Final'] = output['loss']['loss_fape']['loss_bb_rec_rec'][-1].item()
        stats['Loss_FAPE_BB_Rec_Lig_Final'] = output['loss']['loss_fape']['loss_bb_rec_lig'][-1, :].min().item()
        stats['Loss_FAPE_AA_Rec_Rec_Final'] = output['loss']['loss_fape']['loss_aa_rec_rec'].item()
        stats['Loss_FAPE_AA_Rec_Lig_Final'] = output['loss']['loss_fape']['loss_aa_rec_lig'].min().item()
        stats['Loss_FAPE_BB_Rec_Rec_MeanTraj'] = output['loss']['loss_fape']['loss_bb_rec_rec'].mean().item()
        stats['Loss_FAPE_BB_Rec_Lig_MeanTraj'] = output['loss']['loss_fape']['loss_bb_rec_lig'].min(-1).values.mean().item()
        stats['Loss_LigDmat_MeanTraj'] = output['loss']['loss_lig_dmat'].mean().item()
        stats['Loss_PredDmat_RecRec'] = output['loss']['loss_pred_dmat']['rr'].item()
        stats['Loss_PredDmat_LigLig'] = output['loss']['loss_pred_dmat']['ll'].item()
        stats['Loss_PredDmat_RecLig'] = output['loss']['loss_pred_dmat']['rl'].item()

        if 'lig_init_rec_lddt_true_total' in output['loss']['lddt_values']:
            stats['LDDT_Lig_Init'] = output['loss']['lddt_values']['lig_init_rec_lddt_true_total'].min().item()
            stats['LDDT_Lig_Pred_Minus_Init'] = stats['LDDT_Lig_Final'] - stats['LDDT_Lig_Init']
            stats['LDDT_Lig_Improved'] = 1 if stats['LDDT_Lig_Final'] > stats['LDDT_Lig_Init'] else 0

        if 'loss_affinity' in output['loss']:
            stats['Loss_Affinity'] = output['loss']['loss_affinity'].item()
            aff_lig_id = input['ground_truth']['gt_affinity_lig_id'][0].item()
            aff_probs = F.softmax(output['struct_out']['lig_affinity'].detach()[0, aff_lig_id])  # (6)
            aff_label_true = input['ground_truth']['gt_affinity_label'][0].item()
            aff_label_pred = torch.argmax(aff_probs).item()
            #aff_probs = F.softmax(aff_pred)
            stats[f'Affinity/Label_{aff_label_true}/TP'] = int(aff_label_pred == aff_label_true)
            if HOROVOD_RANK == 0:
                print('Affinity preds', aff_probs)
                sys.stdout.flush()
            for i in range(aff_probs.shape[0]):
                stats[f'Affinity/Label_{i}/Prediction'] = int(i == aff_label_pred)
            #stats[f'Affinity/Label{aff_label}/Count'] = 1

        if 'violations' in output['loss']:
            viol = output['loss']['violations']
            stats['Violations/Loss'] = viol['loss'].item()
            stats['Violations/Extreme_CA_CA'] = viol['between_residues']['violations_extreme_ca_ca'].item()

            stats['Violations/Inter_ResRes_Bonds'] = viol['between_residues']['connections_per_residue_violation_mask'].mean().item()
            stats['Violations/Inter_ResRes_Clash'] = viol['between_residues']['clashes_per_atom_clash_mask'].max(-1).values.mean().item()
            stats['Violations/Intra_Residue_Violations'] = viol['within_residues']['per_atom_violations'].max(-1).values.mean().item()
            stats['Violations/Total_Residue_Violations'] = viol['total_per_residue_violations_mask'].mean().item()

            stats['Violations/Inter_LigRes_Clash'] = viol['lig_rec']['clashes_lig_per_atom_clash_mask'].mean().item()
            stats['Violations/Inter_ResLig_Clash'] = viol['lig_rec']['clashes_rec_per_atom_clash_mask'].max(-1).values.mean().item()
            stats['Violations/Intra_Ligand_Violations'] = viol['lig']['per_atom_violations'].mean().item()
            stats['Violations/Total_Ligand_Violations'] = viol['total_per_lig_atom_violations_mask'].mean().item()

            num_rec_atoms = torch.sum(input['target']['rec_atom14_atom_exists'][0]).item()
            stats['Violations/between_bonds_c_n_mean_loss'] = viol['between_residues']['bonds_c_n_loss_mean'].item()
            stats['Violations/between_angles_ca_c_n_mean_loss'] = viol['between_residues']['angles_ca_c_n_loss_mean'].item()
            stats['Violations/between_angles_c_n_ca_mean_loss'] = viol['between_residues']['angles_c_n_ca_loss_mean'].item()
            stats['Violations/between_clashes_mean_loss'] = viol['between_residues']['clashes_per_atom_loss_sum'].sum().item() / (1e-6 + num_rec_atoms)
            stats['Violations/within_mean_loss'] = viol['within_residues']['per_atom_loss_sum'].sum().item() / (1e-6 + num_rec_atoms)
            stats['Violations/lig_rec_clashes_mean_loss'] = viol['lig_rec']['clashes_mean_loss'].mean().item()
            stats['Violations/lig_lig_mean_loss'] = viol['lig']['per_atom_loss_sum'].mean().item()

        if HOROVOD_RANK == 0:
            print('rec LDDT true')
            print(output['loss']['lddt_values']['rec_rec_lddt_true_per_residue'][-1])
            print('rec LDDT pred')
            print(output['struct_out']['rec_lddt'][0, -1])
            print('lig LDDT true')
            print(output['loss']['lddt_values']['lig_rec_lddt_true_per_atom'][-1])
            print('lig LDDT pred')
            print(output['struct_out']['lig_lddt'][0, -1])

        for k, v in stats.items():
            any_nan = 0
            if math.isnan(v):
                any_nan += 1
                print(f'Process {HOROVOD_RANK}: {k} is nan')
            #if any_nan > 0:
            #    print(output)

        if (not train) or (pdb_log_interval is not None and ((global_step + HOROVOD_RANK) % pdb_log_interval == 0)):
            ix = input['target']['ix'][0].item()
            case_name = dataset.data[ix]['case_name']
            group_name = dataset.data[ix]['group_name']
            if train:
                file_name = f'train_epoch_{epoch}_{case_name}_{group_name}_{loss:.3f}.pdb'
            else:
                file_name = f'valid_epoch_{epoch}_{case_name}_{group_name}_{loss:.3f}.pdb'
            pred_to_pdb((log_dir / 'pdbs').mkdir_p() / file_name, input, output)
            stats_dump = stats.copy()
            stats_dump['Used_HH_templates'] = 'hhpred' in input
            stats_dump['Used_frag_templates'] = 'fragments' in input
            stats_dump['HH_lig_coverage'] = None
            if 'hhpred' in input:
                stats_dump['HH_lig_coverage'] = (input['hhpred']['lig_1d'][0, :, :, -1].sum(-1) / input['hhpred']['lig_1d'].shape[2]).detach().numpy().tolist()
            utils.write_json(stats_dump, (log_dir / 'pdbs' / file_name).stripext() + '.json')

    if HOROVOD:
        print(HOROVOD_RANK, ':', 'gathering')
        all_stats = hvd.allgather_object(stats)
        print(HOROVOD_RANK, ':', 'gathered')
    else:
        all_stats = [stats]

    if HOROVOD_RANK == 0:
        for idx, case_stats in enumerate(all_stats):
            for key, val in case_stats.items():
                if key not in global_stats:
                    global_stats[key] = []
                global_stats[key].append(val)

                if train:
                    writer.add_scalar(key + '/Step/' + stage, case_stats[key], global_step + idx)

        for x in range(torch.cuda.device_count()):
            print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
        print('global step', global_step)
        for k, v in stats.items():
            print(k, ":", v)
        sys.stdout.flush()

    return all_stats


def report_epoch_end(epoch, global_stats, stage='Train', save_model=True):
    global global_step

    if HOROVOD_RANK == 0:
        writer.add_scalar('HasNans/Epoch/' + stage, math.isnan(sum(global_stats['Loss_Total'])), epoch)
        for key in global_stats.keys():
            vals = [x for x in global_stats[key] if not math.isnan(x)]
            global_stats[key] = sum(vals) / len(vals) if len(vals) > 0 else math.nan
            writer.add_scalar(key + '/Epoch/' + stage, global_stats[key], epoch)
        writer.add_scalar('LearningRate/Epoch/' + stage, optimizer.param_groups[0]['lr'], epoch)
        print('Epoch_stats', global_stats)

    if HOROVOD:
        global_stats = hvd.broadcast_object(global_stats, root_rank=0)

    scheduler.step(global_stats['Loss_Total'], epoch=epoch)

    if HOROVOD_RANK == 0 and save_model:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': global_stats,
            'global_step': global_step,
        }, log_dir / f'epoch_{epoch}_loss_{global_stats["Loss_Total"]:.3f}.pth')


def check_grads():
    grads_are_nan = sum([torch.isnan(x.grad).any().item() for x in model.parameters() if x.grad is not None])
    if grads_are_nan > 0:
        raise utils.GeneratedNans(f'Process {HOROVOD_RANK}: gradients are nan')


def validate(epoch):
    model.eval()
    dset = dataset.DockingDataset(
        config.DATA_DIR,
        valid_json,
        max_hh_templates=0,
        max_frag_main=0,
        max_frag_extra=0,
        clamp_fape_prob=0,
        max_num_res=None,
        seed=123456
    )
    #dset.data = dset.data[:7]
    #dset = dataset.DockingDatasetSimulated(size=4, num_frag_main=64, num_frag_extra=256, num_res=350, num_hh=4)

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, shuffle=False, **kwargs)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, **kwargs)

    global_stats = {}
    local_step = 0
    num_recycles = config_summit['recycling_num_iter'] if config_summit['recycling_on'] else 1

    for inputs in (tqdm(loader, desc=f'Epoch {epoch} (valid)') if HOROVOD_RANK == 0 else loader):
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(USE_AMP):
                    for recycle_iter in range(num_recycles):
                        output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)

        except:
            print(HOROVOD_RANK, ':', 'Exception in validation')
            traceback.print_exc()
            sys.stdout.flush()
            output = {}

        step_stats = report_step(inputs, output, epoch, local_step, dset, global_stats, train=False)
        local_step += 1
        sys.stdout.flush()
        torch.cuda.empty_cache()

    report_epoch_end(epoch, global_stats, stage='Valid', save_model=False)


def train(epoch):
    model.train()
    dset = dataset.DockingDataset(
        config.DATA_DIR,
        train_json,
        max_hh_templates=0,
        max_frag_main=0,
        max_frag_extra=0,
        sample_to_size=3000,
        seed=epoch * 101
    )
    #dset = dataset.DockingDatasetSimulated(size=4, num_frag_main=64, num_frag_extra=256, num_res=400, num_hh=6)
    #dset.data = dset.data[30 * 6:]

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, shuffle=False, **kwargs)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, **kwargs)

    global_stats = {}
    local_step = 0
    global global_step

    # number of recycling iterations
    recycling_on = config_summit['recycling_on']
    num_recycles = config_summit['recycling_num_iter'] if recycling_on else 1

    for inputs in (tqdm(loader, desc=f'Epoch {epoch} (train)') if HOROVOD_RANK == 0 else loader):
        optimizer.zero_grad()

        if False:
            for k1, v1 in inputs.items():
                print(k1)
                for k2, v2 in v1.items():
                    print('    ', k2, v1[k2].shape, v1[k2].dtype)

        # sync recycling iteration for which the grad will be computed
        recycle_iter_grad_on = torch.randint(0, num_recycles, [1])[0].item() if HOROVOD_RANK == 0 and recycling_on else 0
        if HOROVOD:
            #print(HOROVOD_RANK, "before :", recycle_iter_grad_on); sys.stdout.flush()
            recycle_iter_grad_on = hvd.broadcast_object(recycle_iter_grad_on, root_rank=0)
            #print(HOROVOD_RANK, "done :", recycle_iter_grad_on); sys.stdout.flush()

        try:
            print(HOROVOD_RANK, ": sample id - ", inputs['target']['ix'])
            losses = []

            with torch.cuda.amp.autocast(USE_AMP):
                for recycle_iter in range(num_recycles):
                    with torch.set_grad_enabled((recycle_iter == recycle_iter_grad_on) or not recycling_on):
                        output = model(inputs, recycling=output['recycling_input'] if recycle_iter > 0 else None)
                    losses.append(output['loss']['loss_total'])

                    if HOROVOD_RANK == 0:
                        print(HOROVOD_RANK, ':', f'loss[{recycle_iter}]', losses[-1].item()); sys.stdout.flush()

            # calculate grads for the recycling iteration
            if USE_AMP and USE_AMP_SCALER:
                amp_scaler.scale(losses[recycle_iter_grad_on]).backward()
            else:
                losses[recycle_iter_grad_on].backward()

            # check that grads are not nan, throw GeneratedNans if yes
            check_grads()

        except RuntimeError:
            # this is for CUDA out of memory error, if encountered we will just move to the next sample
            print(HOROVOD_RANK, ':', 'Exception in training', 'sample id:', inputs['target']['ix'])
            traceback.print_exc()
            sys.stdout.flush()
            output = {}

        if HOROVOD:
            # following pytorch example from horovod docs
            optimizer.synchronize()
            print(HOROVOD_RANK, ':', 'optimizer.synchronize()'); sys.stdout.flush()

        if USE_AMP and USE_AMP_SCALER and CLIP_GRADIENT:
            amp_scaler.unscale_(optimizer)
        print(HOROVOD_RANK, ':', 'amp_scaler.unscale_(optimizer)'); sys.stdout.flush()
        if CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT_VALUE)
        print(HOROVOD_RANK, ':', 'torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT_VALUE)'); sys.stdout.flush()

        if HOROVOD:
            with optimizer.skip_synchronize():
                if USE_AMP and USE_AMP_SCALER:
                    amp_scaler.step(optimizer)
                else:
                    optimizer.step()
        else:
            if USE_AMP and USE_AMP_SCALER:
                amp_scaler.step(optimizer)
            else:
                optimizer.step()
        print(HOROVOD_RANK, ':', 'optimizer.step()'); sys.stdout.flush()

        if USE_AMP and USE_AMP_SCALER:
            amp_scaler.update()
        print(HOROVOD_RANK, ':', 'amp_scaler.update()'); sys.stdout.flush()

        step_stats = report_step(inputs, output, epoch, local_step, dset, global_stats, train=True)
        global_step += len(step_stats)
        local_step += 1
        sys.stdout.flush()
        torch.cuda.empty_cache()

    report_epoch_end(epoch, global_stats, stage='Train', save_model=True)


def find_last_pth(dir):
    pths = Path(dir).glob('epoch_*.pth')
    if len(pths) == 0:
        return None
    return sorted(pths, key=lambda x: -int(x.basename().split('_')[1]))[0]


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    logging.getLogger('.prody').setLevel('CRITICAL')

    kwargs = {'num_workers': 0, 'pin_memory': True}
    model = docker.DockerIteration(config_summit, config_summit)

    if HOROVOD_RANK == 0:
        print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        print('Num param sets:', len([p for p in model.parameters() if p.requires_grad]))
        for x in range(torch.cuda.device_count()):
            print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
        sys.stdout.flush()

    lr_scaler = 1 if not HOROVOD else hvd.size()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * lr_scaler) #, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    #optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler, momentum=0.9, nesterov=True)

    start_epoch = 1
    scheduler_state = scheduler.state_dict()
    pth_file = find_last_pth(log_dir)
    if pth_file is not None and HOROVOD_RANK == 0:
        pth = torch.load(pth_file)
        print('Loading saved model from', pth_file)
        global_step = pth['global_step']
        start_epoch = pth['epoch'] + 1
        model.load_state_dict(pth['model_state_dict'])
        optimizer.load_state_dict(pth['optimizer_state_dict'])
        scheduler_state = pth['scheduler_state_dict']

        #if start_epoch != 64:
        #    scheduler_state = pth['scheduler_state_dict']

        #if start_epoch == 64:
        #    for g in optimizer.param_groups:
        #        g['lr'] = g['lr'] * SCHEDULER_FACTOR

    if HOROVOD:
        global_step = hvd.broadcast_object(global_step, root_rank=0)
        start_epoch = hvd.broadcast_object(start_epoch, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        scheduler_state = hvd.broadcast_object(scheduler_state, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)

    scheduler.load_state_dict(scheduler_state)

    if USE_AMP and USE_AMP_SCALER:
        amp_scaler = torch.cuda.amp.GradScaler()

    if HOROVOD_RANK == 0:
        writer = SummaryWriter(log_dir)

    for epoch in range(start_epoch, start_epoch + 100):
        with torch.autograd.set_detect_anomaly(True):
            train(epoch)
            validate(epoch)
