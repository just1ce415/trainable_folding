import torch
import torch.optim as optim
import logging
import sys
from copy import deepcopy
from path import Path

from alphadock import docker
from alphadock import config
from alphadock import dataset
from alphadock import all_atom
from alphadock import utils

import torchvision
from torch.utils.tensorboard import SummaryWriter


config_diff = {
    'Evoformer': {
        'num_iter': 16,
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
        'num_iter': 4,
        'device': 'cuda:0',
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
train_json = 'train_split/train_12k.json'
valid_json = 'train_split/valid_12k.json'
pdb_log_interval = 1
global_step = 0

HOROVOD = True
HOROVOD_RANK = 0
if HOROVOD:
    import horovod.torch as hvd
    if __name__ == '__main__':
        hvd.init()
        HOROVOD_RANK = hvd.rank()


def pred_to_pdb(out_pdb, input_dict, out_dict):
    with open(out_pdb, 'w') as f:
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


def horovod_average_stats(stats_list):
    stats = {}
    for stats_instance in stats_list:
        for k, v in stats_instance.items():
            if k not in stats:
                stats[k] = []
            stats[k].append(v)
    for k in stats.keys():
        stats[k] = sum(stats[k]) / len(stats[k]) if len(stats[k]) > 0 else 0.
    return stats


def report_step(input, output, epoch, local_step, total_steps, dataset, global_stats, train=True):
    stage = 'Train' if train else 'Valid'
    loss = output['loss']['loss_total'].item()
    if HOROVOD_RANK == 0:
        print(f'{stage} Epoch: {epoch} [{local_step}/{total_steps} ({100. * local_step / total_steps:.0f}%)]\tLoss: {loss:.6f}')

    stats = {}
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
    if 'loss_affinity' in output['loss']:
        stats['Loss_Affinity'] = output['loss']['loss_affinity'].item()

    if (train and local_step % pdb_log_interval == 0 and HOROVOD_RANK == 0) or not train:
        ix = input['target']['ix'][0].item()
        case_name = dataset.data[ix]['case_name']
        group_name = dataset.data[ix]['group_name']
        if train:
            file_name = f'train_step_{global_step:06d}_{case_name}_{group_name}_{loss:.3f}.pdb'
        else:
            file_name = f'valid_epoch_{epoch}_{case_name}_{group_name}_{loss:.3f}.pdb'
        pred_to_pdb((log_dir / 'pdbs').mkdir_p() / file_name, input, output)
        utils.write_json(stats, (log_dir / 'pdbs' / file_name).stripext() + '.json')

    if HOROVOD:
        stats_gathered = hvd.allgather_object(stats)
        if HOROVOD_RANK == 0:
            stats = horovod_average_stats(stats_gathered)

    if HOROVOD_RANK == 0:
        for x in range(torch.cuda.device_count()):
            print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
        print('global step', global_step)
        print(stats)
        sys.stdout.flush()

        for key, val in stats.items():
            if key not in global_stats:
                global_stats[key] = 0.0
            global_stats[key] += val

        for key in stats.keys():
            if train:
                writer.add_scalar(key + '/Step/' + stage, stats[key], global_step)

    #stats['step_size'] = 1
    #if HOROVOD:
    #    stats['step_size'] = len(stats_gathered)

    return stats


def report_epoch_end(epoch, global_stats, total_steps, train=True):
    if HOROVOD_RANK == 0:
        stage = 'Train' if train else 'Valid'
        for key in global_stats.keys():
            writer.add_scalar(key + '/Epoch/' + stage, global_stats[key] / total_steps, epoch)
        global global_step
        if train:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': global_stats,
                'global_step': global_step,
            }, log_dir / f'{stage.lower()}_epoch_{epoch}_loss_{global_stats["Loss_Total"]:.3f}.pth')
        print('Epoch_stats', global_stats)


def validate(epoch):
    model.eval()
    dset = dataset.DockingDataset(
        config.DATA_DIR,
        valid_json,
        max_hh_templates=4,
        max_frag_main=64,
        max_frag_extra=256,
        clamp_fape_prob=0,
        max_num_res=350,
        seed=123456
    )
    #dset.data = dset.data[:7]
    #dset = dataset.DockingDatasetSimulated(size=4, num_frag_main=64, num_frag_extra=256, num_res=350, num_hh=4)

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank())
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, **kwargs)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, **kwargs)

    global_stats = {}
    local_step = 0
    for inputs in loader:
        print(HOROVOD_RANK, ':', 'batch', local_step)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(inputs)

        total_steps = len(loader)
        report_step(inputs, output, epoch, local_step, total_steps, dset, global_stats, train=False)
        sys.stdout.flush()
        local_step += 1

    report_epoch_end(epoch, global_stats, local_step + 1, train=False)


def train(epoch):
    model.train()
    dset = dataset.DockingDataset(
        config.DATA_DIR,
        train_json,
        max_hh_templates=6,
        max_frag_main=64,
        max_frag_extra=256,
        sample_to_size=5000,
        seed=epoch * 100
    )
    #dset = dataset.DockingDatasetSimulated(size=4, num_frag_main=64, num_frag_extra=256, num_res=400, num_hh=6)

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank())
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, **kwargs)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, **kwargs)

    global_stats = {}
    local_step = 0
    global global_step
    for inputs in loader:
        print(HOROVOD_RANK, ':', 'batch', local_step)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss = output['loss']['loss_total']
            amp_scaler.scale(loss).backward()

        if HOROVOD:
            optimizer.synchronize()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            with optimizer.skip_synchronize():
                amp_scaler.step(optimizer)
        else:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)

        # Update scaler in case of overflow/underflow
        amp_scaler.update()

        total_steps = len(loader)
        report_step(inputs, output, epoch, local_step, total_steps, dset, global_stats, train=True)
        sys.stdout.flush()
        local_step += 1
        global_step += 1

    report_epoch_end(epoch, global_stats, local_step + 1, train=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    torch.cuda.manual_seed(123456)
    logging.getLogger('.prody').setLevel('CRITICAL')
    print(HOROVOD_RANK, ': 1')
    sys.stdout.flush()

    kwargs = {'num_workers': 0, 'pin_memory': True}
    lr = 0.0001
    model = docker.DockerIteration(config_summit, config_summit)
    print(HOROVOD_RANK, ': 2')
    print('Num params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Num param sets:', len([p for p in model.parameters() if p.requires_grad]))
    for x in range(torch.cuda.device_count()):
        print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
    sys.stdout.flush()

    lr_scaler = 1 #if not HOROVOD else hvd.size()
    optimizer = optim.Adam(model.parameters(), lr=lr * lr_scaler) #, momentum=args.momentum)
    #optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler, momentum=0.9, nesterov=True)
    print(HOROVOD_RANK, ': 3')
    sys.stdout.flush()

    if HOROVOD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        print(HOROVOD_RANK, ': 4')
        sys.stdout.flush()

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        print(HOROVOD_RANK, ': 5')
        sys.stdout.flush()

        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Average)
        print(HOROVOD_RANK, ': 6')
        sys.stdout.flush()

    amp_scaler = torch.cuda.amp.GradScaler()

    if HOROVOD_RANK == 0:
        writer = SummaryWriter(log_dir) #, purge_step=None)

    for epoch in range(1, 100):
        train(epoch)
        validate(epoch)
