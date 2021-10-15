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
        'num_iter': 4,
        'device': 'cuda:2'
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
            'device': 'cuda:1'
        }
    },
    'StructureModule': {
        'num_iter': 4,
        'device': 'cuda:2',
    }
}


'''config_diff = {
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
}'''


config_summit = utils.merge_dicts(deepcopy(config.config), config_diff)


log_dir = Path('train_log').mkdir_p()


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


def report(input, output, epoch, local_step, global_step, total_steps, dataset, global_stats, stage='Train'):
    loss = output['loss']['loss_total'].item()
    print(f'{stage} Epoch: {epoch} [{local_step}/{total_steps} ({100. * local_step / total_steps:.0f}%)]\tLoss: {loss:.6f}')

    #if hvd.rank() == 0 and local_step % 50 == 0:
    if local_step % 50 == 0 and HOROVOD_RANK == 0:
        ix = input['target']['ix'][0].item()
        case_name = dataset.data[ix]['case_name']
        group_name = dataset.data[ix]['group_name']
        pred_to_pdb((log_dir / 'pdbs').mkdir_p() / f'{global_step:06d}_{case_name}_{group_name}_{loss:4.3f}.pdb', input, output)

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

    for key, val in stats.items():
        if key not in global_stats:
            global_stats[key] = 0.0
        global_stats[key] += val

    for key, val in stats.items():
        writer.add_scalar(key + '/Step/' + stage, val, global_step)
        if local_step == total_steps - 1:
            writer.add_scalar(key + '/Epoch/' + stage, val / local_step, epoch)

    print(stats)
    if local_step == total_steps - 1:
        print(global_stats)


def train(epoch):
    model.train()
    dset = dataset.DockingDataset(
        config.DATA_DIR,
        'train_split/train_12k.json',
        max_hh_templates=6,
        max_frag_main=64,
        max_frag_extra=256,
        sample_to_size=5000,
        seed=epoch * 100
    )

    if HOROVOD:
        sampler = torch.utils.data.distributed.DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank())
        loader = torch.utils.data.DataLoader(dset, batch_size=1, sampler=sampler, **kwargs)
        sampler.set_epoch(epoch)
    else:
        loader = torch.utils.data.DataLoader(dset, batch_size=1, **kwargs)

    global_stats = {}

    for batch_idx, inputs in enumerate(loader):
        print(HOROVOD_RANK, ':', 'batch', batch_idx)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(inputs)
            loss = output['loss']['loss_total']
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step = batch_idx
        total_steps = len(loader)
        report(inputs, output, epoch, batch_idx, global_step, total_steps, dset, global_stats, stage='Train')
        sys.stdout.flush()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    torch.cuda.manual_seed(123456)
    logging.getLogger('.prody').setLevel('CRITICAL')

    kwargs = {'num_workers': 0, 'pin_memory': True}
    lr = 0.0001

    #valid_set = dataset.DockingDataset(config.DATA_DIR, 'train_split/debug.json', max_hh_templates=6, max_frag_main=128, max_frag_extra=512)
    #valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=hvd.size(), rank=hvd.rank())
    #valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, sampler=valid_sampler, **kwargs)

    model = docker.DockerIteration(config_summit, config_summit)

    lr_scaler = 1 if not HOROVOD else hvd.size()
    optimizer = optim.Adam(model.parameters(), lr=lr * lr_scaler) #, momentum=args.momentum)

    if HOROVOD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             op=hvd.Average)
                                            #gradient_predivide_factor=args.gradient_predivide_factor)

    writer = SummaryWriter(log_dir) #, purge_step=None)
    for epoch in range(1, 2):
        train(epoch)
        for x in range(3):
            print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
