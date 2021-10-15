import torch
import torch.nn as nn
import torch.optim as optim
#import horovod.torch as hvd
import logging
from torch.utils.checkpoint import checkpoint
import sys

from alphadock import docker
from alphadock import config
from alphadock import dataset
from alphadock import modules
from alphadock import all_atom
from alphadock import loss
from alphadock import quat_affine


config_summit = {
    'rec_in_c': 39,
    'lig_in_c': 47,
    'extra_in_c': 238,

    'lig_in2d_c': 6,
    'rec_in2d_c': 40,
    'rec_relpos_c': 65,

    'frag_rec': 23,
    'frag_lig': 48,
    'frag_rr': 82,
    'frag_ll': 140,
    'frag_rl': 108,
    'frag_lr': 108,

    'hh_rec': 24,
    'hh_lig': 49,
    'hh_rr': 84,
    'hh_ll': 142,
    'hh_rl': 110,
    'hh_lr': 110,

    'position_scale': 10,
    'num_torsions': 7,
    'rep_1d': {
        'num_c': 64
    },
    'rep_2d': {
        'num_c': 64
    },
    'rec_dist_num_bins': 40,
    'num_single_c': 384,

    'loss': {
        'fape_loss_unit_distance': 10.0,
        'fape_clamp_distance': 10.0,
        'loss_bb_rec_rec_weight': 0.5 * 0.25,
        'loss_bb_rec_lig_weight': 0.5 * 0.25,
        'loss_aa_rec_rec_weight': 0.5 * 0.25,
        'loss_aa_rec_lig_weight': 0.5 * 0.25,
        'loss_chi_value_weight': 0.5 * 0.5,
        'loss_chi_norm_weight': 0.5 * 0.5,
        'loss_rec_rec_lddt_weight': 0.01 * 0.5,
        'loss_lig_rec_lddt_weight': 0.01 * 0.5,
        'loss_affinity_weight': 0.01,
        'lddt_rec_bin_size': 2,
        'lddt_rec_num_bins': 50,
        'lddt_lig_bin_size': 2,
        'lddt_lig_num_bins': 50,
    },

    'Evoformer': {
        'num_iter': 2,
        'device': 'cuda:0',
        'EvoformerIteration': {
            'checkpoint': True,
            'RowAttentionWithPairBias': {
                'attention_num_c': 32,
                'num_heads': 8
            },
            'LigColumnAttention': {
                'attention_num_c': 32,
                'num_heads': 8
            },
            'LigTransition': {
                'n': 4
            },
            'RecTransition': {
                'n': 4
            },
            'OuterProductMean': {
                'mid_c': 32
            },
            'TriangleMultiplicationIngoing': {
                'mid_c': 128,
            },
            'TriangleMultiplicationOutgoing': {
                'mid_c': 128,
            },
            'TriangleAttentionStartingNode': {
                'attention_num_c': 32,
                'num_heads': 4,
                'rand_remove': 0.25
            },
            'TriangleAttentionEndingNode': {
                'attention_num_c': 32,
                'num_heads': 4,
                'rand_remove': 0.25
            },
            'PairTransition': {
                'n': 4
            }
        }
    },
    'InputEmbedder': {
        'device': 'cuda:0',
        'TemplatePairStack': {
            'num_iter': 2,
            'checkpoint': True,
            'device': 'cuda:0',
            'TemplatePairStackIteration': {
                #'checkpoint': True,
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.25
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.25
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 64
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 64
                },
                'PairTransition': {
                    'n': 2
                }
            }
        },
        'TemplatePointwiseAttention': {
            'device': 'cuda:0',
            'attention_num_c': 64,
            'num_heads': 4
        },
        'FragExtraStack': {
            'num_iter': 4,
            'device': 'cuda:0',
            'FragExtraStackIteration': {
                'checkpoint': True,
                'RowAttentionWithPairBias': {
                    'attention_num_c': 8,
                    'num_heads': 4
                },
                'ExtraColumnGlobalAttention': {
                    'attention_num_c': 8,
                    'num_heads': 8
                },
                'LigTransition': {
                    'n': 4
                },
                'RecTransition': {
                    'n': 4
                },
                'OuterProductMean': {
                    'mid_c': 32
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 128,
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 128,
                },
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.25
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4,
                    'rand_remove': 0.25
                },
                'PairTransition': {
                    'n': 4
                }
            }
        }
    },
    'StructureModule': {
        'num_iter': 1,
        'device': 'cuda:0',
        'StructureModuleIteration': {
            'checkpoint': False,
            'InvariantPointAttention': {
                'num_head': 8,
                'num_scalar_qk': 16,
                'num_point_qk': 4,
                'num_2d_qk': 16,
                'num_scalar_v': 16,
                'num_point_v': 8,
                'num_2d_v': 16
            },
            'PredictSidechains': {
                'num_c': 128
            },
            'PredictRecLDDT': {
                'num_c': 128,
                'num_bins': 50
            },
            'PredictLigLDDT': {
                'num_c': 128,
                'num_bins': 50
            }
        },
        'PredictAffinity': {
            'num_c': 128,
            'num_bins': 6
        }
    }
}


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda:0'
        self.l1 = nn.Linear(7, 7, bias=False).to(self.device)
        self.l2 = nn.Linear(2, 2, bias=False).to(self.device)
        self.l3 = nn.Linear(7, 7, bias=False).to(self.device)

    def forward(self, inputs):
        inputs = {k: {k1: v1.to(self.device) for k1, v1 in v.items()} for k, v in inputs.items()}
        print(list(self.l1.parameters()))

        rec_T = inputs['ground_truth']['gt_bb_affine'][None].clone() # (1, 1, N, 7)
        rec_T[:, :, inputs['ground_truth']['gt_bb_affine_mask'][0] < 1., 0] = 1
        rec_T[:, :, inputs['ground_truth']['gt_bb_affine_mask'][0] < 1., 1:] = 0

        lig_coords = inputs['ground_truth']['gt_lig_coords'][0, 0]  # (N, 3)
        lig_T = torch.zeros_like(rec_T)[:, :, :lig_coords.shape[0], :]
        lig_T[..., -3:] = lig_coords[None, None]
        lig_T[..., 0] = 1  # (1, 1, N, 7)
        #print('input')
        #print(rec_T)
        #print(lig_T)

        rec_T = self.l1(rec_T)
        lig_T = self.l3(lig_T)
        struct_out = {
            'rec_T': rec_T,
            'lig_T': lig_T,
            'rec_torsions': inputs['ground_truth']['gt_torsions_sin_cos'][None],
            #'rec_lddt': torch.zeros_like(rec_T)[:, :, :inputs['ground_truth']['gt_lig_coords'].shape[1], :],
            #'lig_lddt': torch.stack(lig_lddt, dim=1),
        }
        #print('mlp')
        #print(struct_out['rec_T'])
        #print(struct_out['lig_T'])

        struct_out['rec_T'] = quat_affine.QuatAffine.from_tensor(struct_out['rec_T'], normalize=True).to_tensor()
        struct_out['lig_T'] = quat_affine.QuatAffine.from_tensor(struct_out['lig_T'], normalize=True).to_tensor()

        #print('normalized')
        #print(struct_out['rec_T'])
        #print(struct_out['lig_T'])

        assert struct_out['rec_T'].shape[0] == 1
        final_all_atom = all_atom.backbone_affine_and_torsions_to_all_atom(
            struct_out['rec_T'][0, -1].clone(),
            struct_out['rec_torsions'][0][-1],
            inputs['target']['rec_aatype'][0]
        )

        #print()
        out_dict = {}
        out_dict['loss'] = loss.total_loss(inputs, struct_out, final_all_atom, config_summit)
        out_dict['final_all_atom'] = final_all_atom
        out_dict['struct_out'] = struct_out
        return out_dict


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


def train(epoch):
    model.train()

    train_set = dataset.DockingDataset(
        config.DATA_DIR,
        'train_split/train_12k.json',
        #sample_to_size=10000,
        seed=epoch * 100
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, **kwargs)

    for batch_idx, inputs in enumerate(train_loader):
        print('batch', batch_idx)
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        output = model(inputs)
        loss = output['loss']['loss_total']
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(output['loss'])
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss.item()))
        print('lddt pred', torch.max(output['struct_out']['lig_lddt'][0][0], dim=-1).indices * 2)

        #if hvd.rank() == 0:
        if batch_idx % 50 == 0:
            ix = inputs['target']['ix'][0].item()
            case_name = train_set.data[ix]['case_name']
            group_name = train_set.data[ix]['group_name']
            pred_to_pdb((config.DATA_DIR / 'tmp').mkdir_p() / f'{batch_idx:05d}_{loss:4.3f}.pdb', inputs, output)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(123456)
    torch.cuda.manual_seed(123456)
    logging.getLogger('.prody').setLevel('CRITICAL')

    kwargs = {'num_workers': 0, 'pin_memory': True}
    lr = 0.0001

    model = docker.DockerIteration(config_summit, config_summit)
    #model = Toy()
    optimizer = optim.Adam(model.parameters(), lr=lr )

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(1, 2):
            train(epoch)
        #for x in range(3):
        #    print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
