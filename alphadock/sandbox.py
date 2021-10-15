import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint

from alphadock import modules
from alphadock import config
from alphadock import docker
from alphadock import residue_constants


def test_TriangleAttentionStartingNode():
    model = modules.TriangleAttentionStartingNode({'attention_num_c': 32, 'num_heads': 4}, {'rep_2d': {'num_c': 64}})
    model.cuda()

    size = 512
    input = torch.ones((1, size, size, 64)).cuda()
    model(input)
    print(torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024**2)


def test_Evoformer():
    local_config = {
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
            'num_heads': 2
        },
        'TriangleAttentionEndingNode': {
            'attention_num_c': 32,
            'num_heads': 2
        },
        'PairTransition': {
            'n': 4
        }
    }

    model = modules.EvoformerIteration(local_config, {'rep_2d': {'num_c': 64}, 'rep_1d': {'num_c': 64}})
    model.cuda()

    num_res = 400
    num_atoms = 50
    num_msa = 64
    input = [
        torch.ones((1, num_res, 64)).cuda(),
        torch.ones((1, num_msa, num_atoms, 64)).cuda(),
        torch.ones((1, num_res+num_atoms, num_res+num_atoms, 64)).cuda()
    ]
    with torch.cuda.amp.autocast():
        #model(*input)[2].sum().backward()
        out = model(*input)
        (out[0].sum() + out[1].sum() + out[2].sum()).backward()

    #print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_memory_usage", row_limit=5))
    print(torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024**2)


def test_FragExtraStackIteration():
    local_config = {
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
            'num_heads': 2,
            'rand_remove': 0.25
        },
        'TriangleAttentionEndingNode': {
            'attention_num_c': 32,
            'num_heads': 2,
            'rand_remove': 0.25
        },
        'PairTransition': {
            'n': 4
        }
    }

    model = modules.FragExtraStackIteration(local_config, {'rep_2d': {'num_c': 64}, 'rep_1d': {'num_c': 64}})
    model.cuda()

    num_res = 400
    num_atoms = 50
    num_msa = 512
    input = [
        torch.ones((1, num_res, 64)).cuda(),
        torch.ones((1, num_msa, num_atoms, 64)).cuda(),
        torch.ones((1, num_res+num_atoms, num_res+num_atoms, 64)).cuda()
    ]

    with torch.cuda.amp.autocast():
        out = model(*input)
        (out[0].sum() + out[1].sum() + out[2].sum()).backward()

    #print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_memory_usage", row_limit=5))
    print(torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024**2)


def test_AlphaDock():
    config_local = {
        'rec_in_c': 23,
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
            'num_iter': 32,
            'device': 'cuda:2',
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
                'device': 'cuda:1',
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
            'num_iter': 4,
            'device': 'cuda:2',
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

    aatype = 'S'
    aaorder = residue_constants.restype_order_with_x[aatype]
    num_res = 400
    num_atoms = 50
    num_frag_main = 128
    num_frag_extra = 512
    num_hh = 6

    inputs = {
        'target': {
            'rec_1d': torch.ones(torch.Size([1, num_res, 23]), dtype= torch.float32),
            'rec_2d': torch.ones(torch.Size([1, num_res, num_res, 40]) ,dtype= torch.float32),
            'rec_relpos': torch.zeros(torch.Size([1, num_res, num_res, 65]) , dtype= torch.float32),
            'rec_atom14_coords': torch.zeros(torch.Size([1, num_res, 14, 3]) , dtype=torch.float32),
            'rec_atom14_has_coords': torch.zeros(torch.Size([1, num_res, 14]), dtype=torch.float32),
            'rec_atom37_coords': torch.zeros(torch.Size([1, num_res, 37, 3]) , dtype=torch.float32),
            'rec_atom37_has_coords': torch.zeros(torch.Size([1, num_res, 37]) , dtype=torch.float32),
            'rec_aatype': torch.zeros(torch.Size([1, num_res]), dtype=torch.int64),
            'rec_bb_affine': torch.zeros(torch.Size([1, num_res, 7]) , dtype= torch.float32),
            'rec_bb_affine_mask': torch.ones(torch.Size([1, num_res]) , dtype= torch.float32),
            'rec_atom14_atom_is_ambiguous': torch.zeros(torch.Size([1, num_res, 14]) , dtype= torch.float32),
            'rec_atom14_atom_exists': torch.zeros(torch.Size([1, num_res, 14]), dtype=  torch.float32),
            'rigidgroups_gt_frames': torch.zeros(torch.Size([1, num_res, 8, 12]) , dtype= torch.float32),
            'rigidgroups_gt_exists': torch.zeros(torch.Size([1, num_res, 8]) , dtype= torch.float32),
            'rigidgroups_group_exists': torch.zeros(torch.Size([1, num_res, 8]) , dtype= torch.float32),
            'rigidgroups_group_is_ambiguous': torch.zeros(torch.Size([1, num_res, 8]), dtype=  torch.float32),
            'rigidgroups_alt_gt_frames': torch.zeros(torch.Size([1, num_res, 8, 12]) , dtype= torch.float32),
            'lig_1d': torch.ones(torch.Size([1, num_atoms, 47]) , dtype= torch.float32),
            'lig_2d': torch.ones(torch.Size([1, num_atoms, num_atoms, 6]), dtype=  torch.float32),
        },
        'ground_truth': {
            'gt_aatype': torch.zeros(torch.Size([1, num_res]) , dtype=torch.int64),
            'gt_atom14_coords': torch.zeros(torch.Size([1, num_res, 14, 3]) , dtype=torch.float32),
            'gt_atom14_has_coords': torch.zeros(torch.Size([1, num_res, 14]) , dtype=torch.float32),
            'gt_atom14_coords_alt': torch.zeros(torch.Size([1, num_res, 14, 3]) , dtype=torch.float32),
            'gt_atom14_has_coords_alt': torch.zeros(torch.Size([1, num_res, 14]) , dtype=torch.float32),
            'gt_atom14_atom_is_ambiguous': torch.zeros(torch.Size([1, num_res, 14]) , dtype=torch.float32),
            'gt_torsions_sin_cos': torch.zeros(torch.Size([1, num_res, 7, 2]) , dtype=torch.float32),
            'gt_torsions_sin_cos_alt': torch.zeros(torch.Size([1, num_res, 7, 2]) , dtype=torch.float32),
            'gt_torsions_mask': torch.zeros(torch.Size([1, num_res, 7]) , dtype=torch.float32),
            'gt_residue_index': torch.zeros(torch.Size([1, num_res]) , dtype=torch.int64),
            'gt_has_frame': torch.zeros(torch.Size([1, num_res]) , dtype=torch.float32),
            'gt_lig_coords': torch.zeros(torch.Size([1, 4, num_atoms, 3]) , dtype=torch.float32),
            'gt_lig_has_coords': torch.zeros(torch.Size([1, 4, num_atoms]) , dtype=torch.float32),
        },
        'hhpred': {
            'lig_1d': torch.ones(torch.Size([1, num_hh, num_atoms, 49]) , dtype=torch.float32),
            'rec_1d': torch.ones(torch.Size([1, num_hh, num_res, 24]) , dtype=torch.float32),
            'll_2d': torch.ones(torch.Size([1, num_hh, num_atoms, num_atoms, 142]) , dtype=torch.float32),
            'rr_2d': torch.ones(torch.Size([1, num_hh, num_res, num_res, 84]) , dtype=torch.float32),
            'rl_2d': torch.ones(torch.Size([1, num_hh, num_res, num_atoms, 110]) , dtype=torch.float32),
            'lr_2d': torch.ones(torch.Size([1, num_hh, num_atoms, num_res, 110]) , dtype=torch.float32),
        },
        'fragments': {
            'main': torch.ones(torch.Size([1, num_frag_main, num_atoms, 238]) , dtype=torch.float32),
            'extra': torch.ones(torch.Size([1, num_frag_extra, num_atoms, 238]) , dtype=torch.float32),
        }
    }
    inputs['target']['rec_atom14_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
    inputs['target']['rec_atom37_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom37_mask[aaorder])
    inputs['target']['rec_aatype'][:] = aaorder
    inputs['target']['rec_bb_affine'][:, :, 0] = 1
    inputs['target']['rec_atom14_atom_exists'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
    inputs['target']['rigidgroups_gt_frames'][:] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    inputs['target']['rigidgroups_gt_exists'][:, :, :5] = 1
    inputs['target']['rigidgroups_group_exists'][:, :, :5] = 1
    inputs['target']['rigidgroups_alt_gt_frames'] = inputs['target']['rigidgroups_gt_frames'].clone()

    inputs['ground_truth']['gt_aatype'][:] = aaorder
    inputs['ground_truth']['gt_atom14_has_coords'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
    inputs['ground_truth']['gt_atom14_has_coords_alt'][:] = torch.from_numpy(residue_constants.restype_atom14_mask[aaorder])
    inputs['ground_truth']['gt_torsions_sin_cos'][:, :, :, 0] = 1
    inputs['ground_truth']['gt_torsions_sin_cos_alt'][:, :, :, 0] = 1
    inputs['ground_truth']['gt_torsions_mask'][:, :, :4] = 1
    inputs['ground_truth']['gt_has_frame'][:] = 1
    inputs['ground_truth']['gt_lig_has_coords'][:] = 1

    for k1, v1 in inputs.items():
        print(k1)
        for k2, v2 in v1.items():
            print('    ', k2, v1[k2].shape, v1[k2].dtype)

    model = docker.DockerIteration(config_local, config_local)
    model.train()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num params:', pytorch_total_params)

    with torch.cuda.amp.autocast():
        out = model(inputs)
        print(out['loss_total'])
        out['loss_total'].backward()

    for x in range(3):
        print('cuda:' + str(x), ':', torch.cuda.memory_stats(x)['allocated_bytes.all.peak'] / 1024**2)
    #for x in model.parameters():
    #    print(x.grad)


test_AlphaDock()

