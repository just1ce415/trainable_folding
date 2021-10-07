from path import Path
import numpy as np

DATA_DIR = Path(__file__).abspath().dirname().dirname() / 'data_preparation' / 'data'

TEST_DATA_DIR = Path(__file__).abspath().dirname() / 'test_data'

DTYPE_FLOAT = np.float32

DTYPE_INT = np.int64

config = {
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
        'lddt_rec_bin_size': 2,
        'lddt_rec_num_bins': 50,
        'lddt_lig_bin_size': 2,
        'lddt_lig_num_bins': 50,
    },

    'Evoformer': {
        'num_iter': 16,
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
                'num_heads': 4
            },
            'TriangleAttentionEndingNode': {
                'attention_num_c': 32,
                'num_heads': 4
            },
            'PairTransition': {
                'n': 4
            }
        }
    },
    'InputEmbedder': {
        'TemplatePairStack': {
            'num_iter': 2,
            'checkpoint': True,
            'TemplatePairStackIteration': {
                #'checkpoint': True,
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 128
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 128
                },
                'PairTransition': {
                    'n': 2
                }
            }
        },
        'TemplatePointwiseAttention': {
            'attention_num_c': 32,
            'num_heads': 4
        },
        'CEPPairStack': {
            'num_iter': 2,
            'checkpoint': True,
            'TemplatePairStackIteration': {
                'TriangleAttentionStartingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4
                },
                'TriangleMultiplicationOutgoing': {
                    'mid_c': 128
                },
                'TriangleMultiplicationIngoing': {
                    'mid_c': 128
                },
                'PairTransition': {
                    'n': 2
                }
            }
        },
        'CEPPointwiseAttention': {
            'attention_num_c': 32,
            'num_heads': 4
        },
        'FragExtraStack': {
            'num_iter': 4,
            'FragExtraStackIteration': {
                'checkpoint': True,
                'RowAttentionWithPairBias': {
                    'attention_num_c': 8,
                    'num_heads': 8
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
                    'num_heads': 4
                },
                'TriangleAttentionEndingNode': {
                    'attention_num_c': 32,
                    'num_heads': 4
                },
                'PairTransition': {
                    'n': 4
                }
            }
        }
    },
    'StructureModule': {
        'num_iter': 2,
        'StructureModuleIteration': {
            'checkpoint': False,
            'InvariantPointAttention': {
                'num_head': 4,
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
        }
    }
}