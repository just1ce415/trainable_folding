from path import Path
import numpy as np

DATA_DIR = Path(__file__).abspath().dirname().dirname() / 'data_preparation' / 'data'

DTYPE_FLOAT = np.float32

DTYPE_INT = np.int64

config = {
    'rec_in_c': 24,
    'lig_in_c': 40,

    'lig_in2d_c': 6,
    'rec_in2d_c': 40,
    'rec_relpos_c': 65,

    'frag_rec': 24,
    'frag_lig': 41,
    'frag_rr': 84,
    'frag_ll': 126,
    'frag_rl': 102,
    'frag_lr': 102,

    'hh_rec': 24,
    'hh_lig': 41,
    'hh_rr': 84,
    'hh_ll': 126,
    'hh_rl': 102,
    'hh_lr': 102,
    
    'num_torsions': 7,

    'template': {
        'num_feats': 10
    },
    'rep_1d': {
        'num_c': 64
    },
    'rep_2d': {
        'num_c': 64
    },
    'rec_dist_num_bins': 40,
    'num_single_c': 384,

    'Evoformer': {
        'num_iter': 2,
        'EvoformerIteration': {
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
        'TemplatePointwiseAttention': {
            'attention_num_c': 32,
            'num_heads': 4
        },
        'CEPPairStack': {
            'num_iter': 2,
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
        }
    },
    'StructureModule': {
        'num_iter': 1,
        'StructureModuleIteration': {
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
            }
        }
    }
}