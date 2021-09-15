from path import Path

DATA_DIR = Path(__file__).abspath().dirname().dirname() / 'data_preparation' / 'data'

config = {
    'rec_in_c': 64,
    'lig_in_c': 64,
    'cep_in_c': 64,
    'lig_in2d_c': 43,
    'rec_in2d_c': 34,
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
    'Evoformer': {
        'num_blocks': 2,
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
    }
}