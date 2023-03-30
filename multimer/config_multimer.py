config_multimer= {
    'aatype': 22,
    'msa': 51,
    'rel_feat': 73,
    'extra_msa_act': 26,
    'recycle': True,
    'model': {
        'embeddings_and_evoformer': {
            'evoformer_num_block': 49,
            'evoformer': {
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'attention_channel': 32,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'attention_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 256
                },
                'msa_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 256
                },
                'outer_product_mean': {
                    'chunk_size': 128,
                    'dropout_rate': 0.0,
                    'first': True,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'num_output_channel':128, 
                    'norm_channel': 256
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True,
                    'attention_channel': 32,
                    'norm_channel': 128
                },
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'attention_channel': 32,
                    'norm_channel': 128
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': 0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                }
            },
            'extra_msa': {
                'msa_column_attention': {
                    'dropout_rate': 0.0,
                    'gating': True,
                    'num_head': 8,
                    'attention_channel': 8,
                    'orientation': 'per_column',
                    'shared_dropout': True
                },
                'msa_row_attention_with_pair_bias': {
                    'dropout_rate': 0.15,
                    'gating': True,
                    'num_head': 8,
                    'attention_channel': 8,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 64
                },
                'msa_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 64
                },
                'outer_product_mean': {
                    'chunk_size': 128,
                    'dropout_rate': 0.0,
                    'first': True,
                    'num_outer_channel': 32,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'num_output_channel':128,
                    'norm_channel': 64
                },
                'pair_transition': {
                    'dropout_rate': 0.0,
                    'num_intermediate_factor': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                },
                'triangle_attention_ending_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_column',
                    'shared_dropout': True,
                    'attention_channel': 32,
                    'norm_channel': 128
                },
                'triangle_attention_starting_node': {
                    'dropout_rate': 0.25,
                    'gating': True,
                    'num_head': 4,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'attention_channel': 32,
                    'norm_channel': 128
                },
                'triangle_multiplication_incoming': {
                    'dropout_rate': 0.25,
                    'equation': 'kjc,kic->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                },
                'triangle_multiplication_outgoing': {
                    'dropout_rate': 0.25,
                    'equation': 'ikc,jkc->ijc',
                    'num_intermediate_channel': 128,
                    'orientation': 'per_row',
                    'shared_dropout': True,
                    'norm_channel': 128
                }
            },
            'extra_msa_channel': 64,
            'extra_msa_stack_num_block': 4,
            'num_msa': 252,
            'num_extra_msa': 1152,
            'masked_msa': {
                'profile_prob': 0.1,
                'replace_fraction': 0.15,
                'same_prob': 0.1,
                'uniform_prob': 0.1
            },
            'use_chain_relative': True,
            'max_relative_chain': 2,
            'max_relative_idx': 32,
            'seq_channel': 384,
            'msa_channel': 256,
            'pair_channel': 128,
            'prev_pos': {
                'max_bin': 20.75,
                'min_bin': 3.25,
                'num_bins': 15
            },
            'recycle_features': True,
            'recycle_pos': True,
            'template': {
                'attention': {
                    'gating': False,
                    'num_head': 4
                },
                'dgram_features': {
                    'max_bin': 50.75,
                    'min_bin': 3.25,
                    'num_bins': 39
                },
                'enabled': False,
                'max_templates': 4,
                'num_channels': 64,
                'subbatch_size': 128,
                'template_pair_stack': {
                    'num_block': 2,
                    'pair_transition': {
                        'dropout_rate': 0.0,
                        'num_intermediate_factor': 2,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'norm_channel': 64
                    },
                    'triangle_attention_ending_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'num_head': 4,
                        'orientation': 'per_column',
                        'shared_dropout': True,
                        'attention_channel': 16,
                        'norm_channel': 64
                    },
                    'triangle_attention_starting_node': {
                        'dropout_rate': 0.25,
                        'gating': True,
                        'num_head': 4,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'attention_channel': 16,
                        'norm_channel': 64
                    },
                    'triangle_multiplication_incoming': {
                        'dropout_rate': 0.25,
                        'equation': 'kjc,kic->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'norm_channel': 64
                    },
                    'triangle_multiplication_outgoing': {
                        'dropout_rate': 0.25,
                        'equation': 'ikc,jkc->ijc',
                        'num_intermediate_channel': 64,
                        'orientation': 'per_row',
                        'shared_dropout': True,
                        'norm_channel': 64
                    }
                }
            },
        },
        'global_config': {
            'deterministic': False,
            'multimer_mode': True,
            'subbatch_size': 4,
            'use_remat': False,
            'zero_init': True
        },
        'heads': {
            'distogram': {
                'first_break': 2.3125,
                'last_break': 21.6875,
                'num_bins': 64,
                'weight': 0.3
            },
            'experimentally_resolved': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'weight': 0.01
            },
            'masked_msa': {
                'weight': 2.0
            },
            'predicted_aligned_error': {
                'filter_by_resolution': True,
                'max_error_bin': 31.0,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 64,
                'num_channels': 128,
                'weight': 0.1
            },
            'predicted_lddt': {
                'filter_by_resolution': True,
                'max_resolution': 3.0,
                'min_resolution': 0.1,
                'num_bins': 50,
                'num_channels': 128,
                'weight': 0.01
            },
            'structure_module': {
                'angle_norm_weight': 0.01,
                'chi_weight': 0.5,
                'clash_overlap_tolerance': 1.5,
                'dropout': 0.1,
                'interface_fape': {
                    'atom_clamp_distance': 1000.0,
                    'loss_unit_distance': 20.0
                },
                'intra_chain_fape': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0
                },
                'num_channel': 384,
                'num_head': 12,
                'num_layer': 9,
                'num_layer_in_transition': 3,
                'num_point_qk': 4,
                'num_point_v': 8,
                'num_scalar_qk': 16,
                'num_scalar_v': 16,
                'position_scale': 20.0,
                'sidechain': {
                    'atom_clamp_distance': 10.0,
                    'loss_unit_distance': 10.0,
                    'num_channel': 128,
                    'num_residual_block': 2,
                    'weight_frac': 0.5
                },
                'structural_violation_loss_weight': 1.0,
                'violation_tolerance_factor': 12.0,
                'weight': 1.0
            }
        },
        'num_ensemble_eval': 1,
        'num_recycle': 3,
        'max_num_recycle_eval': 3,
        'min_num_recycle_eval': 3,
        'confident_plddt': 70,
        'resample_msa_in_recycling': False
    },
    'virtual_point': {
        'scale': 0.11892,
        'axis': 1
    }
}
