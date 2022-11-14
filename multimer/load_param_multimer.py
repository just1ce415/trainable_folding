from enum import Enum
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
from typing import Union, List
import sys
import os

sys.path.insert(1, '../')
from alphadock import docker
from alphadock import config

_NPZ_KEY_PREFIX = "alphafold/alphafold_iteration/"


class ParamType(Enum):
    LinearWeight = partial(  # hack: partial prevents fns from becoming methods
        lambda w: w.transpose(-1, -2)
    )
    LinearWeightMHA = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearMHAOutputWeight = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearWeightOPM = partial(
        lambda w: w.reshape(*w.shape[:-3], -1, w.shape[-1]).transpose(-1, -2)
    )
    LinearWeightTP = partial(lambda w: w.reshape(w.shape[0], 1))
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn


@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False


def _process_translations_dict(d, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            prefix = _NPZ_KEY_PREFIX if top_layer else ""
            sub_flat = {
                (prefix + "/".join([k, k_prime])): v_prime
                for k_prime, v_prime in _process_translations_dict(
                    v, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            k = "/" + k if not top_layer else k
            flat[k] = v

    return flat


def stacked(param_dict_list, out=None):
    """
    Args:
        param_dict_list:
            A list of (nested) Param dicts to stack. The structure of
            each dict must be the identical (down to the ParamTypes of
            "parallel" Params). There must be at least one dict
            in the list.
    """
    if out is None:
        out = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        if type(v[0]) is dict:
            out[k] = {}
            stacked(v, out=out[k])
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type,
                stacked=True,
            )

            out[k] = stacked_param

    return out


def assign(translation_dict, orig_weights):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(orig_weights[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
                if 'evoformer_iteration' in k:
                    weights = torch.unbind(weights, 0)
                else:
                    weights = torch.unbind(weights, 0)
            else:
                weights = [weights]
                ref = [ref]

            try:
                weights = list(map(param_type.transformation, weights))
                for p, w in zip(ref, weights):
                    p.copy_(w)
            except:
                print(k)
                print(ref[0].shape)
                print(weights[0].shape)
                raise


def import_jax_weights_(model):
    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))

    LinearBias = lambda l: (Param(l))

    LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))

    LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))

    LinearWeightOPM = lambda l: (Param(l, param_type=ParamType.LinearWeightOPM))

    LinearWeightTP = lambda l: (Param(l, param_type=ParamType.LinearWeightTP))

    LinearBiasTP = lambda l: (Param(l))

    LinearParamTP = lambda l: {
        "weights": LinearWeightTP(l.weight),
        "bias": LinearBiasTP(l.bias),
    }
    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }
    LayerNormParams = lambda l: {
        "scale": Param(l.weight),
        "offset": Param(l.bias),
    }
    GlobalColMSAAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt.query_norm),
        "attention":
            {
                "query_w": LinearWeightMHA(matt.q.weight),
                "key_w": LinearWeight(matt.k.weight),
                "value_w": LinearWeight(matt.v.weight),
                "gating_w": LinearWeightMHA(matt.gate.weight),
                "gating_b": LinearBiasMHA(matt.gate.bias),
                "output_w": Param(
                    matt.output.weight,
                    param_type=ParamType.LinearMHAOutputWeight,
                ),
                "output_b": LinearBias(matt.output.bias),
            }
    }

    AttentionParams = lambda att: {
        "query_w": LinearWeightMHA(att.linear_q.weight),
        "key_w": LinearWeightMHA(att.linear_k.weight),
        "value_w": LinearWeightMHA(att.linear_v.weight),
        "output_w": Param(
            att.linear_o.weight,
            param_type=ParamType.LinearMHAOutputWeight,
        ),
        "output_b": LinearBias(att.linear_o.bias),
    }

    AttentionGatedParams = lambda att: dict(
        **AttentionParams(att),
        **{
            "gating_w": LinearWeightMHA(att.linear_g.weight),
            "gating_b": LinearBiasMHA(att.linear_g.bias),
        },
    )
    MSAAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt.query_norm),
        "attention": AttentionGatedParams(matt.mha)
            #{
            #    "query_w": LinearWeightMHA(matt.q.weight),
            #    "key_w": LinearWeightMHA(matt.k.weight),
            #    "value_w": LinearWeightMHA(matt.v.weight),
            #    "gating_w": LinearWeightMHA(matt.gate.weight),
            #    "gating_b": LinearBiasMHA(matt.gate.bias),
            #    "output_w": Param(
            #        matt.output.weight,
            #        param_type=ParamType.LinearMHAOutputWeight,
            #    ),
            #    "output_b": LinearBias(matt.output.bias),
            #}
    }
    MSAAttPairBiasParams = lambda matt: dict(
        **MSAAttParams(matt),
        **{
            "feat_2d_norm": LayerNormParams(matt.feat_2d_norm),
            "feat_2d_weights": LinearWeight(matt.feat_2d_weights.weight),
        },
    )


    MSATransitionParams = lambda m: {
        "input_layer_norm": LayerNormParams(m.input_layer_norm),
        "transition1": LinearParams(m.transition1),
        "transition2": LinearParams(m.transition2),
    }
    OuterProductMeanParams = lambda o: {
        "layer_norm_input": LayerNormParams(o.layer_norm_input),
        "left_projection": LinearParams(o.left_projection),
        "right_projection": LinearParams(o.right_projection),
        "output_w": LinearWeightOPM(o.output.weight),
        "output_b": LinearBias(o.output.bias),
    }
    TriMulOutParams = lambda tri_mul: {
        "layer_norm_input": LayerNormParams(tri_mul.layer_norm_input),
        "left_projection": LinearParams(tri_mul.left_projection),
        "right_projection": LinearParams(tri_mul.right_projection),
        "left_gate": LinearParams(tri_mul.left_gate),
        "right_gate": LinearParams(tri_mul.right_gate),
        "center_layer_norm": LayerNormParams(tri_mul.center_layer_norm),
        "output_projection": LinearParams(tri_mul.output_projection),
        "gating_linear": LinearParams(tri_mul.gating_linear),
    }
    TriMulInParams = lambda tri_mul: {
        "layer_norm_input": LayerNormParams(tri_mul.layer_norm_input),
        "left_projection": LinearParams(tri_mul.left_projection),
        "right_projection": LinearParams(tri_mul.right_projection),
        "left_gate": LinearParams(tri_mul.left_gate),
        "right_gate": LinearParams(tri_mul.right_gate),
        "center_layer_norm": LayerNormParams(tri_mul.center_layer_norm),
        "output_projection": LinearParams(tri_mul.output_projection),
        "gating_linear": LinearParams(tri_mul.gating_linear),
    }
    TriAttParams = lambda tri_att: {
        "query_norm": LayerNormParams(tri_att.query_norm),
        "feat_2d_weights": LinearWeight(tri_att.feat_2d_weights.weight),
        "attention": AttentionGatedParams(tri_att.mha),
        #"attention": {
        #    "query_w": LinearWeightMHA(tri_att.q.weight),
        #    "key_w": LinearWeightMHA(tri_att.k.weight),
        #    "value_w": LinearWeightMHA(tri_att.v.weight),
        #    "gating_w": LinearWeightMHA(tri_att.gate.weight),
        #    "gating_b": LinearBiasMHA(tri_att.gate.bias),
        #    "output_w": Param(
        #        tri_att.output.weight,
        #        param_type=ParamType.LinearMHAOutputWeight,
        #    ),
        #    "output_b": LinearBias(tri_att.output.bias),
        #}
    }

    IPAParams = lambda ipa: {
        "q_scalar_projection":{
                "weights": LinearWeightMHA(ipa.q.weight),
        } ,
        "k_scalar_projection": {
                "weights": LinearWeightMHA(ipa.k.weight),
        } ,
        "v_scalar_projection": {
                "weights": LinearWeightMHA(ipa.v.weight),
        } ,
        "q_point_projection": {
            "point_projection":{
                "weights": LinearWeightMHA(ipa.q_points.weight),
                "bias": LinearBiasMHA(ipa.q_points.bias)
            }
        },
        "k_point_projection": {
            "point_projection":{
                "weights": LinearWeightMHA(ipa.k_points.weight),
                "bias": LinearBiasMHA(ipa.k_points.bias)
            }
        },
        "v_point_projection": {
            "point_projection":{
                "weights": LinearWeightMHA(ipa.v_points.weight),
                "bias": LinearBiasMHA(ipa.v_points.bias)
            }
        },
        "trainable_point_weights": Param(
            param=ipa.trainable_w, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.rr_kqv_2d),
        "output_projection": LinearParams(ipa.final_r),
    }

    FoldIterationParams = lambda sm: {
        "invariant_point_attention": IPAParams(sm.InvariantPointAttention),
        "attention_layer_norm": LayerNormParams(sm.rec_norm),
        "transition": LinearParams(sm.transition_r[0]),
        "transition_1": LinearParams(sm.transition_r[2]),
        "transition_2": LinearParams(sm.transition_r[4]),
        "transition_layer_norm": LayerNormParams(sm.rec_norm2),
        "quat_rigid": {
                "rigid": LinearParams(sm.backbone_update),
            },
        "rigid_sidechain": {
            "input_projection": LinearParams(sm.PredictSidechains.s_cur),
            "input_projection_1": LinearParams(sm.PredictSidechains.s_ini),
            "resblock1": LinearParams(sm.PredictSidechains.res1[1]),
            "resblock2": LinearParams(sm.PredictSidechains.res1[3]),
            "resblock1_1": LinearParams(sm.PredictSidechains.res2[1]),
            "resblock2_1": LinearParams(sm.PredictSidechains.res2[3]),
            "unnormalized_angles": LinearParams(sm.PredictSidechains.final[1]),
        },
    }

    TemplatePairBlockParams = lambda b: {
        "triangle_attention_starting_node": TriAttParams(b.TriangleAttentionStartingNode),
        "triangle_attention_ending_node": TriAttParams(b.TriangleAttentionEndingNode),
        "triangle_multiplication_outgoing": TriMulOutParams(b.TriangleMultiplicationOutgoing),
        "triangle_multiplication_incoming": TriMulInParams(b.TriangleMultiplicationIngoing),
        "pair_transition": MSATransitionParams(b.PairTransition),
    }

    def EvoformerBlockParams(b, is_extra_msa=False):
        if is_extra_msa:
            col_att_name = "msa_column_global_attention"
            msa_col_att_params = GlobalColMSAAttParams(b.ExtraColumnGlobalAttention)
        else:
            col_att_name = "msa_column_attention"
            msa_col_att_params = MSAAttParams(b.LigColumnAttention)

        d = {
            "msa_row_attention_with_pair_bias": MSAAttPairBiasParams(
                b.RowAttentionWithPairBias
            ),
            col_att_name: msa_col_att_params,
            "msa_transition": MSATransitionParams(b.RecTransition),
            "outer_product_mean":
                OuterProductMeanParams(b.OuterProductMean),
            "triangle_multiplication_outgoing":
                TriMulOutParams(b.TriangleMultiplicationOutgoing),
            "triangle_multiplication_incoming":
                TriMulInParams(b.TriangleMultiplicationIngoing),
            "triangle_attention_starting_node":
                TriAttParams(b.TriangleAttentionStartingNode),
            "triangle_attention_ending_node":
                TriAttParams(b.TriangleAttentionEndingNode),
            "pair_transition":
                MSATransitionParams(b.PairTransition),
        }

        return d

    ExtraMSABlockParams = partial(EvoformerBlockParams, is_extra_msa=True)

    tps_blocks = model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.TemplateEmbeddingIteration
    tps_blocks_params = stacked([TemplatePairBlockParams(b) for b in tps_blocks])

    ems_blocks = model.InputEmbedder.FragExtraStack.layers
    ems_blocks_params = stacked([ExtraMSABlockParams(b) for b in ems_blocks])

    evo_blocks = model.Evoformer
    evo_blocks_params = stacked([EvoformerBlockParams(b) for b in evo_blocks])

    translations = {
        "evoformer": {
            "preprocess_1d": LinearParams(model.InputEmbedder.preprocessing_1d),
            "preprocess_msa": LinearParams(model.InputEmbedder.preprocess_msa),
            "left_single": LinearParams(model.InputEmbedder.left_single),
            "right_single": LinearParams(model.InputEmbedder.right_single),
            "prev_pos_linear": LinearParams(model.InputEmbedder.RecyclingEmbedder.prev_pos_linear),
            "prev_msa_first_row_norm": LayerNormParams(
                model.InputEmbedder.RecyclingEmbedder.prev_msa_first_row_norm
            ),
            "prev_pair_norm": LayerNormParams(
                model.InputEmbedder.RecyclingEmbedder.prev_pair_norm
            ),
            "~_relative_encoding":{
                "position_activations": LinearParams(model.InputEmbedder.RecyclingEmbedder.position_activations)
            },
            "template_embedding": {
                "single_template_embedding": {
                    "query_embedding_norm": LayerNormParams(
                        model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.query_embedding_norm
                    ),
                    "template_pair_embedding_0": LinearParams(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_0),
                    "template_pair_embedding_1": LinearParamTP(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_1),
                    "template_pair_embedding_2": LinearParams(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_2),
                    "template_pair_embedding_3": LinearParams(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_3),
                    "template_pair_embedding_4": LinearParamTP(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_4),
                    "template_pair_embedding_5": LinearParamTP(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_5),
                    "template_pair_embedding_6": LinearParamTP(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_6),
                    "template_pair_embedding_7": LinearParamTP(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_7),
                    "template_pair_embedding_8": LinearParams(model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.template_pair_emb_8),
                    "template_embedding_iteration":  tps_blocks_params,
                    "output_layer_norm": LayerNormParams(
                        model.InputEmbedder.TemplateEmbedding.SingleTemplateEmbedding.output_layer_norm
                    ),
                },
            #     "attention": {
            #         "query_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.q.weight),
            #         "key_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.k.weight),
            #         "value_w": LinearWeightMHA(model.InputEmbedder.TemplatePointwiseAttention.v.weight),
            #         "output_w": Param(
            #             model.InputEmbedder.TemplatePointwiseAttention.out.weight,
            #             param_type=ParamType.LinearMHAOutputWeight,
            #         ),
            #     }
                "output_linear": LinearParams(model.InputEmbedder.TemplateEmbedding.output_linear),
            },
            "extra_msa_activations": LinearParams(
                model.InputEmbedder.extra_msa_activations
            ),
            "extra_msa_stack": ems_blocks_params,
            "template_single_embedding": LinearParams(
                model.TemplateEmbedding1D.template_single_embedding
            ),
            "template_projection": LinearParams(
                model.TemplateEmbedding1D.template_projection
            ),
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.EvoformerExtractSingleRec),
        },
        "structure_module": {
            "single_layer_norm": LayerNormParams(
                model.StructureModule.single_layer_norm
            ),
            "initial_projection": LinearParams(
                model.StructureModule.initial_projection
            ),
            "pair_layer_norm": LayerNormParams(
                model.StructureModule.pair_layer_norm
            ),
            "fold_iteration": FoldIterationParams(model.StructureModule.StructureModuleIteration),
        },
        "predicted_lddt_head": {
            "input_layer_norm": LayerNormParams(
                model.PredictedLddt.input_layer_norm
            ),
            "act_0": LinearParams(model.PredictedLddt.act_0),
            "act_1": LinearParams(model.PredictedLddt.act_1),
            "logits": LinearParams(model.PredictedLddt.logits),
        },
        "distogram_head": {
            "half_logits": LinearParams(model.Distogram.half_logits),
        },
        "predicted_aligned_error_head": {
            "logits": LinearParams(model.PredictedAlignedError.logits),
        },
        "experimentally_resolved_head": {
            "logits": LinearParams(
                model.ExperimentallyResolvedHead.logits
            ),
        },
        "masked_msa_head": {
            "logits": LinearParams(model.MaskedMsaHead.logits),
        },

    }
    flat = _process_translations_dict(translations)
    data = np.load('/data1/thunguyen/params/params_model_1_multimer_v2.npz')
    assign(flat, data)


# if __name__ == '__main__':
#     param_path = sys.argv[1]
#     data = np.load(param_path)
#     model = docker.DockerIteration(config.config, config.config)
#     import_jax_weights_(model)
#
#     param_out_path = os.path.basename(param_path)[:-4] + ".pth"
#     torch.save(model.state_dict(), param_out_path)
