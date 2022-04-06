import numpy as np
from functools import partial
import torch
from enum import Enum
from typing import Union, List
from dataclasses import dataclass

data = np.load('/pool-data/data/thu/params/params_model_1_ptm.npz')
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
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn

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

def assign(translation_dict, orig_weights):
    for k, param in translation_dict.items():
        with torch.no_grad():
            weights = torch.as_tensor(orig_weights[k])
            ref, param_type = param.param, param.param_type
            if param.stacked:
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

@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False

def import_jax_weights_(model):

    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))

    LinearBias = lambda l: (Param(l))
    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }

    translations = {
        "evoformer": {
            "preprocess_1d": LinearParams(model.InputEmbedder.rec_1d_project),
            "preprocess_msa": LinearParams(model.InputEmbedder.main_msa_project),
            "left_single": LinearParams(model.InputEmbedder.InitPairRepresentation.r_proj1),
            "right_single": LinearParams(model.InputEmbedder.InitPairRepresentation.r_proj2),
            #"prev_pos_linear": LinearParams(model.recycling_embedder.linear),
            # "prev_msa_first_row_norm": LayerNormParams(
            #     model.recycling_embedder.layer_norm_m
            # ),
            # "prev_pair_norm": LayerNormParams(
            #     model.recycling_embedder.layer_norm_z
            # ),
            "pair_activiations": LinearParams(
                model.InputEmbedder.InitPairRepresentation.relpos_proj
            ),
            "template_embedding": {
                "single_template_embedding": {
                    "embedding2d": LinearParams(
                        model.template_pair_embedder.linear
                    ),
                    "template_pair_stack": {
                        "__layer_stack_no_state": tps_blocks_params,
                    },
                    "output_layer_norm": LayerNormParams(
                        model.template_pair_stack.layer_norm
                    ),
                },
                "attention": AttentionParams(model.template_pointwise_att.mha),
            },
            "extra_msa_activations": LinearParams(
                model.extra_msa_embedder.linear
            ),
            "extra_msa_stack": ems_blocks_params,
            "template_single_embedding": LinearParams(
                model.template_angle_embedder.linear_1
            ),
            "template_projection": LinearParams(
                model.template_angle_embedder.linear_2
            ),
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.evoformer.linear),
        },
    }
    flat = _process_translations_dict(translations)
    assign(flat, data)


for k in list(data.keys()):
    print(k, data[k].shape)
