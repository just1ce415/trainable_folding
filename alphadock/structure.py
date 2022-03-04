import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint
import math

from alphadock import quat_affine


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        self.num_head = config['num_head']
        self.num_scalar_qk = config['num_scalar_qk']
        self.num_point_qk = config['num_point_qk']
        self.num_2d_qk = config['num_2d_qk']
        self.num_scalar_v = config['num_scalar_v']
        self.num_point_v = config['num_point_v']
        self.num_2d_v = config['num_2d_v']

        self.num_output_c = global_config['num_single_c'] #config['num_channel']
        self.rep_1d_num_c = global_config['num_single_c']
        self.rep_2d_num_c = global_config['rep_2d']['num_c']

        self.rec_kqv_1d = nn.Linear(self.rep_1d_num_c, (self.num_scalar_qk * 2 + self.num_scalar_v) * self.num_head, bias=False)
        self.rec_kqv_point = nn.Linear(self.rep_1d_num_c, (self.num_point_qk * 2 + self.num_point_v) * self.num_head * 3, bias=False)
        self.rr_kqv_2d = nn.Linear(self.rep_2d_num_c, self.num_head, bias=False)
        self.final_r = nn.Linear(self.num_head * (self.rep_2d_num_c + self.num_scalar_v + 4 * self.num_point_v), self.num_output_c)

    def forward(self, rec_1d, rep_2d, rec_T):
        batch = rec_1d.shape[0]
        num_res = rec_1d.shape[1]
        rec_T = quat_affine.QuatAffine.from_tensor(rec_T)

        kqv = self.rec_kqv_1d(rec_1d).view([batch, num_res, self.num_head, self.num_scalar_qk * 2 + self.num_scalar_v])
        rec_k_1d, rec_q_1d, rec_v_1d = torch.split(kqv, [self.num_scalar_qk, self.num_scalar_qk, self.num_scalar_v], dim=-1)
        rr = self.rr_kqv_2d(rep_2d).view([batch, num_res, num_res, self.num_head])

        kqv_local = self.rec_kqv_point(rec_1d)  # [b, r, c]
        kqv_global = torch.stack(rec_T.apply_to_point(torch.chunk(kqv_local, 3, dim=-1)))  # [3, b, r, x]
        kqv_global = kqv_global.movedim(0, 2)   # [b, r, 3, x]
        kqv_global = kqv_global.view(*kqv_global.shape[:-1], self.num_head, self.num_point_qk*2+self.num_point_v)  # [b, r, 3, h, x]
        rec_k_point, rec_q_point, rec_v_point = torch.split(kqv_global, [self.num_point_qk, self.num_point_qk, self.num_point_v], dim=-1)

        q_point = rec_q_point  # [b, r, 3, h, p]
        k_point = rec_k_point
        d2mat = torch.sum(torch.square(q_point[:, :, None] - k_point[:, None, :]), dim=3)  # -> [b, r, r, h, p]

        # add sq distances
        #aff = -self.d2_weights * Wc * d2mat.sum(axis=-2)
        Wc = math.sqrt(2. / (9. * self.num_point_qk))
        aff = -Wc * d2mat.sum(axis=-1)   # TODO: add learnable weights

        # add pair bias
        aff += rr

        # add 1d affinity
        aff += torch.einsum('bihc,bjhc->bijh', rec_q_1d, rec_k_1d) / math.sqrt(self.num_scalar_qk)

        Wl = math.sqrt(1. / 3.)
        weights = torch.softmax(Wl * aff, dim=2)  # bijh

        out = []
        out.append((weights[..., None] * rep_2d[..., None, :]).sum(2).flatten(start_dim=-2))  # [b, r, c*h]

        out.append((weights.unsqueeze(-1) * rec_v_1d.unsqueeze(2)).sum(2).flatten(start_dim=-2))  # [b, r, c*h]

        out_global = torch.einsum('bijh,bjdhp->bidph', weights, rec_v_point)
        #out_global = out_global.movedim(1, 0)

        rec_out_local = torch.cat(rec_T.invert_point(torch.chunk(out_global, 3, dim=2)), dim=2) # [b, i, 3, p, h]
        out_local = rec_out_local.permute([0, 1, 3, 4, 2])  # [b, i, p, h, 3]

        # add local coords
        out.append(out_local.flatten(start_dim=2))  # [b, r, p*h*3]

        # add norm
        out.append(torch.sqrt(torch.square(out_local).sum(-1)).flatten(start_dim=2))  # [b, a+r, p*h]

        out_cat = torch.cat(out, dim=-1)
        return self.final_r(out_cat)


class PredictSidechains(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['num_single_c']
        num_c = config['num_c']
        self.num_torsions = global_config['num_torsions']

        self.s_cur = nn.Linear(num_in_c, num_c)
        self.s_ini = nn.Linear(num_in_c, num_c)

        self.res1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.res2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c)
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_c, self.num_torsions * 2)
        )

    def forward(self, s_cur, s_ini):
        a = self.s_cur(s_cur.clone()) + self.s_ini(s_ini)
        a += self.res1(a.clone())
        a += self.res2(a.clone())
        return self.final(a).reshape(*a.shape[:-1], self.num_torsions, 2)


class PredictLDDT(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['num_single_c']
        num_c = config['num_c']
        num_bins = config['num_bins']

        self.layers = nn.Sequential(
            nn.LayerNorm(num_in_c),
            nn.Linear(num_in_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_c),
            nn.ReLU(),
            nn.Linear(num_c, num_bins)
            #nn.Softmax(-1)
        )

    def forward(self, rep_1d):
        return self.layers(rep_1d)


class StructureModuleIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InvariantPointAttention = InvariantPointAttention(config['InvariantPointAttention'], global_config)
        self.drop = nn.Dropout(0.1)
        self.rec_norm = nn.LayerNorm(global_config['num_single_c'])
        self.rec_norm2 = nn.LayerNorm(global_config['num_single_c'])

        num_1dc = global_config['num_single_c']
        self.transition_r = nn.Sequential(
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc),
            nn.ReLU(),
            nn.Linear(num_1dc, num_1dc)
        )

        self.backbone_update = nn.Linear(num_1dc, 6)
        self.PredictSidechains = PredictSidechains(config['PredictSidechains'], global_config)
        self.PredictRecLDDT = PredictLDDT(config['PredictRecLDDT'], global_config)

    def forward(self, rec_1d_init, rec_1d, rep_2d, rec_T, rec_torsions):
        #rec_1d_init, rec_1d, rep_2d, rec_T = inputs['rec_1d_init'], inputs['rec_1d'], inputs['rep_2d'], inputs['rec_T']

        # IPA
        rec_1d_update = self.InvariantPointAttention(rec_1d.clone(), rep_2d, rec_T)
        rec_1d = self.rec_norm(self.drop(rec_1d + rec_1d_update))

        # transition
        rec_1d_update = self.transition_r(rec_1d)
        rec_1d = self.rec_norm2(self.drop(rec_1d + rec_1d_update))

        # update backbone
        rec_T = quat_affine.QuatAffine.from_tensor(rec_T)
        rec_T = rec_T.pre_compose(self.backbone_update(rec_1d.clone()))

        # sidechains
        rec_torsions = rec_torsions + self.PredictSidechains(rec_1d, rec_1d_init)

        # If output is a dict, gradient checkpointing doesn't calculate gradients,
        # so I have to return a list instead
        return rec_1d, rec_T.to_tensor(), rec_torsions, self.PredictRecLDDT(rec_1d.clone())


class PredictDistogram(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        num_in_c = global_config['rep_2d']['num_c']
        self.rr_proj = nn.Linear(num_in_c, config['rec_num_bins'])

    def forward(self, rep_2d, rec_size):
        sym = rep_2d + rep_2d.transpose(1, 2)
        rr = self.rr_proj(sym)
        return {'rr': rr}


class StructureModule(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.num_iter = config['num_iter']
        num_1dc = global_config['num_single_c']

        self.layers = nn.ModuleList([StructureModuleIteration(config['StructureModuleIteration'], global_config) for _ in range(self.num_iter)])
        self.norm_rec_1d_init = nn.LayerNorm(num_1dc)
        self.norm_2d_init = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.rec_1d_proj = nn.Linear(num_1dc, num_1dc)
        self.pred_distogram = PredictDistogram(config['PredictDistogram'], global_config)

        self.position_scale = global_config['position_scale']
        self.config = config
        self.global_config = global_config

    def forward(self, inputs):
        # batch size must be one
        assert inputs['r1d'].shape[0] == 1

        rec_1d_init = self.norm_rec_1d_init(inputs['r1d'])
        pair = self.norm_2d_init(inputs['pair'])
        rec_1d = self.rec_1d_proj(rec_1d_init)

        # Set masked frames to origin
        #rec_T = inputs['rec_bb_affine'].clone().to(rec_1d.device)
        #rec_T_masked = torch.where(inputs['rec_bb_affine_mask'] < 1)[1]
        #rec_T[:, rec_T_masked, :] = 0
        #rec_T[:, rec_T_masked, 0] = 1
        #rec_T[:, :, -3:] = rec_T[:, :, -3:] / self.position_scale

        rec_T = torch.zeros((1, rec_1d.shape[1], 7), device=rec_1d.device, dtype=rec_1d.dtype)
        rec_T[:, :, 0] = 1
        rec_T.requires_grad = True

        rec_torsions = torch.zeros((1, rec_1d.shape[1], self.global_config['num_torsions'], 2), device=rec_1d.device, dtype=rec_1d.dtype)
        rec_torsions[..., 0] = 1

        struct_dict = {
            'rec_1d_init': rec_1d_init,
            'rec_1d': rec_1d,
            'rep_2d': pair,
            'rec_T': rec_T,
            'rec_torsions': rec_torsions
        }

        struct_traj = [struct_dict]
        for l in self.layers:
            args = [
                struct_traj[-1]['rec_1d_init'],
                struct_traj[-1]['rec_1d'],
                struct_traj[-1]['rep_2d'],
                struct_traj[-1]['rec_T'],
                struct_traj[-1]['rec_torsions']
            ]
            if self.config['StructureModuleIteration']['checkpoint']:
                update = checkpoint(l, *args)
            else:
                update = l(*args)
            struct_traj.append(
                {
                    'rec_1d_init': rec_1d_init,
                    'rec_1d': update[0],
                    'rep_2d': pair,
                    'rec_T': update[1],
                    'rec_torsions': update[2],
                    'rec_lddt': update[3]
                }
            )

        return {
            'rec_T': torch.stack([x['rec_T'] for x in struct_traj[1:]], dim=1),
            'rec_torsions': torch.stack([x['rec_torsions'] for x in struct_traj[1:]], dim=1),
            'rec_1d': struct_traj[-1]['rec_1d'],
            'rec_lddt': torch.stack([x['rec_lddt'] for x in struct_traj[1:]], dim=1),
            'distogram': self.pred_distogram(pair, rec_1d.shape[1])
        }


if __name__ == '__main__':
    pass

