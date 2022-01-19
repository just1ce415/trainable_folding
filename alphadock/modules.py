import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint
import math

from alphadock import utils


class RowAttentionWithPairBias(nn.Module):
    '''
    TODO: add gating
    '''
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        rec_num_c = global_config['rep_1d']['num_c']
        lig_num_c = global_config['rep_1d']['num_c']
        pair_rep_num_c = global_config['rep_2d']['num_c']

        self.rec_norm = nn.LayerNorm(rec_num_c)
        self.lig_norm = nn.LayerNorm(lig_num_c)

        self.rec_qkv = nn.Linear(rec_num_c, 6 * attn_num_c * num_heads, bias=False)
        self.lig_qkv = nn.Linear(lig_num_c, 6 * attn_num_c * num_heads, bias=False)

        self.rec_rec_project = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.lig_lig_project = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.rec_lig_project = nn.Linear(pair_rep_num_c, num_heads, bias=False)
        self.lig_rec_project = nn.Linear(pair_rep_num_c, num_heads, bias=False)

        self.rec_final = nn.Linear(attn_num_c * num_heads, rec_num_c)
        self.lig_final = nn.Linear(attn_num_c * num_heads, lig_num_c)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, rec_profile, lig_profile, pair):
        batch_size = rec_profile.shape[0]
        num_res = rec_profile.shape[1]
        num_atoms = lig_profile.shape[2]
        num_cep = lig_profile.shape[1]

        rec_profile = self.rec_norm(rec_profile)
        lig_profile = self.lig_norm(lig_profile)

        rec_lq, rec_lk, rec_lv, rec_rq, rec_rk, rec_rv = torch.chunk(self.rec_qkv(rec_profile).view(*rec_profile.shape[:-1], self.attn_num_c, 6 * self.num_heads), 6, dim=-1)
        lig_lq, lig_lk, lig_lv, lig_rq, lig_rk, lig_rv = torch.chunk(self.lig_qkv(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, 6 * self.num_heads), 6, dim=-1)

        weights = torch.zeros((batch_size, num_cep, num_res + num_atoms, num_res + num_atoms, self.num_heads), device=rec_profile.device, dtype=rec_profile.dtype)
        
        rec_rec_aff = torch.einsum('bich,bjch->bijh', rec_rq, rec_rk)
        rec_lig_aff = torch.einsum('bich,bmjch->bmijh', rec_lq, lig_rk)
        lig_rec_aff = torch.einsum('bich,bmjch->bmjih', rec_lk, lig_rq)
        lig_lig_aff = torch.einsum('bmich,bmjch->bmijh', lig_lq, lig_lk)

        rec_rec = pair[:, :num_res, :num_res]
        rec_lig = pair[:, :num_res, num_res:]
        lig_rec = pair[:, num_res:, :num_res]
        lig_lig = pair[:, num_res:, num_res:]

        factor = 1 / math.sqrt(self.attn_num_c)
        rec_rec_bias = self.rec_rec_project(rec_rec).view(*rec_rec.shape[:-1], self.num_heads) * factor
        lig_lig_bias = self.lig_lig_project(lig_lig).view(*lig_lig.shape[:-1], self.num_heads) * factor
        rec_lig_bias = self.rec_lig_project(rec_lig).view(*rec_lig.shape[:-1], self.num_heads) * factor
        lig_rec_bias = self.lig_rec_project(lig_rec).view(*lig_rec.shape[:-1], self.num_heads) * factor

        #print(rec_rec_aff.shape)
        #print(rec_rec_bias.shape)
        weights[:, :, :num_res, :num_res] = rec_rec_aff + rec_rec_bias
        weights[:, :, :num_res, num_res:] = rec_lig_aff + rec_lig_bias
        weights[:, :, num_res:, :num_res] = lig_rec_aff + lig_rec_bias
        weights[:, :, num_res:, num_res:] = lig_lig_aff + lig_lig_bias
        weights = torch.softmax(weights, dim=-2)

        rec_profile = torch.einsum('brch,birh->bich', rec_rv, weights[:, 0, :num_res, :num_res]) + torch.einsum('bmrch,bmirh->bmich', lig_rv, weights[:, :, :num_res, num_res:]).mean(1)
        lig_profile = torch.einsum('bmrch,bmirh->bmich', lig_lv, weights[:, :, num_res:, num_res:]) + torch.einsum('brch,bmirh->bmich', rec_lv, weights[:, :, num_res:, :num_res])

        rec_profile = self.rec_final(rec_profile.reshape(*rec_profile.shape[:-2], -1))
        lig_profile = self.lig_final(lig_profile.reshape(*lig_profile.shape[:-2], -1))
        return rec_profile, lig_profile


class LigColumnAttention(nn.Module):
    '''
    TODO: add gating
    '''
    def __init__(self, config, global_config):
        super().__init__()

        attn_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        lig_num_c = global_config['rep_1d']['num_c']

        self.lig_norm = nn.LayerNorm(lig_num_c)
        self.lig_qkv = nn.Linear(lig_num_c, attn_num_c * num_heads * 3, bias=False)
        self.lig_final = nn.Linear(attn_num_c * num_heads, lig_num_c)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, lig_profile):
        lig_profile = self.lig_norm(lig_profile)

        lig_q, lig_k, lig_v = torch.chunk(self.lig_qkv(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads * 3), 3, dim=-1)

        lig_lig_aff = torch.einsum('bmich,bnich->bmnih', lig_q, lig_k) / math.sqrt(self.attn_num_c)
        weights = torch.softmax(lig_lig_aff, dim=2)

        #lig_profile = torch.sum(lig_v[:, None] * weights[..., None, :], dim=2)
        lig_profile = torch.einsum('bmnih,bnich->bmich', weights, lig_v)
        lig_profile = self.lig_final(lig_profile.reshape(*lig_profile.shape[:-2], -1))
        return lig_profile


class ExtraColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.attn_num_c = config['attention_num_c']
        self.num_heads = config['num_heads']

        self.norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.kqv = nn.Linear(global_config['rep_1d']['num_c'], self.attn_num_c * (self.num_heads + 2), bias=False)
        self.gate = nn.Linear(global_config['rep_1d']['num_c'], self.attn_num_c * self.num_heads)
        self.final = nn.Linear(self.attn_num_c * self.num_heads, global_config['rep_1d']['num_c'])

    def forward(self, lig_profile):
        x = self.norm(lig_profile)

        q, k, v = torch.split(self.kqv(x).view(*x.shape[:-1], self.attn_num_c, self.num_heads + 2), [self.num_heads, 1, 1], dim=-1)
        q = torch.mean(q, dim=1)

        g = self.gate(x).view(*x.shape[:-1], self.attn_num_c, self.num_heads)

        w = torch.softmax(torch.einsum('bich,bsic->bsih', q, k.squeeze(-1)) / math.sqrt(self.attn_num_c), dim=1)
        out = g * torch.sum(w[..., None, :] * v, dim=1)[:, None]

        return self.final(out.view(*out.shape[:-2], self.attn_num_c * self.num_heads))


class Transition(nn.Module):
    def __init__(self, num_c, n):
        super().__init__()
        self.norm = nn.LayerNorm(num_c)
        self.l1 = nn.Linear(num_c, num_c * n)
        self.l2 = nn.Linear(num_c * n, num_c)

    def forward(self, x):
        x = self.norm(x)
        x = self.l1(x).relu_()
        x = self.l2(x)
        return x


class OuterProductMean(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c, out_c = global_config['rep_1d']['num_c'], global_config['rep_2d']['num_c']
        mid_c = config['mid_c']
        self.r_norm = nn.LayerNorm(in_c)
        self.l_norm = nn.LayerNorm(in_c)
        self.r_l = nn.Linear(in_c, mid_c * 4)
        self.l_l = nn.Linear(in_c, mid_c * 4)
        self.rr_final = nn.Linear(mid_c * mid_c, out_c)
        self.rl_final = nn.Linear(mid_c * mid_c, out_c)
        self.lr_final = nn.Linear(mid_c * mid_c, out_c)
        self.ll_final = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, rec_1d, lig_1d, pw_rep):
        rec_1d = self.r_norm(rec_1d)
        lig_1d = self.l_norm(lig_1d)

        r_ri, r_rj, r_li, r_lj = [x[..., -1] for x in torch.chunk(self.r_l(rec_1d).view(*rec_1d.shape[:-1], self.mid_c, 4), 4, dim=-1)]
        l_ri, l_rj, l_li, l_lj = [x[..., -1] for x in torch.chunk(self.l_l(lig_1d).view(*lig_1d.shape[:-1], self.mid_c, 4), 4, dim=-1)]

        rr = torch.einsum('bix,bjy->bijxy', r_ri.clone(), r_rj.clone())
        rl = torch.einsum('bmix,bmjy->bijxy', r_li.unsqueeze(1).repeat((1, lig_1d.shape[1], 1, 1)), l_rj) / lig_1d.shape[1]
        lr = torch.einsum('bmix,bmjy->bjixy', r_lj.unsqueeze(1).repeat((1, lig_1d.shape[1], 1, 1)), l_ri) / lig_1d.shape[1]
        ll = torch.einsum('bmix,bmjy->bijxy', l_li, l_lj) / lig_1d.shape[1]

        num_res = rec_1d.shape[1]
        pw_update = torch.zeros_like(pw_rep)
        pw_update[:, :num_res, :num_res] = self.rr_final(rr.flatten(start_dim=-2))
        pw_update[:, :num_res, num_res:] = self.rl_final(rl.flatten(start_dim=-2))
        pw_update[:, num_res:, :num_res] = self.lr_final(lr.flatten(start_dim=-2))
        pw_update[:, num_res:, num_res:] = self.ll_final(ll.flatten(start_dim=-2))
        return pw_update


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['rep_2d']['num_c']
        mid_c = config['mid_c']
        self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(mid_c)
        self.l1i = nn.Linear(in_c, mid_c)
        self.l1j = nn.Linear(in_c, mid_c)
        self.l1i_sigm = nn.Linear(in_c, mid_c)
        self.l1j_sigm = nn.Linear(in_c, mid_c)
        self.l2_proj = nn.Linear(mid_c, in_c)
        self.l3_sigm = nn.Linear(in_c, in_c)

    def forward(self, x2d, mask=None):
        x2d = self.norm1(x2d)
        i = self.l1i(x2d) * torch.sigmoid(self.l1i_sigm(x2d))
        j = self.l1j(x2d) * torch.sigmoid(self.l1j_sigm(x2d))
        if mask is not None:
            i *= mask[..., None]
            j *= mask[..., None]
        out = torch.einsum('bikc,bjkc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
        if mask is not None:
            out *= mask[..., None]
        return out


class TriangleMultiplicationOutgoingRecLig(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['rep_2d']['num_c']
        mid_c = config['mid_c']
        #self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(mid_c)
        self.l1i = nn.Linear(in_c, mid_c)
        self.l1j = nn.Linear(in_c, mid_c)
        self.l1i_sigm = nn.Linear(in_c, mid_c)
        self.l1j_sigm = nn.Linear(in_c, mid_c)
        self.l2_proj = nn.Linear(mid_c, in_c)
        self.l3_sigm = nn.Linear(in_c, in_c)

    def forward(self, x2d, i_range, j_range):
        #x2d = self.norm1(x2d)
        x2d_i, x2d_j = x2d[:, i_range[0]:i_range[1]], x2d[:, j_range[0]:j_range[1]]
        i = self.l1i(x2d_i) * torch.sigmoid(self.l1i_sigm(x2d_i))
        j = self.l1j(x2d_j) * torch.sigmoid(self.l1j_sigm(x2d_j))
        out = torch.einsum('bikc,bjkc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d[:, i_range[0]:i_range[1], j_range[0]:j_range[1]]))
        return out


class TriangleMultiplicationIngoing(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['rep_2d']['num_c']
        mid_c = config['mid_c']
        self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(mid_c)
        self.l1i = nn.Linear(in_c, mid_c)
        self.l1j = nn.Linear(in_c, mid_c)
        self.l1i_sigm = nn.Linear(in_c, mid_c)
        self.l1j_sigm = nn.Linear(in_c, mid_c)
        self.l2_proj = nn.Linear(mid_c, in_c)
        self.l3_sigm = nn.Linear(in_c, in_c)

    def forward(self, x2d, mask=None):
        x2d = self.norm1(x2d)
        i = self.l1i(x2d) * torch.sigmoid(self.l1i_sigm(x2d))
        j = self.l1j(x2d) * torch.sigmoid(self.l1j_sigm(x2d))
        if mask is not None:
            i *= mask[..., None]
            j *= mask[..., None]
        out = torch.einsum('bkic,bkjc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
        if mask is not None:
            out *= mask[..., None]
        return out


class TriangleMultiplicationIngoingRecLig(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        in_c = global_config['rep_2d']['num_c']
        mid_c = config['mid_c']
        #self.norm1 = nn.LayerNorm(in_c)
        self.norm2 = nn.LayerNorm(mid_c)
        self.l1i = nn.Linear(in_c, mid_c)
        self.l1j = nn.Linear(in_c, mid_c)
        self.l1i_sigm = nn.Linear(in_c, mid_c)
        self.l1j_sigm = nn.Linear(in_c, mid_c)
        self.l2_proj = nn.Linear(mid_c, in_c)
        self.l3_sigm = nn.Linear(in_c, in_c)

    def forward(self, x2d, i_range, j_range):
        #x2d = self.norm1(x2d)
        x2d_i, x2d_j = x2d[:, :, i_range[0]:i_range[1]], x2d[:, :, j_range[0]:j_range[1]]
        i = self.l1i(x2d_i) * torch.sigmoid(self.l1i_sigm(x2d_i))
        j = self.l1j(x2d_j) * torch.sigmoid(self.l1j_sigm(x2d_j))
        out = torch.einsum('bkic,bkjc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d[:, i_range[0]:i_range[1], j_range[0]:j_range[1]]))
        return out


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']
        self.rand_remove = config['rand_remove']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.qkv = nn.Linear(num_in_c, attention_num_c * num_heads * 3, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, x2d):
        x2d = self.norm(x2d)

        if self.rand_remove > 0.0 and self.training:
            selection = torch.randperm(x2d.shape[1], device=x2d.device)
            selection = selection[:max(1, int(selection.shape[0] * (1 - self.rand_remove)))]
            shape_full = x2d.shape
            res_ids_cart = torch.cartesian_prod(selection, selection)
            x2d = x2d[:, res_ids_cart[:, 0], res_ids_cart[:, 1]].reshape(x2d.shape[0], len(selection), len(selection), x2d.shape[-1])

        q, k, v = torch.chunk(self.qkv(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads * 3), 3, dim=-1)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(1).transpose_(2, 3)
        w = torch.einsum('bijch,bikch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b

        w = torch.softmax(w, dim=-2)
        out = torch.einsum('bijkh,bikch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))

        if self.rand_remove > 0.0 and self.training:
            out_full = torch.zeros(shape_full, device=out.device, dtype=out.dtype)
            out_full[:, res_ids_cart[:, 0], res_ids_cart[:, 1]] = out.flatten(1, 2)
            out = out_full

        return out


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']
        self.rand_remove = config['rand_remove']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.qkv = nn.Linear(num_in_c, attention_num_c * num_heads * 3, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, x2d):
        x2d = self.norm(x2d)

        if self.rand_remove > 0.0 and self.training:
            selection = torch.randperm(x2d.shape[1], device=x2d.device)
            selection = selection[:max(1, int(selection.shape[0] * (1 - self.rand_remove)))]
            shape_full = x2d.shape
            res_ids_cart = torch.cartesian_prod(selection, selection)
            x2d = x2d[:, res_ids_cart[:, 0], res_ids_cart[:, 1]].reshape(x2d.shape[0], len(selection), len(selection), x2d.shape[-1])

        q, k, v = torch.chunk(self.qkv(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads * 3), 3, dim=-1)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(2)
        w = torch.einsum('bijch,bkjch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b
        w = torch.softmax(w, dim=-2)
        out = torch.einsum('bijkh,bkjch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))

        if self.rand_remove > 0.0 and self.training:
            out_full = torch.zeros(shape_full, device=out.device, dtype=out.dtype)
            out_full[:, res_ids_cart[:, 0], res_ids_cart[:, 1]] = out.flatten(1, 2)
            out = out_full

        return out


class TemplatePairStackIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationIngoing'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])
        self.dropout2d_25 = nn.Dropout2d(0.25)

    def forward(self, x2d):
        x2d += self.TriangleAttentionStartingNode(x2d.clone()) #self.dropout2d_25(self.TriangleAttentionStartingNode(x2d.clone()))
        #x2d += self.dropout2d_25(self.TriangleAttentionEndingNode(x2d.clone()).transpose_(1, 2)).transpose_(1, 2)
        x2d += self.TriangleAttentionEndingNode(x2d.clone())
        x2d += self.TriangleMultiplicationOutgoing(x2d.clone())
        x2d += self.TriangleMultiplicationIngoing(x2d.clone())
        return x2d


class TemplatePairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rr_proj = nn.Linear(global_config['hh_rr'], global_config['rep_2d']['num_c'])
        self.ll_proj = nn.Linear(global_config['hh_ll'], global_config['rep_2d']['num_c'])
        self.rl_proj = nn.Linear(global_config['hh_rl'], global_config['rep_2d']['num_c'])
        self.lr_proj = nn.Linear(global_config['hh_lr'], global_config['rep_2d']['num_c'])

        self.layers = nn.ModuleList([TemplatePairStackIteration(config['TemplatePairStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.norm = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.config = config

    def forward(self, inputs):
        rr = self.rr_proj(inputs['rr_2d']).squeeze(0)
        rl = self.rl_proj(inputs['rl_2d']).squeeze(0)
        lr = self.lr_proj(inputs['lr_2d']).squeeze(0)
        ll = self.ll_proj(inputs['ll_2d']).squeeze(0)

        num_temp = rr.shape[0]
        num_res = rr.shape[1]
        num_atoms = ll.shape[1]
        out = torch.zeros((num_temp, num_res+num_atoms, num_res+num_atoms, rr.shape[-1]), device=rr.device, dtype=rr.dtype)

        out[:, :num_res, :num_res] = rr
        out[:, :num_res, num_res:] = rl
        out[:, num_res:, :num_res] = lr
        out[:, num_res:, num_res:] = ll

        for l in self.layers:
            #if self.config['TemplatePairStackIteration']['checkpoint']:
            #    out = checkpoint(lambda x: l(x), out)
            #else:
            out = l(out)

        return self.norm(out).unsqueeze(0)


class TemplatePointwiseAttention(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads
        self.num_in_c = num_in_c

        #self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, z2d, t2d):
        q = self.q(z2d).view(*z2d.shape[:-1], self.attention_num_c, self.num_heads)
        k = self.k(t2d).view(*t2d.shape[:-1], self.attention_num_c, self.num_heads)
        v = self.v(t2d).view(*t2d.shape[:-1], self.attention_num_c, self.num_heads)

        w = torch.softmax(torch.einsum('bijch,btijch->btijh', q, k) / math.sqrt(self.num_in_c), dim=1)
        out = torch.einsum('btijh,btijch->bijch', w, v)
        out = self.out(out.flatten(start_dim=-2))
        return out


class CEPPairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rr_proj = nn.Linear(global_config['frag_rr'], global_config['rep_2d']['num_c'])
        self.ll_proj = nn.Linear(global_config['frag_ll'], global_config['rep_2d']['num_c'])
        self.rl_proj = nn.Linear(global_config['frag_rl'], global_config['rep_2d']['num_c'])
        self.lr_proj = nn.Linear(global_config['frag_lr'], global_config['rep_2d']['num_c'])

        self.l_proj = nn.Linear(global_config['frag_lig'], global_config['rep_1d']['num_c'])

        self.layers = nn.ModuleList([TemplatePairStackIteration(config['TemplatePairStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.norm = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.config = config

    def forward(self, inputs):
        assert inputs['rr_2d'].shape[0] == 1

        rr = self.rr_proj(inputs['rr_2d']).squeeze(0)
        rl = self.rl_proj(inputs['rl_2d']).squeeze(0)
        lr = self.lr_proj(inputs['lr_2d']).squeeze(0)
        ll = self.ll_proj(inputs['ll_2d']).squeeze(0)

        lig_1d = self.l_proj(inputs['lig_1d']).squeeze(0)
        mapping = inputs['fragment_mapping'].squeeze(0)

        full_2d = torch.zeros((rr.shape[0], rr.shape[1] + ll.shape[1], rr.shape[1] + ll.shape[1], rr.shape[-1]), dtype=rr.dtype, device=rr.device)
        full_2d[:, :rr.shape[1], :rr.shape[1]] = rr
        full_2d[:, rr.shape[1]:, rr.shape[1]:] = ll
        full_2d[:, :rr.shape[1], rr.shape[1]:] = rl
        full_2d[:, rr.shape[1]:, :rr.shape[1]] = lr

        mask_2d = torch.zeros((rr.shape[0], rr.shape[1] + ll.shape[1], rr.shape[1] + ll.shape[1]), dtype=rr.dtype, device=rr.device)
        mask_2d[:, :rr.shape[1], :rr.shape[1]] = inputs['rr_2d_mask']
        mask_2d[:, rr.shape[1]:, rr.shape[1]:] = inputs['ll_2d_mask']
        mask_2d[:, :rr.shape[1], rr.shape[1]:] = inputs['rl_2d_mask']
        mask_2d[:, rr.shape[1]:, :rr.shape[1]] = inputs['lr_2d_mask']

        def checkpoint_fun(function):
            return lambda x, y: function(x, y)

        for l in self.layers:
            if self.config['TemplatePairStackIteration']['checkpoint']:
                full_2d = checkpoint(checkpoint_fun(l), full_2d.clone(), mask_2d)
            else:
                full_2d = l(full_2d.clone(), mask_2d)
        full_2d = self.norm(full_2d)
        full_2d *= mask_2d[..., None]

        # frag-frag interaction part
        ll_out = full_2d[:, rr.shape[1]:, rr.shape[1]:]

        # masked mean over rec residues of fragment-receptor interaction
        # (Nfrag, Natoms, C)
        frag_rec = full_2d[:, rr.shape[1]:, :rr.shape[1]].sum(2) / (mask_2d[:, rr.shape[1]:, :rr.shape[1]].sum(2)[..., None] + 1e-6)

        out_2d_list = []
        out_1d_list = []
        our_frag_rec_list = []
        for frag_id in range(mapping.shape[0]):
            #frag_num_res = num_res[frag_id]
            #frag_num_atoms = num_atoms[frag_id]
            frag_map = mapping[frag_id]
            out_2d_mapped = torch.zeros((frag_map.shape[0], frag_map.shape[0], full_2d.shape[-1]), device=full_2d.device, dtype=full_2d.dtype)

            target_gap_atoms = torch.where(frag_map == -1)[0]
            target_gap_pairs = torch.cartesian_prod(target_gap_atoms, target_gap_atoms)
            out_2d_mapped[target_gap_pairs[:, 0], target_gap_pairs[:, 1], -1] = 1

            target_atoms = torch.where(frag_map > -1)[0]
            template_atoms = frag_map[frag_map > -1]
            target_pairs = torch.cartesian_prod(target_atoms, target_atoms)
            template_pairs = torch.cartesian_prod(template_atoms, template_atoms)
            out_2d_mapped[target_pairs[:, 0], target_pairs[:, 1]] = ll_out[frag_id, template_pairs[:, 0], template_pairs[:, 1]]
            out_2d_list.append(out_2d_mapped)

            out_1d_mapped = torch.zeros((frag_map.shape[0], lig_1d.shape[-1]), device=lig_1d.device, dtype=lig_1d.dtype)
            out_1d_mapped[target_gap_atoms, -1] = 1  # add gap bin (do we need it?)
            out_1d_mapped[target_atoms] = lig_1d[frag_id, template_atoms]
            out_1d_list.append(out_1d_mapped)

            our_frag_rec_mapped = torch.zeros((frag_map.shape[0], frag_rec.shape[-1]), device=lig_1d.device, dtype=lig_1d.dtype)
            our_frag_rec_mapped[target_atoms] = frag_rec[frag_id, template_atoms]
            our_frag_rec_list.append(our_frag_rec_mapped)

        out_2d = torch.stack(out_2d_list).unsqueeze(0)
        out_1d = torch.stack(out_1d_list).unsqueeze(0)
        our_frag_rec = torch.stack(our_frag_rec_list).unsqueeze(0)

        return out_1d, out_2d, our_frag_rec

    def forward_old(self, inputs):
        assert inputs['rr_2d'].shape[0] == 1

        rr = self.rr_proj(inputs['rr_2d']).squeeze(0)
        rl = self.rl_proj(inputs['rl_2d']).squeeze(0)
        lr = self.lr_proj(inputs['lr_2d']).squeeze(0)
        ll = self.ll_proj(inputs['ll_2d']).squeeze(0)

        lig_1d = self.l_proj(inputs['lig_1d']).squeeze(0)

        num_res = inputs['num_res'].squeeze(0)
        num_atoms = inputs['num_atoms'].squeeze(0)
        mapping = inputs['fragment_mapping'].squeeze(0)

        out_2d_list = []
        out_1d_list = []

        for frag_id in range(mapping.shape[0]):
            frag_num_res = num_res[frag_id]
            frag_num_atoms = num_atoms[frag_id]
            out_size = frag_num_res + frag_num_atoms

            out_2d = torch.zeros((1, out_size, out_size, rr.shape[-1]), device=rr.device, dtype=rr.dtype)
            out_2d[0, :frag_num_res, :frag_num_res] = rr[frag_id, :frag_num_res, :frag_num_res]
            out_2d[0, :frag_num_res, frag_num_res:] = rl[frag_id, :frag_num_res, :frag_num_atoms]
            out_2d[0, frag_num_res:, :frag_num_res] = lr[frag_id, :frag_num_atoms, :frag_num_res]
            out_2d[0, frag_num_res:, frag_num_res:] = ll[frag_id, :frag_num_atoms, :frag_num_atoms]

            for l in self.layers:
                out_2d = l(out_2d.clone())
            out_2d = self.norm(out_2d)
            out_2d = out_2d[0, frag_num_res:, frag_num_res:]

            frag_map = mapping[frag_id]
            out_2d_mapped = torch.zeros((frag_map.shape[0], frag_map.shape[0], out_2d.shape[-1]), device=out_2d.device, dtype=out_2d.dtype)

            # TODO: replace with torch.gather or take
            # fill ones for gap pairs
            target_gap_atoms = torch.where(frag_map == -1)[0]
            target_gap_pairs = torch.cartesian_prod(target_gap_atoms, target_gap_atoms)
            out_2d_mapped[target_gap_pairs[:, 0], target_gap_pairs[:, 1], -1] = 1

            target_atoms = torch.where(frag_map > -1)[0]
            template_atoms = frag_map[frag_map > -1]
            target_pairs = torch.cartesian_prod(target_atoms, target_atoms)
            template_pairs = torch.cartesian_prod(template_atoms, template_atoms)
            out_2d_mapped[target_pairs[:, 0], target_pairs[:, 1]] = out_2d[template_pairs[:, 0], template_pairs[:, 1]]
            out_2d_list.append(out_2d_mapped)

            out_1d_mapped = torch.zeros((frag_map.shape[0], lig_1d.shape[-1]), device=lig_1d.device, dtype=lig_1d.dtype)
            out_1d_mapped[target_gap_atoms, -1] = 1
            out_1d_mapped[target_atoms] = lig_1d[frag_id, template_atoms]
            out_1d_list.append(out_1d_mapped)

        out_2d = torch.stack(out_2d_list).unsqueeze(0)
        out_1d = torch.stack(out_1d_list).unsqueeze(0)

        return {
            'frag_1d': out_1d,
            'frag_2d': out_2d
        }


class EvoformerIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.LigColumnAttention = LigColumnAttention(config['LigColumnAttention'], global_config)
        self.LigTransition = Transition(global_config['rep_1d']['num_c'], config['LigTransition']['n'])
        self.RecTransition = Transition(global_config['rep_1d']['num_c'], config['RecTransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)

        self.TriangleMultiplicationOutgoing_norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.TriangleMultiplicationOutgoing_rr = TriangleMultiplicationOutgoingRecLig(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationOutgoing_rl = TriangleMultiplicationOutgoingRecLig(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationOutgoing_lr = TriangleMultiplicationOutgoingRecLig(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationOutgoing_ll = TriangleMultiplicationOutgoingRecLig(config['TriangleMultiplicationOutgoing'], global_config)

        self.TriangleMultiplicationIngoing_norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.TriangleMultiplicationIngoing_rr = TriangleMultiplicationIngoingRecLig(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleMultiplicationIngoing_rl = TriangleMultiplicationIngoingRecLig(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleMultiplicationIngoing_lr = TriangleMultiplicationIngoingRecLig(config['TriangleMultiplicationIngoing'], global_config)
        self.TriangleMultiplicationIngoing_ll = TriangleMultiplicationIngoingRecLig(config['TriangleMultiplicationIngoing'], global_config)

        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)

        self.PairTransition_rr = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])
        self.PairTransition_rl = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])
        self.PairTransition_lr = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])
        self.PairTransition_ll = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)
        # TODO: fix dropout everywhere

    def forward(self, r1d, l1d, pair):
        a, b = self.RowAttentionWithPairBias(r1d.clone(), l1d.clone(), pair.clone())
        r1d += self.dropout1d_15(a)
        l1d += b #self.dropout2d_15(b)
        l1d += self.LigColumnAttention(l1d.clone())
        r1d += self.RecTransition(r1d.clone())
        l1d += self.LigTransition(l1d.clone())
        pair += self.OuterProductMean(r1d.clone(), l1d.clone(), pair)

        lig_size = l1d.shape[2]
        rec_size = r1d.shape[1]

        pair_copy = self.TriangleMultiplicationOutgoing_norm(pair.clone())
        rr = self.TriangleMultiplicationOutgoing_rr(pair_copy, [0, rec_size], [0, rec_size])
        rl = self.TriangleMultiplicationOutgoing_rl(pair_copy, [0, rec_size], [rec_size, rec_size+lig_size])
        lr = self.TriangleMultiplicationOutgoing_lr(pair_copy, [rec_size, rec_size+lig_size], [0, rec_size])
        ll = self.TriangleMultiplicationOutgoing_ll(pair_copy, [rec_size, rec_size+lig_size], [rec_size, rec_size+lig_size])
        pair[:, :rec_size, :rec_size] += rr
        pair[:, :rec_size, rec_size:] += rl
        pair[:, rec_size:, :rec_size] += lr
        pair[:, rec_size:, rec_size:] += ll

        pair_copy = self.TriangleMultiplicationIngoing_norm(pair.clone())
        rr = self.TriangleMultiplicationIngoing_rr(pair_copy, [0, rec_size], [0, rec_size])
        rl = self.TriangleMultiplicationIngoing_rl(pair_copy, [0, rec_size], [rec_size, rec_size+lig_size])
        lr = self.TriangleMultiplicationIngoing_lr(pair_copy, [rec_size, rec_size+lig_size], [0, rec_size])
        ll = self.TriangleMultiplicationIngoing_ll(pair_copy, [rec_size, rec_size+lig_size], [rec_size, rec_size+lig_size])
        pair[:, :rec_size, :rec_size] += rr
        pair[:, :rec_size, rec_size:] += rl
        pair[:, rec_size:, :rec_size] += lr
        pair[:, rec_size:, rec_size:] += ll

        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())

        pair_copy = pair.clone()
        pair[:, :rec_size, :rec_size] += self.PairTransition_rr(pair_copy[:, :rec_size, :rec_size])
        pair[:, :rec_size, rec_size:] += self.PairTransition_rl(pair_copy[:, :rec_size, rec_size:])
        pair[:, rec_size:, :rec_size] += self.PairTransition_lr(pair_copy[:, rec_size:, :rec_size])
        pair[:, rec_size:, rec_size:] += self.PairTransition_ll(pair_copy[:, rec_size:, rec_size:])
        return r1d.clone(), l1d.clone(), pair.clone()


class Evoformer(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerIteration(config['EvoformerIteration'], global_config) for x in range(config['num_iter'])])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class FragExtraStackIteration(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.RowAttentionWithPairBias = RowAttentionWithPairBias(config['RowAttentionWithPairBias'], global_config)
        self.ExtraColumnGlobalAttention = ExtraColumnGlobalAttention(config['ExtraColumnGlobalAttention'], global_config)
        self.LigTransition = Transition(global_config['rep_1d']['num_c'], config['LigTransition']['n'])
        self.RecTransition = Transition(global_config['rep_1d']['num_c'], config['RecTransition']['n'])
        self.OuterProductMean = OuterProductMean(config['OuterProductMean'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

        self.dropout1d_15 = nn.Dropout(0.15)
        self.dropout2d_15 = nn.Dropout2d(0.15)
        self.dropout2d_25 = nn.Dropout2d(0.25)

    def forward(self, rec, extra, pair):
        a, b = self.RowAttentionWithPairBias(rec.clone(), extra.clone(), pair.clone())
        rec += self.dropout1d_15(a)
        extra += b #self.dropout2d_15(b)
        extra += self.ExtraColumnGlobalAttention(extra.clone())
        rec += self.RecTransition(rec.clone())
        extra += self.LigTransition(extra.clone())
        pair += self.OuterProductMean(rec.clone(), extra.clone(), pair)
        #pair += self.dropout2d_25(self.TriangleMultiplicationOutgoing(pair.clone()))
        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        #pair += self.dropout2d_25(self.TriangleMultiplicationIngoing(pair.clone()))
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return rec.clone(), extra.clone(), pair.clone() # {'rec': rec, 'extra': extra, 'pair': pair}


class FragExtraStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.project = nn.Linear(global_config['extra_in_c'], global_config['rep_1d']['num_c'])
        self.layers = nn.ModuleList([FragExtraStackIteration(config['FragExtraStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.config = config

    def forward(self, rec, extra, pair):
        extra = self.project(extra)

        def checkpoint_fun(function):
            return lambda a, b, c: function(a.clone(), b.clone(), c.clone())

        for l in self.layers:
            if self.config['FragExtraStackIteration']['checkpoint']:
                rec, extra, pair = checkpoint(checkpoint_fun(l), rec, extra, pair)
            else:
                rec, extra, pair = l(rec, extra, pair)
        return rec, pair


class InitPairRepresentation(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()

        lig_in_c = global_config['lig_in_c']
        rec_in_c = global_config['rec_in_c']
        lig_in2d_c = global_config['lig_in2d_c']
        rec_in2d_c = global_config['rec_in2d_c']
        relpos_c = global_config['rec_relpos_c']
        pair_num_c = global_config['rep_2d']['num_c']

        self.l_proj1 = nn.Linear(lig_in_c, pair_num_c)
        self.l_proj2 = nn.Linear(lig_in_c, pair_num_c)
        self.r_proj1 = nn.Linear(rec_in_c, pair_num_c)
        self.r_proj2 = nn.Linear(rec_in_c, pair_num_c)
        self.relpos_proj = nn.Linear(relpos_c, pair_num_c)

        self.l2d_proj = nn.Linear(lig_in2d_c, pair_num_c)
        self.r2d_proj = nn.Linear(rec_in2d_c, pair_num_c)

    def forward(self, feats):
        l1d, l2d, r1d, r2d, relpos = feats['lig_1d'], feats['lig_2d'], feats['rec_1d'], feats['rec_2d'], feats['rec_relpos']

        # create pair representation
        l_proj1 = self.l_proj1(l1d)
        l_proj2 = self.l_proj2(l1d)
        r_proj1 = self.r_proj1(r1d)
        r_proj2 = self.r_proj2(r1d)
        ll_pair = l_proj1.unsqueeze(2) + l_proj2.unsqueeze(1)
        rl_pair = r_proj1.unsqueeze(2) + l_proj2.unsqueeze(1)
        lr_pair = l_proj1.unsqueeze(2) + r_proj2.unsqueeze(1)
        rr_pair = r_proj1.unsqueeze(2) + r_proj2.unsqueeze(1)

        # TODO: maybe do something more sophisticated ?
        ll_pair += self.l2d_proj(l2d)
        rr_pair += self.r2d_proj(r2d)

        # add relpos
        rr_pair += self.relpos_proj(relpos)

        num_batch = r1d.shape[0]
        num_res = r1d.shape[1]
        num_atoms = l1d.shape[1]
        num_pair_c = ll_pair.shape[-1]

        pair = torch.zeros((num_batch, num_res+num_atoms, num_res+num_atoms, num_pair_c), device=l1d.device, dtype=l1d.dtype)
        pair[:, :num_res, :num_res] = rr_pair
        pair[:, :num_res, num_res:] = rl_pair
        pair[:, num_res:, :num_res] = lr_pair
        pair[:, num_res:, num_res:] = ll_pair

        return pair


class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.rec_norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.lig_norm = nn.LayerNorm(global_config['rep_1d']['num_c'])
        self.x2d_norm = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.rr_proj = nn.Linear(global_config['rep_2d']['num_c'], config['rec_num_bins'])
        self.ll_proj = nn.Linear(global_config['rep_2d']['num_c'], config['lig_num_bins'])
        self.rl_proj = nn.Linear(global_config['rep_2d']['num_c'], config['rec_lig_num_bins'])
        self.lr_proj = nn.Linear(global_config['rep_2d']['num_c'], config['rec_lig_num_bins'])

        self.config = config

    def forward(self, inputs):
        #for k, v in inputs.items():
        #    print(k, v.device)

        rec_1d = self.rec_norm(inputs['rec_1d_prev'])
        lig_1d = self.lig_norm(inputs['lig_1d_prev'])
        rep_2d = self.x2d_norm(inputs['rep_2d_prev'])

        rec_size = rec_1d.shape[1]

        rec_crd = inputs['rec_cbeta_prev'][0]
        rec_mask = inputs['rec_mask_prev'][0]
        assert len(rec_crd.shape) == 2 and rec_crd.shape[-1] == 3, rec_crd.shape

        lig_crd = inputs['lig_coords_prev'][0]
        assert len(lig_crd.shape) == 2 and lig_crd.shape[-1] == 3, lig_crd.shape

        dmat = torch.sqrt(torch.square(rec_crd[:, None, :] - rec_crd[None, :, :]).sum(-1) + 10e-10)
        dgram = utils.dmat_to_dgram(dmat, self.config['rec_min_dist'], self.config['rec_max_dist'], self.config['rec_num_bins'])[1]
        rep_2d[0, :rec_size, :rec_size] += self.rr_proj(dgram * rec_mask[:, None, None] * rec_mask[None, :, None])

        dmat = torch.sqrt(torch.square(lig_crd[:, None, :] - lig_crd[None, :, :]).sum(-1) + 10e-10)
        dgram = utils.dmat_to_dgram(dmat, self.config['lig_min_dist'], self.config['lig_max_dist'], self.config['lig_num_bins'])[1]
        rep_2d[0, rec_size:, rec_size:] += self.ll_proj(dgram)

        dmat = torch.sqrt(torch.square(rec_crd[:, None, :] - lig_crd[None, :, :]).sum(-1) + 10e-10)
        dgram = utils.dmat_to_dgram(dmat, self.config['rec_lig_min_dist'], self.config['rec_lig_max_dist'], self.config['rec_lig_num_bins'])[1]
        rep_2d[0, :rec_size, rec_size:] += self.rl_proj(dgram * rec_mask[:, None, None])
        rep_2d[0, rec_size:, :rec_size] += self.lr_proj((dgram * rec_mask[:, None, None]).transpose(0, 1))

        return {'pair_update': rep_2d, 'lig_1d_update': lig_1d, 'rec_1d_update': rec_1d}


class InputEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        lig_in_c = global_config['lig_in_c']
        rec_in_c = global_config['rec_in_c']
        l1d_num_c = global_config['rep_1d']['num_c']
        r1d_num_c = global_config['rep_1d']['num_c']

        self.lig_1d_project = nn.Linear(lig_in_c, l1d_num_c).to(config['device'])
        self.rec_1d_project = nn.Linear(rec_in_c, r1d_num_c).to(config['device'])
        self.frag_main_project = nn.Linear(global_config['extra_in_c'], l1d_num_c).to(config['device'])

        self.InitPairRepresentation = InitPairRepresentation(global_config).to(config['device'])
        self.TemplatePairStack = TemplatePairStack(config['TemplatePairStack'], global_config).to(config['TemplatePairStack']['device'])
        self.TemplatePointwiseAttention = TemplatePointwiseAttention(config['TemplatePointwiseAttention'], global_config).to(config['TemplatePointwiseAttention']['device'])
        self.FragExtraStack = FragExtraStack(config['FragExtraStack'], global_config).to(config['FragExtraStack']['device'])
        self.RecyclingEmbedder = RecyclingEmbedder(config['RecyclingEmbedder'], global_config).to(config['device'])

        self.config = config

    def forward(self, inputs, recycling=None):
        # create pair representation
        pair = self.InitPairRepresentation({k: v.to(self.config['device']) for k, v in inputs['target'].items()})
        rec_1d = self.rec_1d_project(inputs['target']['rec_1d'].to(self.config['device']))

        # make lig 1d rep
        lig_1d = self.lig_1d_project(inputs['target']['lig_1d'].to(self.config['device'])).unsqueeze(1)
        if 'fragments' in inputs:
            lig_1d = self.frag_main_project(inputs['fragments']['main'].to(self.config['device'])) + lig_1d.clone()

        # add recycling
        if recycling is not None:
            recyc_out = self.RecyclingEmbedder({k: v.to(self.config['device']) for k, v in recycling.items()})
            pair += recyc_out['pair_update']
            rec_1d += recyc_out['rec_1d_update']
            lig_1d[:, 0] += recyc_out['lig_1d_update']

        # make template embedding
        if 'hhpred' in inputs:
            hh_inputs = {k: v.to(self.config['TemplatePairStack']['device']) for k, v in inputs['hhpred'].items()}
            if self.config['TemplatePairStack']['checkpoint']:
                hh_2d = checkpoint(lambda x: self.TemplatePairStack(x), hh_inputs)
            else:
                hh_2d = self.TemplatePairStack(hh_inputs)
            template_embedding = self.TemplatePointwiseAttention(pair.clone().to(self.config['TemplatePointwiseAttention']['device']), hh_2d.to(self.config['TemplatePointwiseAttention']['device']))

            # add embeddings to the pair rep
            pair += template_embedding.to(pair.device)

        # embed extra stack
        if 'fragments' in inputs and 'extra' in inputs['fragments']:
            rec_1d, pair = self.FragExtraStack(
                rec_1d.clone().to(self.config['FragExtraStack']['device']),
                inputs['fragments']['extra'].to(self.config['FragExtraStack']['device']),
                pair.to(self.config['FragExtraStack']['device'])
            )

        return {'l1d': lig_1d, 'r1d': rec_1d, 'pair': pair}


def example():
    with torch.autograd.set_detect_anomaly(True):
        model = EvoformerWithEmbedding(config, config)
        maps = [-1, -1, 0, 1, 2, -1, -1, -1, -1, -1]
        input = {
            'l1d': torch.zeros((1, 10, config['lig_in_c'])),
            'r1d': torch.zeros((1, 100, config['rec_in_c'])),
            'cep_feats_2d': torch.zeros((1, 3, 130, 130, config['cep_in_c'])),
            'cep_crops': torch.tensor([[[10, 5], [15, 7], [5, 8]]]),
            'cep_maps': torch.tensor([[maps, maps, maps]]),
            'template_feats_2d':  torch.zeros((1, 4, 110, 110, config['template']['num_feats'])),
        }
        #print(model(input)['pair'].sum().backward())

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Num params:', pytorch_total_params)


def example2():
    from config import config

    with torch.autograd.set_detect_anomaly(True):
        model = InputEmbedder(config['InputEmbedder'], config)
        num_res = 20
        num_atoms = 10
        num_hh = 3
        num_frag = 3
        num_frag_atoms = 20
        num_frag_res = 30

        mapping = -torch.ones((1, num_frag, num_atoms), dtype=int)
        mapping[0, 0, [3, 6, 7]] = torch.tensor([1, 2, 0])
        mapping[0, 1, [3, 6, 7, 9]] = torch.tensor([1, 2, 0, 3])
        mapping[0, 2, [3, 6, 4, 9, 2]] = torch.tensor([1, 2, 0, 3, 4])

        sample = {
            'target': {
                'lig_1d': torch.zeros((1, num_atoms, config['lig_in_c'])),
                'lig_2d': torch.zeros((1, num_atoms, num_atoms, config['lig_in2d_c'])),  # bonds
                'rec_1d': torch.zeros((1, num_res, config['rec_in_c'])),  # residue feats
                'rec_2d': torch.zeros((1, num_res, num_res, config['rec_in2d_c'])),  # distogram, cbeta_2d
                'rec_relpos': torch.zeros((1, num_res, num_res, config['rec_relpos_c'])),
            },

            'hhpred': {
                'lig_1d': torch.zeros((1, num_hh, num_atoms, config['hh_lig'])),
                'rec_1d': torch.zeros((1, num_hh, num_res, config['hh_rec'])),
                'rr_2d': torch.zeros((1, num_hh, num_res, num_res, config['hh_rr'])),
                'rl_2d': torch.zeros((1, num_hh, num_res, num_atoms, config['hh_rl'])),
                'lr_2d': torch.zeros((1, num_hh, num_atoms, num_res, config['hh_lr'])),
                'll_2d': torch.zeros((1, num_hh, num_atoms, num_atoms, config['hh_ll'])),
            },

            'fragments': {
                'lig_1d': torch.zeros((1, num_frag, num_frag_atoms, config['frag_lig'])),
                'rec_1d': torch.zeros((1, num_frag, num_frag_res, config['frag_rec'])),
                'rr_2d': torch.zeros((1, num_frag, num_frag_res, num_frag_res, config['frag_rr'])),
                'rl_2d': torch.zeros((1, num_frag, num_frag_res, num_frag_atoms, config['frag_rl'])),
                'lr_2d': torch.zeros((1, num_frag, num_frag_atoms, num_frag_res, config['frag_lr'])),
                'll_2d': torch.zeros((1, num_frag, num_frag_atoms, num_frag_atoms, config['frag_ll'])),
                'num_atoms': torch.tensor([[3, 4, 5]]),
                'num_res': torch.tensor([[10, 11, 12]]),
                'fragment_mapping': mapping
            }
        }
        model = InputEmbedder(config['InputEmbedder'], config)
        print([(k, v.shape) for k, v in model(sample).items()])


def example3():
    from config import config, DATA_DIR
    with torch.no_grad():
        model = InputEmbedder(config['InputEmbedder'], config).half() #.cuda()

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)

        from dataset import DockingDataset
        ds = DockingDataset(DATA_DIR, 'train_split/debug.json')
        #print(ds[0])
        item = ds[0]

        for k1, v1 in item.items():
            print(k1)
            for k2, v2 in v1.items():
                v1[k2] = torch.as_tensor(v2)[None].cuda()
                print('    ', k2, v1[k2].shape, v1[k2].dtype)

        print(item['fragments']['num_res'])
        print(item['fragments']['num_atoms'])
        print(model(item))


if __name__ == '__main__':
    example3()
