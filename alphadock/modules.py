import torch
from torch import nn
import torch.functional as F
from torch.utils.checkpoint import checkpoint
import math


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

        self.rec_qkv = nn.Linear(rec_num_c, 3 * attn_num_c * num_heads, bias=False)
        self.lig_qkv = nn.Linear(lig_num_c, 3 * attn_num_c * num_heads, bias=False)

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
        lig_profile = self.rec_norm(lig_profile)

        rec_q, rec_k, rec_v = torch.tensor_split(self.rec_qkv(rec_profile).view(*rec_profile.shape[:-1], self.attn_num_c, 3 * self.num_heads), 3, dim=-1)
        lig_q, lig_k, lig_v = torch.tensor_split(self.lig_qkv(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, 3 * self.num_heads), 3, dim=-1)

        weights = torch.zeros((batch_size, num_cep, num_res + num_atoms, num_res + num_atoms, self.num_heads), device=rec_profile.device, dtype=rec_profile.dtype)

        rec_rec_aff = torch.einsum('bich,bjch->bijh', rec_q, rec_k)
        rec_lig_aff = torch.einsum('bich,bmjch->bmijh', rec_q, lig_k)
        lig_rec_aff = torch.einsum('bich,bmjch->bmjih', rec_k, lig_q)
        lig_lig_aff = torch.einsum('bmich,bmjch->bmijh', lig_q, lig_k)

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

        rec_profile = torch.einsum('brch,birh->bich', rec_v, weights[:, 0, :num_res, :num_res]) + torch.einsum('bmrch,bmirh->bmich', lig_v, weights[:, :, :num_res, num_res:]).mean(1)
        lig_profile = torch.einsum('bmrch,bmirh->bmich', lig_v, weights[:, :, num_res:, num_res:]) + torch.einsum('brch,bmirh->bmich', rec_v, weights[:, :, num_res:, :num_res])

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

        lig_q, lig_k, lig_v = torch.tensor_split(self.lig_qkv(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads * 3), 3, dim=-1)

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
        self.r_l1 = nn.Linear(in_c, mid_c)
        self.r_l2 = nn.Linear(in_c, mid_c)
        self.l_l1 = nn.Linear(in_c, mid_c)
        self.l_l2 = nn.Linear(in_c, mid_c)
        self.rr_final = nn.Linear(mid_c * mid_c, out_c)
        self.rl_final = nn.Linear(mid_c * mid_c, out_c)
        self.lr_final = nn.Linear(mid_c * mid_c, out_c)
        self.ll_final = nn.Linear(mid_c * mid_c, out_c)
        self.mid_c = mid_c
        self.out_c = out_c

    def forward(self, rec_1d, lig_1d, pw_rep):
        r_i = self.r_l1(rec_1d)
        r_j = self.r_l2(rec_1d)
        l_i = self.l_l1(lig_1d)
        l_j = self.l_l2(lig_1d)

        rr = torch.einsum('bix,bjy->bijxy', r_i.clone(), r_j.clone())
        rl = torch.einsum('bmix,bmjy->bmijxy', r_i.unsqueeze_(1), l_j).mean(1)
        lr = torch.einsum('bmix,bmjy->bmjixy', r_j.unsqueeze_(1), l_i).mean(1)
        ll = torch.einsum('bmix,bmjy->bmijxy', l_i, l_j).mean(1)

        num_res = rec_1d.shape[1]
        pw_update = torch.zeros_like(pw_rep)
        pw_update[:, :num_res, :num_res] = self.rr_final(rr.view(*rr.shape[:-2], self.mid_c * self.mid_c))
        pw_update[:, :num_res, num_res:] = self.rl_final(rl.view(*rl.shape[:-2], self.mid_c * self.mid_c))
        pw_update[:, num_res:, :num_res] = self.lr_final(lr.view(*lr.shape[:-2], self.mid_c * self.mid_c))
        pw_update[:, num_res:, num_res:] = self.ll_final(ll.view(*ll.shape[:-2], self.mid_c * self.mid_c))
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


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, x2d, mask=None):
        x2d = self.norm(x2d)
        q = self.q(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        k = self.k(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        v = self.v(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(1).transpose_(2, 3)
        w = torch.einsum('bijch,bikch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b
        if mask is not None:
            w = (w + 100.0) * mask[:, :, None, :, None] - 100.0
        w = torch.softmax(w, dim=-2)
        out = torch.einsum('bijkh,bikch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))
        if mask is not None:
            out *= mask[..., None]
        return out


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        attention_num_c = config['attention_num_c']
        num_heads = config['num_heads']
        num_in_c = global_config['rep_2d']['num_c']

        self.attention_num_c = attention_num_c
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(num_in_c)
        self.q = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.k = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.v = nn.Linear(num_in_c, attention_num_c * num_heads, bias=False)
        self.bias = nn.Linear(num_in_c, num_heads, bias=False)
        self.gate = nn.Linear(num_in_c, attention_num_c * num_heads)
        self.out = nn.Linear(attention_num_c * num_heads, num_in_c)

    def forward(self, x2d, mask=None):
        x2d = self.norm(x2d)
        q = self.q(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        k = self.k(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        v = self.v(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(2)
        w = torch.einsum('bijch,bkjch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b
        if mask is not None:
            w = (w + 100.0) * mask.transpose(1, 2)[:, None, :, :, None] - 100.0
        w = torch.softmax(w, dim=-2)
        out = torch.einsum('bijkh,bkjch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))
        if mask is not None:
            out *= mask[..., None]
        return out


class TemplatePairStackIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationIngoing'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

    def forward(self, x2d, mask=None):
        #x2d = x2d.clone()
        x2d += self.TriangleAttentionStartingNode(x2d.clone(), mask)
        x2d += self.TriangleAttentionEndingNode(x2d.clone(), mask)
        x2d += self.TriangleMultiplicationOutgoing(x2d.clone(), mask)
        x2d += self.TriangleMultiplicationIngoing(x2d.clone(), mask)
        x2d += self.PairTransition(x2d.clone())
        if mask is not None:
            x2d *= mask[..., None]
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

        #out = out.clone()

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

        for l in self.layers:
            full_2d = l(full_2d.clone(), mask_2d)
        full_2d = self.norm(full_2d)
        full_2d *= mask_2d[..., None]

        # frag-frag interaction part
        ll_out = full_2d[:, rr.shape[1]:, rr.shape[1]:]

        # masked mean over rec residues of fragment-receptor interaction
        # (Nfrag, Natoms, C)
        frag_rec = full_2d[:, rr.shape[1]:, :rr.shape[1]].sum(2) / mask_2d[:, rr.shape[1]:, :rr.shape[1]].sum(2)[..., None]

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

        return {
            'frag_1d': out_1d,
            'frag_2d': out_2d,
            'our_frag_rec': our_frag_rec
        }

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
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

    # TODO: add dropout
    def forward(self, x):
        r1d, l1d, pair = x['r1d'], x['l1d'], x['pair']
        a, b = self.RowAttentionWithPairBias(r1d.clone(), l1d.clone(), pair.clone())
        r1d += a
        l1d += b
        l1d += self.LigColumnAttention(l1d.clone())
        r1d += self.RecTransition(r1d.clone())
        l1d += self.LigTransition(l1d.clone())
        pair += self.OuterProductMean(r1d.clone(), l1d.clone(), pair)
        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return {'l1d': l1d, 'r1d': r1d, 'pair': pair}


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

    # TODO: add dropout
    def forward(self, x):
        rec, extra, pair = x['rec'], x['extra'], x['pair']
        a, b = self.RowAttentionWithPairBias(rec.clone(), extra.clone(), pair.clone())
        rec += a
        extra += b
        extra += self.ExtraColumnGlobalAttention(extra.clone())
        rec += self.RecTransition(rec.clone())
        extra += self.LigTransition(extra.clone())
        pair += self.OuterProductMean(rec.clone(), extra.clone(), pair)
        pair += self.TriangleMultiplicationOutgoing(pair.clone())
        pair += self.TriangleMultiplicationIngoing(pair.clone())
        pair += self.TriangleAttentionStartingNode(pair.clone())
        pair += self.TriangleAttentionEndingNode(pair.clone())
        pair += self.PairTransition(pair.clone())
        return {'rec': rec, 'extra': extra, 'pair': pair}


class FragExtraStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.project = nn.Linear(global_config['extra_in_c'], global_config['rep_1d']['num_c'])
        self.layers = nn.ModuleList([FragExtraStackIteration(config['FragExtraStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.config = config

    def forward(self, rec, extra, pair):
        extra = self.project(extra)
        x = {'rec': rec, 'extra': extra, 'pair': pair}
        print({k: v.shape for k, v in x.items()})
        for l in self.layers:
            if self.config['FragExtraStackIteration']['checkpoint']:
                x = checkpoint(lambda y: l(y), x)
            else:
                x = l(x)
            print({k: v.shape for k, v in x.items()})
        return x['rec'], x['pair']


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


class InputEmbedder(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()

        lig_in_c = global_config['lig_in_c']
        rec_in_c = global_config['rec_in_c']
        l1d_num_c = global_config['rep_1d']['num_c']
        r1d_num_c = global_config['rep_1d']['num_c']

        self.l_feat = nn.Linear(lig_in_c, l1d_num_c)
        self.r_feat = nn.Linear(rec_in_c, r1d_num_c)

        self.InitPairRepresentation = InitPairRepresentation(global_config)
        self.CEPPairStack = CEPPairStack(config['CEPPairStack'], global_config)
        self.CEPPointwiseAttention = TemplatePointwiseAttention(config['CEPPointwiseAttention'], global_config)
        self.TemplatePairStack = TemplatePairStack(config['TemplatePairStack'], global_config)
        self.TemplatePointwiseAttention = TemplatePointwiseAttention(config['TemplatePointwiseAttention'], global_config)
        self.FragExtraStack = FragExtraStack(config['FragExtraStack'], global_config)

        self.config = config

    def forward(self, inputs):
        # create pair representation
        pair = self.InitPairRepresentation(inputs['target'])

        # CEP embedding
        num_res = inputs['target']['rec_1d'].shape[1]
        if 'fragments' in inputs:
            if self.config['CEPPairStack']['checkpoint']:
                cep = checkpoint(lambda x: self.CEPPairStack(x), inputs['fragments'])
            else:
                cep = self.CEPPairStack(inputs['fragments'])
            cep_embedding = self.CEPPointwiseAttention(pair[:, num_res:, num_res:], cep['frag_2d'])

        # make template embedding
        if 'hhpred' in inputs:
            if self.config['TemplatePairStack']['checkpoint']:
                hh_2d = checkpoint(lambda x: self.TemplatePairStack(x), inputs['hhpred'])
            else:
                hh_2d = self.TemplatePairStack(inputs['hhpred'])
            template_embedding = self.TemplatePointwiseAttention(pair.clone(), hh_2d)

        rep_rec_1d = self.r_feat(inputs['target']['rec_1d'])
        # TODO: add linear rec template feats. In the future check if we can use rec profiles from cep layer

        # add embeddings to the pair rep
        if 'hhpred' in inputs:
            pair += template_embedding

        # update lig-lig from cep_embedding
        # embed extra stack
        if 'fragments' in inputs:
            pair[:, num_res:, num_res:] += cep_embedding
            rep_rec_1d, pair = self.FragExtraStack(rep_rec_1d.clone(), inputs['fragments']['extra'], pair)

        # make 1d rep
        l_feat = self.l_feat(inputs['target']['lig_1d'])
        if 'fragments' in inputs:
            rep_lig_1d = cep['frag_1d'] + cep['frag_2d'].mean(3) + cep['our_frag_rec'] + l_feat.unsqueeze(1)
        else:
            rep_lig_1d = l_feat.unsqueeze(1)
        # TODO: concat linear lig template feats

        return {'l1d': rep_lig_1d, 'r1d': rep_rec_1d, 'pair': pair}


class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()
        self.m_norm = nn.LayerNorm(global_config['cep_num_c'])
        self.z_norm = nn.LayerNorm(global_config['rep_2d']['num_c'])
        self.d_linear = nn.Linear(global_config['rec_dist_num_bins'], global_config['rep_2d']['num_c'])

    def forward(self, x):
        pass


class EvoformerWithEmbedding(torch.nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.InputEmbedder = InputEmbedder(config['InputEmbedder'], global_config)
        self.Evoformer = Evoformer(config['Evoformer'], global_config)

    def forward(self, feats: dict, recycled=None):
        x = self.InputEmbedder(feats)
        x = self.Evoformer(x)
        return x


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
        print(model(input)['pair'].sum().backward())

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
        model = InputEmbedder(config['InputEmbedder'], config).half().cuda()

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
