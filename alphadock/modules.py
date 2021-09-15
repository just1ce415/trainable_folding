import torch
from torch import nn
import torch.functional as F
import math
from config import config


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

        self.rec_q = nn.Linear(rec_num_c, attn_num_c * num_heads, bias=False)
        self.rec_k = nn.Linear(rec_num_c, attn_num_c * num_heads, bias=False)
        self.rec_v = nn.Linear(rec_num_c, attn_num_c * num_heads, bias=False)
        self.lig_q = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)
        self.lig_k = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)
        self.lig_v = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)

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

        rec_q = self.rec_q(rec_profile).view(*rec_profile.shape[:-1], self.attn_num_c, self.num_heads)
        rec_k = self.rec_k(rec_profile).view(*rec_profile.shape[:-1], self.attn_num_c, self.num_heads)
        rec_v = self.rec_v(rec_profile).view(*rec_profile.shape[:-1], self.attn_num_c, self.num_heads)
        lig_q = self.lig_q(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)
        lig_k = self.lig_k(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)
        lig_v = self.lig_v(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)

        weights = torch.zeros((batch_size, num_cep, num_res + num_atoms, num_res + num_atoms, self.num_heads), device=rec_profile.device, dtype=rec_profile.dtype)

        #print(rec_q.shape)
        #print(lig_k.shape)
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

        self.lig_q = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)
        self.lig_k = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)
        self.lig_v = nn.Linear(lig_num_c, attn_num_c * num_heads, bias=False)

        self.lig_final = nn.Linear(attn_num_c * num_heads, lig_num_c)

        self.attn_num_c = attn_num_c
        self.num_heads = num_heads

    def forward(self, lig_profile):
        lig_profile = self.lig_norm(lig_profile)

        lig_q = self.lig_q(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)
        lig_k = self.lig_k(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)
        lig_v = self.lig_v(lig_profile).view(*lig_profile.shape[:-1], self.attn_num_c, self.num_heads)

        factor = 1 / math.sqrt(self.attn_num_c)
        lig_lig_aff = torch.einsum('bmich,bmjch->bmijh', lig_q, lig_k) * factor
        weights = torch.softmax(lig_lig_aff, dim=-2)

        lig_profile = torch.einsum('bmrch,bmirh->bmich', lig_v, weights)
        lig_profile = self.lig_final(lig_profile.reshape(*lig_profile.shape[:-2], -1))
        return lig_profile


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

    def forward(self, x2d):
        x2d = self.norm1(x2d)
        i = self.l1i(x2d) * torch.sigmoid(self.l1i_sigm(x2d))
        j = self.l1j(x2d) * torch.sigmoid(self.l1j_sigm(x2d))
        out = torch.einsum('bikc,bjkc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
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

    def forward(self, x2d):
        x2d = self.norm1(x2d)
        i = self.l1i(x2d) * torch.sigmoid(self.l1i_sigm(x2d))
        j = self.l1j(x2d) * torch.sigmoid(self.l1j_sigm(x2d))
        out = torch.einsum('bkic,bkjc->bijc', i, j)
        out = self.norm2(out)
        out = self.l2_proj(out)
        out = out * torch.sigmoid(self.l3_sigm(x2d))
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

    def forward(self, x2d):
        x2d = self.norm(x2d)

        q = self.q(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        k = self.k(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        v = self.v(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(1).transpose_(2, 3)
        w = torch.softmax(torch.einsum('bijch,bikch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b, dim=-2)
        out = torch.einsum('bijkh,bikch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))
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

    def forward(self, x2d):
        x2d = self.norm(x2d)

        q = self.q(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        k = self.k(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        v = self.v(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads)
        b = self.bias(x2d)
        g = torch.sigmoid(self.gate(x2d).view(*x2d.shape[:-1], self.attention_num_c, self.num_heads))

        b = b.unsqueeze_(2)
        w = torch.softmax(torch.einsum('bijch,bkjch->bijkh', q, k) / math.sqrt(self.attention_num_c) + b, dim=-2)
        out = torch.einsum('bijkh,bkjch->bijch', w, v) * g
        out = self.out(out.flatten(start_dim=-2))
        return out


class TemplatePairStackIteration(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.TriangleAttentionStartingNode = TriangleAttentionStartingNode(config['TriangleAttentionStartingNode'], global_config)
        self.TriangleAttentionEndingNode = TriangleAttentionEndingNode(config['TriangleAttentionEndingNode'], global_config)
        self.TriangleMultiplicationOutgoing = TriangleMultiplicationOutgoing(config['TriangleMultiplicationOutgoing'], global_config)
        self.TriangleMultiplicationIngoing = TriangleMultiplicationIngoing(config['TriangleMultiplicationIngoing'], global_config)
        self.PairTransition = Transition(global_config['rep_2d']['num_c'], config['PairTransition']['n'])

    def forward(self, x2d):
        x2d += self.TriangleAttentionStartingNode(x2d.clone())
        x2d += self.TriangleAttentionEndingNode(x2d.clone())
        x2d += self.TriangleMultiplicationOutgoing(x2d.clone())
        x2d += self.TriangleMultiplicationIngoing(x2d.clone())
        x2d += self.PairTransition(x2d.clone())
        return x2d


class TemplatePairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.t2d_proj = nn.Linear(global_config['template']['num_feats'], global_config['rep_2d']['num_c'])
        self.layers = nn.ModuleList([TemplatePairStackIteration(config['TemplatePairStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.norm = nn.LayerNorm(global_config['rep_2d']['num_c'])

    def forward(self, t_feats):
        x = self.t2d_proj(t_feats)
        shape = x.shape
        x = x.flatten(end_dim=1)

        for l in self.layers:
            x = l(x)
        x = self.norm(x)

        x = x.view(*shape)
        return x


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
        self.t2d_proj = nn.Linear(global_config['cep_in_c'], global_config['rep_2d']['num_c'])
        self.layers = nn.ModuleList([TemplatePairStackIteration(config['TemplatePairStackIteration'], global_config) for _ in range(config['num_iter'])])
        self.norm = nn.LayerNorm(global_config['rep_2d']['num_c'])

    def forward(self, t_feats, t_crops, t_maps):
        x = self.t2d_proj(t_feats)
        #assert t_feats.shape[0] == 1
        x = x.flatten(end_dim=1)
        t_crops = t_crops.flatten(end_dim=1)
        t_maps = t_maps.flatten(end_dim=1)

        out_list = []
        assert x.shape[0] == t_crops.shape[0]
        for t_id, (num_res, num_atoms) in enumerate(t_crops):
            t_size = num_res + num_atoms
            t_map = t_maps[t_id]
            temp = x[t_id:t_id+1, :t_size, :t_size]
            for l in self.layers:
                temp = l(temp.clone())
            temp = self.norm(temp)[0, num_res:, num_res:]

            t_out = torch.zeros((t_map.shape[0], t_map.shape[0], temp.shape[2]))
            r_atoms = torch.where(t_map > -1)[0]
            t_atoms = t_map[t_map > -1]
            r_prod = torch.cartesian_prod(r_atoms, r_atoms)
            t_prod = torch.cartesian_prod(t_atoms, t_atoms)
            t_out[r_prod[:, 0], r_prod[:, 1]] = temp[t_prod[:, 0], t_prod[:, 1]]
            out_list.append(t_out)

        out = torch.stack(out_list)
        out = out.view(t_feats.shape[0], t_feats.shape[1], *out.shape[1:])
        return out


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
        self.layers = nn.ModuleList([EvoformerIteration(config['EvoformerIteration'], global_config) for x in range(config['num_blocks'])])

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class InitPairRepresentation(torch.nn.Module):
    def __init__(self, global_config):
        super().__init__()

        lig_in_c = global_config['lig_in_c']
        rec_in_c = global_config['rec_in_c']
        lig_in2d_c = global_config['lig_in2d_c']
        rec_in2d_c = global_config['rec_in2d_c']
        pair_num_c = global_config['rep_2d']['num_c']

        self.l_proj1 = nn.Linear(lig_in_c, pair_num_c)
        self.l_proj2 = nn.Linear(lig_in_c, pair_num_c)
        self.r_proj1 = nn.Linear(rec_in_c, pair_num_c)
        self.r_proj2 = nn.Linear(rec_in_c, pair_num_c)

        self.l2d_proj = nn.Linear(lig_in2d_c, pair_num_c)
        self.r2d_proj = nn.Linear(rec_in2d_c, pair_num_c)

    def forward(self, feats):
        l1d, l2d, r1d, r2d = feats['l1d'], feats['l2d'], feats['r1d'], feats['r2d']

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
        cep_in_c = global_config['cep_in_c']
        l1d_num_c = global_config['rep_1d']['num_c']
        r1d_num_c = global_config['rep_1d']['num_c']

        self.l_feat = nn.Linear(lig_in_c, l1d_num_c)
        self.cep_feat = nn.Linear(cep_in_c, l1d_num_c)
        self.r_feat = nn.Linear(rec_in_c, r1d_num_c)

        self.InitPairRepresentation = InitPairRepresentation(global_config)
        self.CEPPairStack = CEPPairStack(config['CEPPairStack'], global_config)
        self.CEPPointwiseAttention = TemplatePointwiseAttention(config['CEPPointwiseAttention'], global_config)
        self.TemplatePairStack = TemplatePairStack(config['TemplatePairStack'], global_config)
        self.TemplatePointwiseAttention = TemplatePointwiseAttention(config['TemplatePointwiseAttention'], global_config)

    def forward(self, feats):
        # create pair representation
        pair = self.InitPairRepresentation(feats)

        # CEP embedding
        num_res = feats['r1d'].shape[1]
        cep_2d = self.CEPPairStack(feats['cep_feats_2d'], feats['cep_crops'], feats['cep_maps'])
        cep_1d = cep_2d.mean(3)
        cep_embedding = self.CEPPointwiseAttention(pair[:, num_res:, num_res:], cep_2d)

        # make template embedding
        t2d = self.TemplatePairStack(feats['template_feats_2d'])
        template_embedding = self.TemplatePointwiseAttention(pair.clone(), t2d)

        # add embeddings to the pair rep
        pair += template_embedding
        pair[:, num_res:, num_res:] += cep_embedding

        r1d_rep = self.r_feat(feats['r1d'])
        # add linear rec template feats

        # make 1d rep
        l_feat = self.l_feat(feats['l1d'])
        rep_1d = cep_1d + l_feat.unsqueeze(1)
        # concat linear lig template feats

        return {'l1d': rep_1d, 'r1d': r1d_rep, 'pair': pair}


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


if __name__ == '__main__':
    example()
