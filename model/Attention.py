import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
    

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class LinearCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)


        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim*2, bias = False)

        self.q_scale = nn.Parameter(torch.randn(dim_head))
        self.k_scale = nn.Parameter(torch.randn(dim_head))

        self.norm = nn.LayerNorm(inner_dim)
        self.norm_context = nn.LayerNorm(inner_dim) if norm_context else nn.Identity()
        
        self.to_out = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, dim, bias = False)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q = self.norm(q)
        k = self.norm_context(k)
        v = self.norm_context(v)
        # print(q.shape, k.shape, v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if mask is not None:
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)
        

    def forward(self, t):
        return self.embedding(t)

