import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class simple_MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        return self.layers(x)

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=1,
        attn_dropout=0.,
        ff_dropout=0.,
        lastmlp_dropout=0.,
        cont_embeddings='MLP',
        attentiontype='col'
    ):
        super().__init__()
        self.dim = dim
        self.num_categories = len(categories)
        self.total_tokens = sum(categories) + num_special_tokens
        self.embeds = nn.Embedding(self.total_tokens, dim)

        # Continuous embeddings
        self.cont_embeddings = cont_embeddings
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, dim]) for _ in range(num_continuous)])

        # Transformer
        self.transformer = Transformer(
            num_tokens=self.total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # Correctly calculate the Transformer output dimensions
        transformer_output_dim = dim * (len(categories) + num_continuous)
        hidden_dimensions = [transformer_output_dim, *map(lambda t: dim * t, mlp_hidden_mults), dim_out]
        self.mlp = simple_MLP(hidden_dimensions)

    def forward(self, x_categ, x_cont):
        # Encode categorical data
        x_categ_enc = self.embeds(x_categ)

        # Encode continuous data
        if self.cont_embeddings == 'MLP':
            x_cont_enc = torch.cat([mlp(x_cont[:, i:i+1]) for i, mlp in enumerate(self.simple_MLP)], dim=1)

        # Combine tensors
        x = torch.cat((x_categ_enc, x_cont_enc), dim=1)

        # Transformer
        x = self.transformer(x)
        x = x.flatten(1)
        return self.mlp(x)
