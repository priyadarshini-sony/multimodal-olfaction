import random
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum
random.seed(2022)

class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads       ##32(4*8)   
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias = False)   ##better to dim to dim*3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        #v = k = q = self.to_qkv(x)
        
        #q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        #k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        #v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):           ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                #Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                #FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# TRANS(image_size,time_size,fre_size,l_target, dim, depth, heads, mlp_dim,dim_head)
class TRANS(nn.Module):
    # def __init__(self,*, image_size, time_size, fre_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls' , dim_head, dropout = 0., emb_dropout = 0.):
    def __init__(self, image_size, time_size, fre_size, num_classes, dim, depth, heads, mlp_dim, dim_head, pool = 'cls' , dropout = 0., emb_dropout = 0.):
        super().__init__()
        num_patches = image_size     
        patch_dim = time_size * fre_size   
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(patch_dim, dim)       ##4 to dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) ##Generate the pos value'
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                ##Generate the class value'
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x

