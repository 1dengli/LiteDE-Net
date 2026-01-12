import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_


class gfmm(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        pool_ratio=16,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.pool_ratio = pool_ratio

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Q, K : channel reduction (C -> 1 per head)
        self.q = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim, num_heads, bias=qkv_bias)

        # V : keep full channel
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # spatial reduction
        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # output projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W

        # ---------- tokenize ----------
        x_flat = x.flatten(2).transpose(1, 2)   # (B, N, C)

        # ---------- Query ----------
        q = self.q(x_flat)                      # (B, N, Head)
        q = q.view(B, N, self.num_heads)
        q = q.permute(0, 2, 1).unsqueeze(-1)    # (B, Head, N, 1)

        # ---------- Key / Value ----------
        x_pool = self.pool(x)                   # (B, C, H/r, W/r)
        x_pool = self.sr(x_pool)

        x_pool = x_pool.flatten(2).transpose(1, 2)  # (B, N', C)
        x_pool = self.norm(x_pool)
        x_pool = self.act(x_pool)

        k = self.k(x_pool)                      # (B, N', Head)
        k = k.view(B, -1, self.num_heads)
        k = k.permute(0, 2, 1).unsqueeze(-1)    # (B, Head, N', 1)

        v = self.v(x_pool)                      # (B, N', C)
        v = v.view(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)               # (B, Head, N', C_head)

        # ---------- Attention ----------
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)             # (B, Head, N, C_head)
        out = out.transpose(1, 2).reshape(B, N, C)

        # ---------- Projection ----------
        out = self.proj(out)
        out = self.proj_drop(out)

        # ---------- restore BCHW ----------
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

if __name__ == '__main__':
    B = 2
    C = 512
    H = 8
    W = 8

    num_heads = 8
    pool_ratio = 8

    x = torch.randn(B, C, H, W)

    cra = gfmm(
        dim=C,
        num_heads=num_heads,
        pool_ratio=pool_ratio
    )

    with torch.no_grad():
        out = cra(x)

    print("input :", x.shape)
    print("output:", out.shape)

