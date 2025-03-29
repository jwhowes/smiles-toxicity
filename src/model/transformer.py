from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

from ..tokenizer import SMILESTokenizer

from .util import RMSNorm, SwiGLU


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0, "Hidden size must be divisible by the number of heads."

        self.n_heads = n_heads
        self.scale = sqrt(d_model / n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
        return torch.view_as_real(
            torch.view_as_complex(
                x.unflatten(-1, (-1, 2))
            ) *
            freqs
        ).flatten(-2)

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, L, _ = x.shape

        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (
            self.apply_rope(q, freqs) @
            self.apply_rope(k, freqs).transpose(-2, -1)
        ) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(B, 1, -1, L)

        return self.W_o(
            rearrange(
                self.dropout(F.softmax(attn, dim=-1)) @ v,
                "b n l d -> b l (n d)"
            )
        )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, ffn_dropout: float = 0.0):
        super(TransformerBlock, self).__init__()
        self.attn = Attention(d_model, n_heads, attn_dropout)
        self.attn_norm = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            RMSNorm(d_model),
            SwiGLU(d_model),
            nn.Dropout(ffn_dropout)
        )

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, attention_mask)

        return x + self.ffn(x)


class Transformer(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, max_length: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(Transformer, self).__init__()
        assert d_model % n_heads == 0, "Hidden size must be divisible by the number of heads."
        assert (d_model // n_heads) % 2 == 0, "Head dimension must be even."

        d_head = d_model // n_heads
        theta = 1.0 / (1e4 ** (2 * torch.arange(d_head // 2) / d_head))
        freqs = torch.outer(torch.arange(max_length), theta)
        self.register_buffer(
            "freqs",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, attn_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        L = x.shape[1]

        for layer in self.layers:
            x = layer(x, self.freqs[:L], attention_mask)

        return x


class SMILESTransformer(Transformer):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, max_length: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(SMILESTransformer, self).__init__(
            d_model, n_heads, n_layers, max_length, attn_dropout, ffn_dropout
        )

        self.emb = nn.Embedding(SMILESTokenizer.vocab_size, d_model, padding_idx=SMILESTokenizer.pad_token)

    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return super(SMILESTransformer, self).forward(self.emb(token_ids), attention_mask)

