from __future__ import annotations

from typing import Optional, Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import nn, Tensor

from ..tokenizer import SMILESTokenizer
from ..config import Config

from .util import RMSNorm
from .transformer import SMILESTransformer


@dataclass
class ModelConfig(Config):
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12

    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0


class BaseModel(nn.Module, ABC):
    @classmethod
    def from_config(cls, config: ModelConfig) -> Self:
        return cls(
            max_length=SMILESTokenizer.max_length,
            **config.__dict__
        )

    @abstractmethod
    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        ...


class MaskedTransformer(SMILESTransformer, BaseModel):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, max_length: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(MaskedTransformer, self).__init__(
            d_model, n_heads, n_layers, max_length, attn_dropout, ffn_dropout
        )

        self.head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, SMILESTokenizer.vocab_size)
        )

    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = super(MaskedTransformer, self).forward(token_ids, attention_mask)

        return self.head(x)
