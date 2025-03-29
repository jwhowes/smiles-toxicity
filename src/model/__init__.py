from __future__ import annotations

import os
from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import yaml
from torch import nn, Tensor

from ..tokenizer import SMILESTokenizer
from ..config import Config

from .util import RMSNorm
from .transformer import SMILESTransformer


@dataclass
class PretrainedCheckpoint:
    experiment: str
    ckpt_num: int


@dataclass
class ModelConfig(Config):
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12

    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0

    pretrained: Optional[PretrainedCheckpoint] = None

    def __init__(
            self, pretrained: Optional[dict] = None,
            **kwargs
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if pretrained is not None:
            self.pretrained = PretrainedCheckpoint(**pretrained)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ModelConfig:
        assert os.path.exists(yaml_path), "yaml path not found."

        yaml.add_multi_constructor('!', cls.unknown)
        yaml.add_multi_constructor('tag:', cls.unknown)

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        if "pretrained" in config and config["pretrained"] is not None:
            pretrained_config = ModelConfig.from_yaml(os.path.join(
                "experiments", config["pretrained"]["experiment"], "model.yaml"
            ))
            pretrained_config.pretrained = config["pretrained"]

            return cls(**pretrained_config.__dict__)

        return cls(
            **{
                k: float(v) if cls.__dataclass_fields__[k].type == "float" else v for k, v in config.items()
            }
        )


class BaseModel(nn.Module, ABC):
    @classmethod
    def from_config(cls, config: ModelConfig, load_pretrained: bool = True) -> MaskedTransformer:
        model = cls(
            max_length=SMILESTokenizer.max_length,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            attn_dropout=config.attn_dropout,
            ffn_dropout=config.ffn_dropout
        )

        if load_pretrained:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        "experiments",
                        config.pretrained.experiment,
                        "ckpts",
                        f"checkpoint_{config.pretrained.ckpt_num:04}.pt"
                    ),
                    weights_only=True,
                    map_location="cpu"
                ),
                strict=False
            )

        return model

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


class ToxicityTransformer(SMILESTransformer, BaseModel):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, max_length: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(ToxicityTransformer, self).__init__(
            d_model, n_heads, n_layers, max_length, attn_dropout, ffn_dropout
        )

        self.classifier = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 2)
        )

    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = super(ToxicityTransformer, self).forward(token_ids, attention_mask)

        return self.classifier(x[:, 0])
