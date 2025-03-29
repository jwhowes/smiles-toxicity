from typing import Optional

from torch import nn, Tensor

from ..tokenizer import SMILESTokenizer

from .transformer import Transformer


class SMILESTransformer(Transformer):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, max_length: int,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(SMILESTransformer, self).__init__(
            d_model, n_heads, n_layers, max_length, attn_dropout, ffn_dropout
        )

        self.emb = nn.Embedding(SMILESTokenizer.vocab_size, d_model, padding_idx=SMILESTokenizer.pad_token)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return super(SMILESTransformer, self).forward(self.emb(x), attention_mask)
