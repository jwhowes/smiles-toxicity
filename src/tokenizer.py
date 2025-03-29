from typing import Mapping, TypedDict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor


class TokenizerOutput(TypedDict):
    token_ids: Tensor
    attention_mask: Tensor


alphabet = [
    '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '=', '@', 'A', 'B', 'C', 'D', 'F', 'G',
    'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'Y', 'Z',
    '[', '\\', ']', 'a', 'b', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'r',
    's', 't', 'u', 'y'
]

class SMILESTokenizer:
    pad_token: int = 0
    bos_token: int = 1
    mask_token: int = 2

    vocab: Mapping[str, int] = {
        c: i + 3 for i, c in enumerate(alphabet)
    }

    vocab_size: int = 3 + len(alphabet)

    max_length: int = 128

    def pad(self, token_ids: List[Tensor]) -> TokenizerOutput:
        return {
            "token_ids": pad_sequence(token_ids, batch_first=True, padding_value=self.pad_token),
            "attention_mask": pad_sequence([
                torch.zeros(t.shape[0]) for t in token_ids
            ], batch_first=True, padding_value=float('-inf'))
        }

    def __call__(self, seqs: List[str] | str) -> TokenizerOutput:
        if isinstance(seqs, str):
            seqs = [seqs]

        return self.pad([
            torch.tensor(
                ([self.bos_token] + [self.vocab[c] for c in seq])[:self.max_length], dtype=torch.long
            )
            for seq in seqs
        ])
