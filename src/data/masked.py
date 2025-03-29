from __future__ import annotations

import os
from typing import Tuple, List
from math import ceil
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from transformers import BatchEncoding

from ..tokenizer import SMILESTokenizer

from .base import BaseDataset, DatasetConfig


@dataclass(kw_only=True)
class MaskedPretrainDatasetConfig(DatasetConfig):
    data_path: str

    train_frac: float = 0.8
    mask_p: float = 0.15


class MaskedPretrainDataset(BaseDataset):
    def __init__(self, seqs: List[str], mask_p: float = 0.15):
        self.tokenizer = SMILESTokenizer()

        self.mask_p = mask_p
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq = self.seqs[idx]

        unmasked_ids = self.tokenizer(seq)["token_ids"].squeeze(0)

        L = unmasked_ids.shape[0]
        masked_ids = unmasked_ids.clone()
        mask_idxs = torch.randperm(L - 1)[:ceil(L * self.mask_p)] + 1  # Don't mask bos token
        masked_ids[mask_idxs] = self.tokenizer.mask_token

        return unmasked_ids, masked_ids

    def collate(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[BatchEncoding, BatchEncoding]:
        unmasked, masked = zip(*batch)

        return (
            BatchEncoding(self.tokenizer.pad(masked)),
            BatchEncoding({
                "unmasked": pad_sequence(unmasked, batch_first=True, padding_value=self.tokenizer.pad_token),
                "mask": pad_sequence([
                    seq == self.tokenizer.mask_token for seq in masked
                ], batch_first=True, padding_value=False)
            })
        )

    @classmethod
    def from_config(cls, config: MaskedPretrainDatasetConfig) -> Tuple[MaskedPretrainDataset, MaskedPretrainDataset]:
        assert os.path.exists(config.data_path), "Data path not found."

        with open(config.data_path) as f:
            seqs = f.read().splitlines()

        split_idx = ceil(len(seqs) * config.train_frac)

        return (
            MaskedPretrainDataset(
                seqs=seqs[:split_idx],
                mask_p=config.mask_p
            ),
            MaskedPretrainDataset(
                seqs=seqs[split_idx:],
                mask_p=config.mask_p
            )
        )
