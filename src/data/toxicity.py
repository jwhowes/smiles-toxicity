from __future__ import annotations

import os
from typing import Tuple, List

import torch
from transformers import BatchEncoding

from ..tokenizer import SMILESTokenizer

from .base import BaseDataset, DatasetConfig


class ToxicityDatasetConfig(DatasetConfig):
    data_dir: str


class ToxicityDataset(BaseDataset):
    def __init__(self, data_dir: str):
        assert os.path.isdir(data_dir), f"Directory {data_dir} not found."

        self.tokenizer = SMILESTokenizer()

        with open(os.path.join(data_dir, "names_smiles.csv")) as f:
            self.seqs = [line.split(",")[-1] for line in f.read().splitlines()]

        with open(os.path.join(data_dir, "names_labels.csv")) as f:
            self.labels = [int(line.split(",")[-1]) for line in f.read().splitlines()]

        assert len(self.seqs) == len(self.labels), "Size mismatch."

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.seqs[idx], self.labels[idx]

    def collate(self, batch: List[Tuple[str, int]]) -> Tuple[BatchEncoding, BatchEncoding]:
        seqs, labels = zip(*batch)

        return (
            BatchEncoding(self.tokenizer(seqs)),
            BatchEncoding({
                "label": torch.tensor(labels, dtype=torch.long)
            })
        )
