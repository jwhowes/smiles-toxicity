import os
from math import ceil
from typing import Tuple

from torch import nn, Tensor

from ..model import MaskedTransformer
from ..data import MaskedPretrainDatasetConfig, MaskedPretrainDataset
from ..tokenizer import SMILESTokenizer

from .base import BaseTrainer


class MaskedCriterion(nn.Module):
    def __init__(self):
        super(MaskedCriterion, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none", ignore_index=SMILESTokenizer.pad_token)

    def forward(self, pred: Tensor, unmasked: Tensor, mask: Tensor) -> Tensor:
        loss = self.cross_entropy(pred.transpose(1, 2), unmasked)

        return loss[mask].sum() / mask.sum()


class MaskedTrainer(BaseTrainer):
    data_config_cls = MaskedPretrainDatasetConfig

    def get_model(self) -> MaskedTransformer:
        return MaskedTransformer.from_config(self.model_config)

    def get_criterion(self) -> MaskedCriterion:
        return MaskedCriterion()

    def get_datasets(self) -> Tuple[MaskedPretrainDataset, MaskedPretrainDataset]:
        assert os.path.exists(os.path.join(os.environ["DATA_DIR"], self.data_config.data_path)), "Data path not found."

        with open(os.path.join(os.environ["DATA_DIR"], self.data_config.data_path)) as f:
            seqs = f.read().splitlines()

        split_idx = ceil(len(seqs) * self.data_config.train_frac)

        return (
            MaskedPretrainDataset(
                seqs=seqs[:split_idx],
                mask_p=self.data_config.mask_p
            ),
            MaskedPretrainDataset(
                seqs=seqs[split_idx:],
                mask_p=self.data_config.mask_p
            )
        )
