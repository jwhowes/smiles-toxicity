import os
from typing import Tuple, Optional, Literal

import numpy as np
import torch
from torch import nn, Tensor

from .base import TrainerConfig
from ..data import ToxicityDataset, ToxicityDatasetConfig
from ..model import ToxicityTransformer

from .base import BaseTrainer


class ToxicityCriterion(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None):
        super(ToxicityCriterion, self).__init__()
        if pos_weight is None:
            weight = None
        else:
            weight = torch.tensor([1.0, pos_weight])

        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred: Tensor, label: Tensor):
        return self.cross_entropy(pred, label)


class ToxicityTrainerConfig(TrainerConfig):
    pos_weight: Optional[Literal["auto"] | float] = "auto"


class ToxicityTrainer(BaseTrainer):
    data_config_cls = ToxicityDatasetConfig

    def get_model(self) -> ToxicityTransformer:
        return ToxicityTransformer.from_config(self.model_config)

    def get_criterion(self) -> ToxicityCriterion:
        if self.train_config.pos_weight is None:
            return ToxicityCriterion()

        if self.train_config.pos_weight == "auto":
            with open(
                    os.path.join(os.environ["DATA_DIR"], self.data_config.data_dir, "NR-ER-train/names_labels.csv")
            ) as f:
                labels = [int(line.split(",")[-1]) for line in f.read().splitlines()]

            counts = np.bincount(labels)
            p = counts / counts.sum()

            pos_weight = p[0] / p[1]
        else:
            pos_weight = self.train_config.pos_weight

        return ToxicityCriterion(pos_weight=pos_weight)

    def get_datasets(self) -> Tuple[ToxicityDataset, ToxicityDataset]:
        return (
            ToxicityDataset(
                os.path.join(os.environ["DATA_DIR"], self.data_config.data_dir, "NR-ER-train")
            ),
            ToxicityDataset(
                os.path.join(os.environ["DATA_DIR"], self.data_config.data_dir, "NR-ER-test")
            )
        )
