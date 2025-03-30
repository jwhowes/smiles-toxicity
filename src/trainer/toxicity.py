import os
from typing import Tuple

from torch import nn, Tensor
from torch.utils.data import DataLoader

from ..data import ToxicityDataset, ToxicityDatasetConfig
from ..model import ToxicityTransformer

from .base import BaseTrainer


class ToxicityCriterion(nn.Module):
    def __init__(self):
        super(ToxicityCriterion, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, label: Tensor):
        return self.cross_entropy(pred, label)


class ToxicityTrainer(BaseTrainer):
    data_config_cls = ToxicityDatasetConfig

    def get_model(self) -> ToxicityTransformer:
        return ToxicityTransformer.from_config(self.model_config)

    def get_criterion(self) -> ToxicityCriterion:
        return ToxicityCriterion()

    def get_datasets(self) -> Tuple[ToxicityDataset, ToxicityDataset]:
        return (
            ToxicityDataset(
                os.path.join(os.environ["DATA_DIR"], self.data_config.data_dir, "NR-ER-train")
            ),
            ToxicityDataset(
                os.path.join(os.environ["DATA_DIR"], self.data_config.data_dir, "NR-ER-test")
            )
        )
