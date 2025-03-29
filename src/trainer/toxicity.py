from torch import nn, Tensor

from ..data import ToxicityDataset
from ..model import ToxicityTransformer

from .base import BaseTrainer


class ToxicityCriterion(nn.Module):
    def __init__(self):
        super(ToxicityCriterion, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, label: Tensor):
        return self.cross_entropy(pred, label)


class ToxicityTrainer(BaseTrainer):
    model_cls = ToxicityTransformer
    criterion_cls = ToxicityCriterion
    dataset_cls = ToxicityDataset
