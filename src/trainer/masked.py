from torch import nn, Tensor

from ..model import MaskedTransformer
from ..data import MaskedPretrainDataset
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
    model_cls = MaskedTransformer
    criterion_cls = MaskedCriterion
    dataset_cls = MaskedPretrainDataset
