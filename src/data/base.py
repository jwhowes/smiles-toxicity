from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar

from transformers import BatchEncoding
from torch.utils.data import Dataset

from ..config import Config

T = TypeVar("T")


class DatasetConfig(Config):
    batch_size: int = 32


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def collate(self, batch: List[T]) -> Tuple[BatchEncoding, BatchEncoding]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        ...
