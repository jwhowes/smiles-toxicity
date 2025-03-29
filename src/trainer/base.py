from abc import ABC
from typing import Type, Callable
from dataclasses import dataclass

import torch.optim
from torch import Tensor
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..config import Config
from ..data.base import BaseDataset, DatasetConfig
from ..model import BaseModel, ModelConfig


accelerator = Accelerator()


@dataclass
class TrainerConfig(Config):
    lr: float = 5e-5
    weight_decay: float = 0.05


class BaseTrainer(ABC):
    model_cls: Type[BaseModel]
    criterion_cls: Type[Callable[[Tensor, ...], Tensor]]
    dataset_cls: Type[BaseDataset]

    @classmethod
    def train(cls, model_config: ModelConfig, data_config: DatasetConfig, train_config: TrainerConfig):
        model = cls.model_cls.from_config(model_config)

        dataset = cls.dataset_cls.from_config(data_config)
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=dataset.collate
        )

        criterion = cls.criterion_cls()

        opt = torch.optim.AdamW(
            model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
        )

        model, dataloader, criterion, opt = accelerator.prepare(
            model, dataloader, criterion, opt
        )

        for i, (input, target) in enumerate(dataloader):
            opt.zero_grad()

            pred = model(**input)

            loss = criterion(pred, **target)

            accelerator.backward(loss)
            opt.step()

            if accelerator.is_main_process:
                print(i)
