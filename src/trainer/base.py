import os.path
from abc import ABC
from typing import Type, Callable
from dataclasses import dataclass
from datetime import datetime

import torch
import yaml
from torch import Tensor
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..config import Config
from ..data.base import BaseDataset, DatasetConfig
from ..model import BaseModel, ModelConfig


accelerator = Accelerator()


@dataclass
class TrainerConfig(Config):
    lr: float = 5e-5
    weight_decay: float = 0.05

    log_interval: int = 100
    save_interval: int = 1000


class BaseTrainer(ABC):
    model_cls: Type[BaseModel]
    criterion_cls: Type[Callable[[Tensor, ...], Tensor]]
    dataset_cls: Type[BaseDataset]

    @classmethod
    def train(cls, exp_name: str, model_config: ModelConfig, data_config: DatasetConfig, train_config: TrainerConfig):
        ckpt = 1
        total_loss = 0
        prev_save = 0

        def save():
            nonlocal ckpt, total_loss, prev_save
            avg_loss = total_loss / (i - prev_save)

            with open(log_path, "a") as f:
                f.write(f"{ckpt},{avg_loss:.4f},{datetime.now()}\n")

            torch.save(
                accelerator.get_state_dict(model),
                os.path.join(ckpt_dir, f"checkpoint_{ckpt:04}.pt")
            )

            prev_save = i
            total_loss = 0
            ckpt += 1

        exp_dir = os.path.join("experiments", exp_name)
        ckpt_dir = os.path.join(exp_dir, "ckpts")
        log_path = os.path.join(exp_dir, "log.csv")

        if accelerator.is_main_process:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)

            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)

            with open(log_path, "w+") as f:
                f.write("ckpt,avg loss,timestamp\n")

            with open(os.path.join(exp_dir, "model.yaml"), "w+") as f:
                yaml.dump(model_config, f)

            with open(os.path.join(exp_dir, "data.yaml"), "w+") as f:
                yaml.dump(data_config, f)

            with open(os.path.join(exp_dir, "train.yaml"), "w+") as f:
                yaml.dump(train_config, f)

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
        lr_scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=len(dataloader)
        )

        model, dataloader, criterion, opt, lr_scheduler = accelerator.prepare(
            model, dataloader, criterion, opt, lr_scheduler
        )

        total_loss = 0
        prev_save = 0
        for i, (input, target) in enumerate(dataloader):
            opt.zero_grad()

            pred = model(**input)

            loss = criterion(pred, **target)

            accelerator.backward(loss)

            opt.step()
            lr_scheduler.step()

            total_loss += loss.item()

            if accelerator.is_main_process and i % train_config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\t Loss: {loss.item():.4f}")

            if accelerator.is_main_process and i > 0 and i % train_config.save_interval == 0:
                save()

        if accelerator.is_main_process:
            save()
