import os.path
from abc import ABC, abstractmethod
from typing import Type, Callable, Tuple
from datetime import datetime

import torch
from torch import Tensor
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..config import Config
from ..data.base import BaseDataset, DatasetConfig
from ..model import SMILESModel, ModelConfig


accelerator = Accelerator()


class TrainerConfig(Config):
    lr: float = 5e-5
    weight_decay: float = 0.05

    num_epochs: int = 5

    log_interval: int = 100
    save_interval: int = 1000


class BaseTrainer(ABC):
    model_config_cls: Type[ModelConfig] = ModelConfig
    data_config_cls: Type[DatasetConfig] = DatasetConfig
    train_config_cls: Type[TrainerConfig] = TrainerConfig

    def __init__(self, config_dir: str):
        self.exp_name = os.path.basename(config_dir)

        self.exp_dir = os.path.join(os.environ["EXP_DIR"], self.exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "ckpts")
        self.train_log_path = os.path.join(self.exp_dir, "log-train.csv")
        self.eval_log_path = os.path.join(self.exp_dir, "log-eval.csv")

        assert os.path.isdir(config_dir), f"Config directory {config_dir}, not found."

        model_config_path = os.path.join(config_dir, "model.yaml")
        assert os.path.exists(model_config_path), "Config directory is missing model.yaml"

        data_config_path = os.path.join(config_dir, "data.yaml")
        assert os.path.exists(data_config_path), "Config directory is missing data.yaml"

        train_config_path = os.path.join(config_dir, "train.yaml")
        assert os.path.exists(train_config_path), "Config directory is missing train.yaml"

        self.model_config = self.model_config_cls.from_yaml(model_config_path)
        self.data_config = self.data_config_cls.from_yaml(data_config_path)
        self.train_config = self.train_config_cls.from_yaml(train_config_path)

        if accelerator.is_main_process:
            if not os.path.isdir(self.exp_dir):
                os.makedirs(self.exp_dir)

            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

            self.model_config.to_yaml(os.path.join(self.exp_dir, "model.yaml"))
            self.data_config.to_yaml(os.path.join(self.exp_dir, "data.yaml"))
            self.train_config.to_yaml(os.path.join(self.exp_dir, "train.yaml"))

            with open(self.train_log_path, "w+") as f:
                f.write("ckpt,avg loss,timestamp\n")

            with open(self.eval_log_path, "w+") as f:
                f.write("epoch,avg loss,timestamp\n")

    @abstractmethod
    def get_model(self) -> SMILESModel:
        ...

    @abstractmethod
    def get_criterion(self) -> Callable[[Tensor, ...], Tensor]:
        ...

    @abstractmethod
    def get_datasets(self) -> Tuple[BaseDataset, BaseDataset]:
        ...

    def train(self):
        def save():
            nonlocal ckpt, total_loss, prev_save
            avg_loss = total_loss / (i - prev_save)

            with open(self.train_log_path, "a") as f:
                f.write(f"{ckpt},{avg_loss:.4f},{datetime.now()}\n")

            torch.save(
                accelerator.get_state_dict(model),
                os.path.join(self.ckpt_dir, f"checkpoint_{ckpt:04}.pt")
            )

            prev_save = i
            total_loss = 0
            ckpt += 1

        model = self.get_model()

        train_dataset, eval_dataset = self.get_datasets()

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.data_config.batch_size,
            collate_fn=train_dataset.collate
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.data_config.batch_size,
            collate_fn=eval_dataset.collate
        )

        criterion = self.get_criterion()

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.train_config.lr, weight_decay=self.train_config.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.train_config.num_epochs * len(train_dataloader)
        )

        model, train_dataloader, eval_dataloader, criterion, opt, lr_scheduler = accelerator.prepare(
            model, train_dataloader, eval_dataloader, criterion, opt, lr_scheduler
        )

        ckpt = 0
        for epoch in range(self.train_config.num_epochs):
            if accelerator.is_main_process:
                print(f"EPOCH {epoch + 1} / {self.train_config.num_epochs}")
                print("Training...")

            model.train()

            total_loss = 0
            prev_save = 0
            for i, (input, target) in enumerate(train_dataloader):
                opt.zero_grad()

                pred = model(**input)

                loss = criterion(pred, **target)

                accelerator.backward(loss)

                opt.step()
                lr_scheduler.step()

                total_loss += loss.item()

                if accelerator.is_main_process and i % self.train_config.log_interval == 0:
                    print(f"\t{i} / {len(train_dataloader)} iters.\t Loss: {loss.item():.4f}")

                if accelerator.is_main_process and i > 0 and i % self.train_config.save_interval == 0:
                    save()

            if accelerator.is_main_process:
                save()

            if accelerator.is_main_process:
                print("Evaluating...")

            model.eval()
            total_loss = 0
            for i, (input, target) in enumerate(eval_dataloader):
                with torch.no_grad():
                    pred = model(**input)
                    loss = criterion(pred, **target)

                total_loss += loss.item()

                if accelerator.is_main_process and i % self.train_config.log_interval == 0:
                    print(f"\t{i} / {len(eval_dataloader)} iters.\t Loss: {loss.item():.4f}")

            with open(self.eval_log_path, "a") as f:
                f.write(f"{epoch + 1},{total_loss / len(eval_dataloader):.4f},{datetime.now()}\n")
