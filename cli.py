import os

import click

from src.model import ModelConfig
from src.data import MaskedPretrainDatasetConfig
from src.trainer import TrainerConfig, MaskedTrainer


@click.group()
@click.argument("config-path", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def cli(ctx: click.Context, config_path: str):
    ctx.ensure_object(dict)

    ctx.obj["config_path"] = config_path


@cli.command("masked-pretrain")
@click.pass_context
def masked_pretrain(ctx: click.Context):
    model_config = ModelConfig.from_yaml(os.path.join(ctx.obj["config_path"], "model.yaml"))
    data_config = MaskedPretrainDatasetConfig.from_yaml(os.path.join(ctx.obj["config_path"], "data.yaml"))
    train_config = TrainerConfig.from_yaml(os.path.join(ctx.obj["config_path"], "train.yaml"))

    MaskedTrainer.train(
        model_config, data_config, train_config
    )


if __name__ == "__main__":
    cli()
