from typing import Type

import click
from dotenv import load_dotenv

from src.trainer import MaskedTrainer
from src.trainer.base import BaseTrainer
from src.trainer.toxicity import ToxicityTrainer


@click.group()
@click.argument("config-dir", type=click.Path(exists=True, file_okay=False))
@click.pass_context
def cli(ctx: click.Context, config_dir: str):
    assert load_dotenv(".env"), ".env file not found."

    ctx.ensure_object(dict)

    ctx.obj["config_dir"] = config_dir


def training_task(trainer_cls: Type[BaseTrainer], command_name: str):
    @cli.command(command_name)
    @click.pass_context
    def process(ctx: click.Context):
        trainer_cls(ctx.obj["config_dir"]).train()
    
    return process


training_task(MaskedTrainer, "masked-pretrain")
training_task(ToxicityTrainer, "toxicity")


if __name__ == "__main__":
    cli()
