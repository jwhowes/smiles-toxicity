import os
from abc import ABC
from typing import Self

import yaml

from pydantic import BaseModel


class Config(ABC, BaseModel):
    @classmethod
    def from_yaml(cls, yaml_path: str) -> Self:
        assert os.path.exists(yaml_path), "yaml path not found."

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        return cls.model_validate(config)

    def to_yaml(self, yaml_path: str):
        with open(yaml_path, "w+") as f:
            yaml.dump(self.model_dump(), f)
