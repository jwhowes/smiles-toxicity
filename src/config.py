import os
from dataclasses import dataclass
from abc import ABC

import yaml


@dataclass
class Config(ABC):
    @classmethod
    def from_yaml(cls, yaml_path: str):
        assert os.path.exists(yaml_path), "yaml path not found."

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        return cls(
            **{
                k: float(v) if cls.__dataclass_fields__[k].type == "float" else v for k, v in config.items()
            }
        )
