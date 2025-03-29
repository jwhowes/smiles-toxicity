import os
from dataclasses import dataclass
from abc import ABC

import yaml


@dataclass
class Config(ABC):
    @staticmethod
    def unknown(loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            constructor = loader.__class__.construct_scalar
        elif isinstance(node, yaml.SequenceNode):
            constructor = loader.__class__.construct_sequence
        elif isinstance(node, yaml.MappingNode):
            constructor = loader.__class__.construct_mapping

        data = constructor(loader, node)

        return data

    @classmethod
    def from_yaml(cls, yaml_path: str):
        assert os.path.exists(yaml_path), "yaml path not found."

        yaml.add_multi_constructor('!', cls.unknown)
        yaml.add_multi_constructor('tag:', cls.unknown)

        with open(yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if config is None:
            return cls()

        return cls(
            **{
                k: float(v) if cls.__dataclass_fields__[k].type == "float" else v for k, v in config.items()
            }
        )
