from typing import Any, Sequence, Optional

import yaml


def read_yaml_config(yaml_config_path: str) -> dict[str, Any]:
    try:
        with open(yaml_config_path) as conf:
            config = yaml.safe_load(conf)
            return config
    except FileNotFoundError:
        return {}


def get_value_from_config(config: dict[str, Any], keys: Sequence[str], default: Any = None) -> Optional[Any]:
    for idx, key in enumerate(keys[:-1]):
        config = config.get(key, {})
    return config.get(keys[-1], default)
