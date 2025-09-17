from threading import Lock
from typing import Optional

import yaml


class ConfigSingleton:
    _instance = None
    _lock: Lock = Lock()

    @classmethod
    def get_instance(cls, path: Optional[str] = "config/app_config.yaml"):
        with cls._lock:
            if cls._instance is None:
                if not path:
                    raise ValueError(
                        "First call to get_instance must include a config file path.")
                with open('config/app_config.yaml', "r") as f:
                    raw = yaml.safe_load(f)
                cls._instance = raw
            return cls._instance


if __name__ == "__main__":
    config = ConfigSingleton.get_instance("app_config.yaml")
    print(config)
