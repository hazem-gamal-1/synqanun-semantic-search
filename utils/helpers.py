import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.yaml"))
@lru_cache
def get_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)