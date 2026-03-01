import yaml
from pathlib import Path
from functools import lru_cache
from app.schemas.settings_schema import Settings


APP_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_yaml_file(file_path: Path) -> dict:
    """Load one YAML file and enforce a top-level mapping structure."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top-level: {file_path}")
    return data


@lru_cache()
def get_settings() -> Settings:
    """Load and cache merged application settings from the config directory."""
    config_files = sorted(APP_CONFIG_DIR.glob("*_config.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config files found in {APP_CONFIG_DIR}")

    data = {}
    for config_file in config_files:
        data.update(_load_yaml_file(config_file))

    return Settings(**data)


@lru_cache()
def get_prompts() -> dict:
    """Load and cache prompt definitions from the prompt config file."""
    prompt_path = APP_CONFIG_DIR / "prompt.yaml"
    return _load_yaml_file(prompt_path)
