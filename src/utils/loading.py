import json

from pathlib import Path


def load_config(config_path: Path) -> dict:
    """Read JSON configuration file"""
    try:
        with open(config_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        return {}
    except json.JSONDecodeError:
        print("Invalid JSON format in config file")
        return {}
