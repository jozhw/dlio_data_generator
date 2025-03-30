from pathlib import Path

from data_generator import DataGenerator
from utils.loading import load_config


def main():

    config_path: Path = Path("./config/data_generator_config.json")

    config: dict = load_config(config_path)

    if config:
        generator: DataGenerator = DataGenerator(**config)

        generator.run()

    return 0


if __name__ == "__main__":

    main()
