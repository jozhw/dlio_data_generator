import os
from pathlib import Path


class Removing:

    @staticmethod
    def remove_compressed_imgs(filepath: Path):
        try:
            os.unlink(filepath)
        except OSError as e:
            print(f"Error: {filepath} : {e.strerror}")
