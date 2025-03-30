import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from calculations import Calculations
from filenaming import FileNaming
from removing import Removing


class Compression:

    @staticmethod
    def _compress_to_jpg(compressed_filepath: Path, image: np.ndarray) -> None:
        try:
            img = Image.fromarray(image.astype("uint8"), "RGB")
            # quality to 95 as that is recommend from the docs for the highest image res
            img.save(compressed_filepath, quality=95)
        except Exception as e:
            print("Error: {}".format(e))

    @staticmethod
    def _compress_to_npz(compressed_filepath: Path, image: np.ndarray) -> None:
        try:
            np.savez_compressed(compressed_filepath, image)
        except Exception as e:
            print("Error: {}".format(e))

    @staticmethod
    def _compression_types_wrapper(
        file_type: str,
        filename: str,
        image: np.ndarray,
        path: str,
    ) -> Path:
        if file_type == "npz":

            compressed_image_path = FileNaming.generate_raw_filename(
                path,
                filename,
                "npz",
            )
            Compression._compress_to_npz(compressed_image_path, image)
        elif file_type == "jpg":
            compressed_image_path = FileNaming.generate_raw_filename(
                path,
                filename,
                "jpg",
            )
            Compression._compress_to_jpg(compressed_image_path, image)
        else:
            raise ValueError(
                "The given compression type is not supported: {}".format(file_type)
            )
        # put other compression algorithms here

        return compressed_image_path

    @staticmethod
    def compress_and_calculate(
        filename: str,
        image: np.ndarray,
        dimensions: Tuple,
        compression_types: List[str],
        path,
        remove: bool = True,
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {}

        for compression_type in compression_types:

            # the label for compressed file size in bytes
            key_csize: str = "{}_compressed_image_size".format(compression_type)

            # the label for the calculated compression ratio
            key_compression_ratio: str = "{}_compression_ratio".format(compression_type)

            # run the wrapper, where the compressed image will be generated
            # for the specified compression_type
            compressed_path: Path = Compression._compression_types_wrapper(
                compression_type, filename, image, path
            )

            # calculate the compression ratio
            compression_ratio: float = Calculations.calculate_compression_ratio(
                compressed_path, dimensions
            )

            # in bytes
            compressed_size: int = os.path.getsize(compressed_path)

            # save the calculations to the results object
            result[key_csize] = compressed_size
            result[key_compression_ratio] = compression_ratio

            if remove:
                Removing.remove_compressed_imgs(compressed_path)

        return result

    @staticmethod
    def compress(
        filename: str,
        image: np.ndarray,
        compression_types: List[str],
        path,
        remove: bool = True,
    ) -> int:

        for compression_type in compression_types:

            # run the wrapper, where the compressed image will be generated
            # for the specified compression_type
            compressed_path: Path = Compression._compression_types_wrapper(
                compression_type, filename, image, path
            )

            if remove:
                Removing.remove_compressed_imgs(compressed_path)

        return 0
