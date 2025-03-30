import math
import os
from pathlib import Path
from typing import Dict

import numpy as np


class Calculations:

    @staticmethod
    def count_occurrences(image_data: np.ndarray) -> Dict:
        occurrences: Dict = {}
        for row in image_data:
            for pixel in row:
                for value in pixel:
                    if value in occurrences:
                        occurrences[value] += 1
                    else:
                        occurrences[value] = 1
        return occurrences

    @staticmethod
    def calculate_entropy(occurrences: Dict) -> float:

        # the summation of all the occurrences for each intensity value will
        # give you the image size
        file_size: int = 0
        for value in occurrences.values():
            file_size += value

        entropy: float = 0
        for count in occurrences.values():
            if count != 0:
                entropy += -(count / file_size) * math.log2(count / file_size)
        return entropy

    @staticmethod
    def calculate_compression_ratio(compressed_filepath: Path, dimensions):
        # Get the size of the image file
        compressed_file_size: int = os.path.getsize(compressed_filepath)

        # Calculate the number of pixels in the image
        num_pixels: int = dimensions[0] * dimensions[1]

        num_dimensions: int = len(dimensions)

        # if three dimensions, know that there are more than one channel, non black and white image
        if num_dimensions == 3:
            num_channels: int = dimensions[2]
        # if there are two dimensions, know that it is a black and white image so set channels to 1
        elif num_dimensions == 2:
            num_channels: int = 1
        else:
            raise ValueError(
                "Invalid dimensions. Valid Dimensions are length 2 or 3, received {}".format(
                    num_dimensions
                )
            )

        # Calculate the compression ratio
        # must be greater than or equal to 1
        # equation is uncompressed/compressed
        compression_ratio: float = (num_pixels * num_channels) / compressed_file_size

        return compression_ratio

    @staticmethod
    def calculate_mean_intensity_value(occurrences: Dict) -> int:

        total_sum: int = 0
        total_count: int = 0

        for key, value in occurrences.items():

            total_sum += key * value
            total_count += value

            if total_count == 0:
                return 0

        # round mean
        mean: int = round(total_sum / total_count)
        return mean

    @staticmethod
    def calculate_squared_equivalent(dimensions) -> int:

        a = dimensions[0] * dimensions[1]

        rounded_square = int(math.sqrt(a))

        return rounded_square


class EntropyCalculator:

    def __init__(self, window_size: int, shift_size: int, image: np.ndarray):

        self.window_size = window_size
        self.shift_size = shift_size
        self.image = image
        self.flattened_image = image.flatten()
        self.flattened_image_length = len(self.flattened_image)
        self.dimensions = image.shape
        self.size_bytes = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        self.occurrences = {}

    def calculate_entropy(self):
        start = 0
        # non inclusive, should match the window size since the start is at 0
        end = self.window_size

        while end <= self.flattened_image_length:

            arr = self.flattened_image[start:end]
            key = tuple(arr)

            if key in self.occurrences:
                self.occurrences[key] += 1
            else:
                self.occurrences[key] = 1

            start += self.shift_size
            end += self.shift_size

        entropy = Calculations.calculate_entropy(self.occurrences)

        return entropy

    def run(self):

        entropy = self.calculate_entropy()

        save_path = Path("./generated_files/compressed_image.npz")

        np.savez_compressed(save_path, self.image)

        compressed_size = os.path.getsize(save_path)

        os.remove(save_path)

        cratio = self.size_bytes / compressed_size

        return (self.window_size, entropy, cratio)


class Entropy:

    @staticmethod
    def _calcPI(image):
        """calculating number of I"""
        N = len(image)
        P = {}
        for i in image:
            if str(i) in P.keys():
                P[str(i)] += 1.0 / N
            else:
                P[str(i)] = 1.0 / N
        return P

    @staticmethod
    def _calcPIJ(image):
        """calculating probility of [IJ], given I already"""
        N = len(image)
        PIJ = {}
        for i in range(N - 1):
            k = str(image[i]) + "-" + str(image[i + 1])
            if k in PIJ.keys():
                PIJ[k] += 1.0 / (N - 1)
            else:
                PIJ[k] = 1.0 / (N - 1)
        return PIJ

    @staticmethod
    def _calcPIJK(image):
        N = len(image)
        PIJK = {}
        for i in range(N - 2):
            k = str(image[i]) + "-" + str(image[i + 1]) + "-" + str(image[i + 2])
            if k in PIJK.keys():
                PIJK[k] += 1.0 / (N - 2)
            else:
                PIJK[k] = 1.0 / (N - 2)

        return PIJK

    @staticmethod
    def _calcE(P):
        e = 0.0
        for k in P.keys():
            p = P[k]
            if p > 0.0:
                e += -p * np.log2(p)
        return e

    @staticmethod
    def _F1(image):
        P = Entropy._calcPI(image)
        return Entropy._calcE(P)

    @staticmethod
    def _F2(image):
        PIJ = Entropy._calcPIJ(image)
        return Entropy._calcE(PIJ)

    @staticmethod
    def _F3(image):
        PIJK = Entropy._calcPIJK(image)
        return Entropy._calcE(PIJK)

    @staticmethod
    def E1(image):
        return Entropy._F1(image)

    @staticmethod
    def E2(image):
        return Entropy._F2(image) - Entropy._F1(image)

    @staticmethod
    def E3(image):
        return Entropy._F3(image) - Entropy._F2(image)


if __name__ == "__main__":

    from PIL import Image

    paths = [
        "./assets/test_images/test1.jpg",
        "./assets/test_images/test2.jpg",
        "./assets/test_images/test3.jpg",
        "./assets/test_images/test4.jpg",
        "./assets/test_images/test5.jpg",
    ]

    window_size = 15

    shift_size = 3

    for path in paths:

        image = np.array(Image.open(path)).astype(np.uint8)

        flattened_image = image.flatten()

        e = Entropy.E3(flattened_image)

        print(e)

        # ec = EntropyCalculator(window_size, shift_size, image)

        # results = ec.run()

        # print(results)

        # occurrences = Calculations.count_occurrences(image)

        # entropy = Calculations.calculate_entropy(occurrences)

        # confirm_result = (1, entropy)

        # print(confirm_result)
