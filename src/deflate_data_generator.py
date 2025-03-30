import math
import joblib
import json

import numpy as np

from pathlib import Path
from scipy.stats import entropy, norm

from typing import Dict


"""
Class of methods to generate synthetic images using the deflate algorithm (lempel-ziv 77)
"""


class DeflateDataGenerator:

    inverse_path: Path = Path("./data/models/pretrained/deflate-inverse-model-v1.pk1")

    nls_path: Path = Path("./data/models/pretrained/deflate_nls_model-v1.json")

    @staticmethod
    def load_models(inverse_path: Path, nls_path: Path) -> Dict:
        """
        To be used in the DataGenerator
        """

        # inverse_model
        def _inverse_model(x, a, h, k):
            return a / (x - h) + k

        # nls_model
        def _nls_model(x, a, b, c, d):
            return a / (x + b) + c * np.log(c) + d * x**2

        # inverse model params
        inv_data = joblib.load(inverse_path)
        inv_params = inv_data["params"]
        inv_a = inv_params["a"]
        inv_h = inv_params["h"]
        inv_k = inv_params["k"]

        # nls model params
        with open(nls_path, "r") as f:
            nls_data = json.load(f)

        nls_a = nls_data["params"]["a"]
        nls_b = nls_data["params"]["b"]
        nls_c = nls_data["params"]["c"]
        nls_d = nls_data["params"]["d"]

        models = {
            "inverse_model": {
                "function": _inverse_model,
                "params": [inv_a, inv_h, inv_k],
            },
            "nls_model": {
                "function": _nls_model,
                "params": [nls_a, nls_b, nls_c, nls_d],
            },
        }

        return models

    @staticmethod
    def _calculate_entropy(compression_ratio, models: Dict, boundary=1.984) -> float:

        # use inverse_model
        if compression_ratio <= boundary:
            inverse_model = models["inverse_model"]["function"]
            params = models["inverse_model"]["params"]

            return inverse_model(compression_ratio, *params)

        # use nls_model
        else:
            nls_model = models["nls_model"]["function"]
            params = models["nls_model"]["params"]

            return nls_model(compression_ratio, *params)

    @staticmethod
    def _generate_pdf(pdx=np.arange(256), mean: float = 127.0, std: float = 30.0):
        """
        Since using 8bit pixels, there are 256 total values (0-255). The mean from previous experiments have shown little distinction in deflate compression ratios.
        """
        # gen probability density function
        pdf = norm.pdf(pdx, mean, std)
        pdf /= pdf.sum()  # normalize the probabilities to sum to 1
        return pdf

    @staticmethod
    def _pdf_calculate_entropy(pdf):
        # calculate shannon entropy using the probability density function
        shannon_ent = entropy(pdf, base=2)

        return shannon_ent

    @staticmethod
    def _calculate_std(
        target_entropy,
        lower_bound: float = 0.01,
        upper_bound: float = 5000.0,
        tolerance: float = 1e-6,
    ):
        if math.floor(target_entropy) > 8:
            raise ValueError("RGB image entropy cannot be greater than 8.")

        while abs(upper_bound - lower_bound) > tolerance:
            mid_std = (lower_bound + upper_bound) / 2
            pdf = DeflateDataGenerator._generate_pdf(std=mid_std)
            mid_entropy = DeflateDataGenerator._pdf_calculate_entropy(pdf)

            if mid_entropy < target_entropy:
                lower_bound = mid_std
            else:
                upper_bound = mid_std

        return (lower_bound + upper_bound) / 2

    @staticmethod
    def _generate_intensity_values(std, size: int) -> np.ndarray:
        """
        Generate a NumPy array of the given size with random integer values from a normal distribution
        with the specified mean and standard deviation, within the range [0, 255].
        """
        # calculate the probabilities for each value in the range [0, 255]
        pdx = np.arange(0, 256)

        pdf = DeflateDataGenerator._generate_pdf(std=std)

        values = np.random.choice(pdx, size=size, p=pdf)

        return values.astype(np.uint8)

    @staticmethod
    def generate_synthetic_image(compression_ratio: float, xside: int, models: Dict):

        entropy = DeflateDataGenerator._calculate_entropy(compression_ratio, models)

        std = DeflateDataGenerator._calculate_std(entropy)

        # will create a square image
        size = (xside**2) * 3
        raw_synthetic_image = DeflateDataGenerator._generate_intensity_values(std, size)

        # assume three channels
        synthetic_image = raw_synthetic_image.reshape((xside, xside, 3))

        return synthetic_image
