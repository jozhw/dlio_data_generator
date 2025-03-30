from typing import List, Dict

import json

from numpy.typing import NDArray

import pandas as pd
import numpy as np


class DistributionGenerator:

    @staticmethod
    def save_probabililites(probabilities, save_path="./data.json"):

        with open(save_path, "w") as file:

            json.dump(probabilities, file)

    @staticmethod
    def calculate_probabilities(occurrences: Dict[str, int], total) -> Dict[str, float]:
        # create a list of tuples with value and probability
        probabilities = {}
        for value, count in occurrences.items():
            probability = count / total
            probabilities[value] = probability

        return probabilities

    @staticmethod
    def get_rand_values(
        probabilities: Dict[str, float],
        size: int,
    ) -> NDArray[np.string_]:

        rng = np.random.default_rng()

        keys: List[str] = [key for key in probabilities.keys()]
        values: List[float] = [value for value in probabilities.values()]

        param: NDArray[np.string_] = rng.choice(keys, size=size, p=values)

        return param

    @staticmethod
    def get_params(
        df: pd.DataFrame, n: int, cr_type: str = "npz", to_round: int = 3
    ) -> Dict[str, NDArray[np.string_]]:

        usizes = df["uncompressed_size"].to_numpy()
        cr_name: str = "{}_compression_ratio".format(cr_type)
        crs = df[cr_name].to_numpy()

        # get length of the arrays
        l = len(usizes)

        occur_dims = {}

        occur_crs = {}

        for i in range(l):
            cr: float = round(crs[i], to_round)
            key_cr: str = f"{cr}"

            usize: int = usizes[i]
            # get the xside, which is one side of the square root of the the dimensions
            # to also use for greyscale, if the original size is not divisible by three,
            # currently does not work with rest of the generator
            if usize % 3 != 0:
                size: int = int(usize / 3)

            # if divisible by three then has rgb channels
            else:
                size: int = int(usize / 3)

            # get the side size of the to generate synthetic
            dim: int = int(np.sqrt(size))
            key_dim: str = f"{dim}"

            if key_cr in occur_crs:
                occur_crs[key_cr] += 1
            else:
                occur_crs[key_cr] = 1

            if key_dim in occur_dims:
                occur_dims[key_dim] += 1
            else:
                occur_dims[key_dim] = 1

        dim_probs = DistributionGenerator.calculate_probabilities(occur_dims, l)
        cr_probs = DistributionGenerator.calculate_probabilities(occur_crs, l)

        dim_param = DistributionGenerator.get_rand_values(dim_probs, n)
        cr_param = DistributionGenerator.get_rand_values(cr_probs, n)

        params: Dict[str, NDArray[np.string_]] = {
            "dimensions": dim_param,
            cr_name: cr_param,
        }

        return params


if __name__ == "__main__":

    df_path = (
        "./results/20240626T192336==3=eagleimagenet--30000-processed-images-results.csv"
    )

    df = pd.read_csv(df_path)

    params = DistributionGenerator.get_params(df, 10)

    print(params)
