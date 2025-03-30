import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)


import numpy as np

from pathlib import Path

from src.train import TrainDeflate as t

from typing import Dict, List, Callable


def train_deflate(models: List[Dict], save: bool = False):
    """
    Iterate through the models for the deflate and save

    Note that the nls model was created using r nonlinear least squares function (nls). I was not able to obtain similar parameter estimates in python so it was not included here.
    """
    n: int = len(models)

    for i in range(n):
        model: Callable = models[i]["model"]
        save_path: Path = models[i]["save_path"]
        data: np.ndarray = models[i]["data"]

        model(data, save_path, save)

    return 0


def main():

    models: List[Dict] = [
        {
            "model": t.train_inverse_model,
            "save_path": Path("./data/models/pretrained/deflate-inverse-model-v1.pk1"),
            "data": Path(
                "./data/processed/inputs/20240605T052200--imagenet-rand-300000-processed__jpg_npz.csv"
            ),
        },
    ]

    train_deflate(models, save=True)

    return 0


if __name__ == "__main__":

    main()
