import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.optimize import curve_fit

# type hints
from typing import Dict, Tuple


class TrainDeflate:

    @staticmethod
    def train_inverse_model(
        data: Path,
        save_path: Path,
        save: bool = False,
    ):
        def _inverse_model(x: np.ndarray, a: np.ndarray, h: np.ndarray, k: np.ndarray):
            return a / (x - h) + k

        df: pd.DataFrame = pd.read_csv(data)
        x: np.ndarray = np.array(df["entropy"])
        y: np.ndarray = np.array(df["npz_compression_ratio"])

        p0: Tuple = (8, 0, 0)
        # opt params and covar matrix of params
        popt, pcov = curve_fit(_inverse_model, x, y, p0=p0)

        a, h, k = popt

        res = y - _inverse_model(x, *popt)

        rss = np.sum(res**2)

        n = len(x)

        p = len(popt)

        mse = rss / n

        perr = np.sqrt(np.diag(pcov))

        model_data: Dict = {
            "params": {"a": a, "h": h, "k": k},
            "function": "deflate_inverse_model",
            "fit_metadata": {
                "mse": mse,
                "stderr": perr,
                "pcov": pcov.tolist(),
                "p0": p0,
            },
        }

        if save:
            joblib.dump(model_data, save_path)
            print(f"Model saved to: {save_path}\n")
            print(f"Model data:\n{model_data}")
        else:
            print(f"Model was not saved.\n")
            print(f"Model data:\n{model_data}")

        return 0

    @staticmethod
    def train_nls_model():
        """
        Consult the r code for the model nls model since python curvefit could not get a close estimate to the r.

        Note: Filter all compression ratios greater than 3

        def _nls_model(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
            return a / (x + b) + c * np.log(x)
        """

        return 0
