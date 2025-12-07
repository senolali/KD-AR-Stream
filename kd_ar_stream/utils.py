# kd_ar_stream/utils.py
import numpy as np
from importlib.resources import files

def load_exclastar():
    """Load ExclaStar dataset from the package."""
    data_path = files("kd_ar_stream").joinpath("data/ExclaStar.txt")
    dataset = np.loadtxt(data_path, delimiter=',', dtype=float)
    X = dataset[:, 1:3]
    y_true = dataset[:, 3]
    return X, y_true
