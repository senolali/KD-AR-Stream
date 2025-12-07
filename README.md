[![PyPI version](https://badge.fury.io/py/mcmstclustering.svg)](https://badge.fury.io/py/mcmstclustering)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)


## KD-AR Stream

KD-AR Stream is a Python package for **real-time data stream clustering** using
Kd-tree and adaptive radius based methods.

## Features

- Adaptive clustering in streaming data
- Cluster merging and splitting
- Supports amount-based and time-based sliding windows
- Minimal modern plotting for visualization



## Installation

```bash
pip install kd-ar-stream

```

## Parameters
If you want to use amount-based sliding window assign WindowType.AMOUNT_BASED
If you want to use time based sliding window, assign WindowType.TIME_BASED
N: int  -> Minimum number of points to form a cluster
r: float  -> Initial cluster radius
r_threshold: float  -> Radius increase/decrease threshold
r_max: float  -> Maximum cluster radius
window_type: WindowType -> {WindowType.AMOUNT_BASED,WindowType.TIME_BASED 
window_size: int  -> For amount-based: number of points in window
verbose: bool {True, False}
	

## Usage

```bash

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from kd_ar_stream import KDARStream, KDARStreamConfig, WindowType, load_exclastar

# Load data 
X, y_true = load_exclastar()

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
np.random.seed(42)

config = KDARStreamConfig(
	N=22,
	r=0.11,
	r_threshold=0.16,
	r_max=0.43,
	window_size=200,
	window_type=WindowType.AMOUNT_BASED,
	verbose=False
)

kdar = KDARStream(config)
timestamps = np.linspace(0, 10, len(X_scaled))

ARI_history = []
for i in range(len(X_scaled)):
	kdar.partial_fit(X_scaled[i:i+1], timestamps[i], np.array([i]))
	
y_pred = kdar.labels_
ARI = adjusted_rand_score(y_true, y_pred)
print(f"Final ARI: {ARI:.4f}")

```

## Advanced Usage

	
```bash

import unittest
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from kd_ar_stream import KDARStream, KDARStreamConfig, WindowType, load_exclastar

# Load data 
X, y_true = load_exclastar()

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
np.random.seed(42)

#Parameters N, r, r_threshold, r_max, and window_size are parameters of KD-AR Stream
#If you want to use amount-based sliding window assign WindowType.AMOUNT_BASED
#If you want to use time based sliding window, assign WindowType.TIME_BASED
config = KDARStreamConfig(
	N=22,
	r=0.11,
	r_threshold=0.16,
	r_max=0.43,
	window_size=200,
	window_type=WindowType.AMOUNT_BASED,
	verbose=False
)

kdar = KDARStream(config)
timestamps = np.linspace(0, 10, len(X_scaled))

ARI_history = []
for i in range(len(X_scaled)):
	kdar.partial_fit(X_scaled[i:i+1], timestamps[i], np.array([i]))
	
	# CAlculate ARI in each 10 points
	if i % 10 == 0 and i > 0:
		current_labels = kdar.labels_[:i+1]
		if len(np.unique(current_labels[current_labels != -1])) > 1:
			ARI = adjusted_rand_score(y_true[:i+1], current_labels)
			ARI_history.append(ARI)
			kdar.plot_data("Current ARI", ARI)
            
# Final ARI
y_pred = kdar.labels_
ARI = adjusted_rand_score(y_true, y_pred)
print(f"Final ARI: {ARI:.4f}")

# Final plot
kdar.plot_data("Final ARI", ARI)

```


## Citation

If you use this algorithm in research, please cite the corresponding paper.

```bash
Şenol, A., & Karacan, H. (2020). Kd-tree and adaptive radius (KD-AR Stream) based real-time data stream clustering. Journal of the Faculty of Engineering and Architecture of Gazi University, 35(1).
```

## BibTeX

```bash
@article{senol2020kd,
  title={Kd-tree and adaptive radius (KD-AR Stream) based real-time data stream clustering},
  author={Şenol, Ali and Karacan, Hacer},
  journal={Journal of the Faculty of Engineering and Architecture of Gazi University},
  volume={35},
  number={1},
  year={2020}
}
```

