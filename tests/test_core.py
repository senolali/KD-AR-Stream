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

for i in range(len(X_scaled)):
	kdar.partial_fit(X_scaled[i:i+1], timestamps[i], np.array([i]))
	# Calculate ARI in each 10 points
	if i % 10 == 0 and i > 0:
		kdar.plot_data()
            
# Final ARI
y_pred = kdar.labels_
ARI = adjusted_rand_score(y_true, y_pred)
print(f"Final ARI: {ARI:.4f}")

# Final plot
kdar.plot_data()