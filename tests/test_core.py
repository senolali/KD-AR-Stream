import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from mcmststream import MCMSTStream, load_exclastar

# Load data 
X, y_true = load_exclastar()

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
np.random.seed(42)

# Initialize with history keeping enabled
clusterer = MCMSTStream(
    W=270,  
    n_micro=2, 
    N=2,   
    r=0.14, 
    random_state=42,
    keep_history=True  # Enable history tracking
)
for i, point in enumerate(X_scaled):
        label = clusterer.partial_fit(point)
        
        # Visualize periodically
        if i % 20 == 0 and i > 0:
            print(f"\nStep {i}:")
            print(f"  Current label for this point: {label}")
            print(f"  Micro-clusters: {len(clusterer.micro_clusters)}")
            print(f"  Macro-clusters: {len([m for m in clusterer.macro_clusters if m['active']])}")
            if clusterer.keep_history:
                hist_labels = np.array(clusterer.history_labels_)
                print(f"  History labels (unique): {np.unique(hist_labels)}")
            
            clusterer.visualize(title=f"Step {i}")
    
ARI=adjusted_rand_score(y_true,clusterer.history_labels_)
print("ARI=%0.4f"%ARI)