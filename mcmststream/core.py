"""
Erdinç, B., Kaya, M., & Şenol, A. (2024). MCMSTStream: applying minimum spanning tree to KD-tree-based micro-clusters to define arbitrary-shaped clusters in streaming data. Neural Computing and Applications, 36(13), 7025-7042.
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import warnings
warnings.filterwarnings("ignore")


class MCMSTStream:
    """
    MCMSTStream: Streaming clustering algorithm based on Minimum Spanning Trees and KD-Trees.
    
    This implementation follows the algorithm from:
    Neural Computing and Applications (2024) 36:7025–7042
    
    Parameters
    ----------
    W : int, default=200
        Sliding window width. Maximum number of data points to keep in memory.
    
    n_micro : int, default=2
        Minimum number of micro-clusters required to form a macro-cluster.
    
    N : int, default=2
        Minimum number of data points required to define a micro-cluster.
    
    r : float, default=0.1
        Radius of micro-clusters. Points within this distance belong to the same micro-cluster.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    keep_history : bool, default=False
        Whether to keep history of all processed data points and their labels.
        Useful for evaluating clustering performance metrics.
    
    Attributes
    ----------
    micro_clusters_ : list
        List of micro-clusters detected by the algorithm.
    
    macro_clusters_ : list
        List of macro-clusters detected by the algorithm.
    
    labels_ : ndarray
        Cluster labels for data points in the current window.
    
    n_clusters_ : int
        Number of active macro-clusters.
    
    statistics_ : dict
        Algorithm statistics including number of data points, micro-clusters, and macro-clusters.
    
    history_labels_ : list
        List of all cluster labels assigned to processed data points (if keep_history=True).
    
    history_data_ : list
        List of all processed data points (if keep_history=True).
    
    Examples
    --------
    >>> from mcmststream import MCMSTStream
    >>> import numpy as np
    
    >>> # Generate synthetic data
    >>> X = np.random.randn(1000, 2)
    
    >>> # Initialize the algorithm with history keeping
    >>> clusterer = MCMSTStream(W=200, n_micro=2, N=2, r=0.1, keep_history=True)
    
    >>> # Process data in streaming fashion
    >>> for point in X:
    ...     labels = clusterer.partial_fit(point)
    
    >>> # Get all historical labels for evaluation
    >>> all_labels = clusterer.history_labels_
    >>> all_data = clusterer.history_data_
    
    >>> # Calculate metrics
    >>> from sklearn.metrics import adjusted_rand_score, silhouette_score
    >>> # Assuming you have true labels
    >>> # ari = adjusted_rand_score(true_labels, all_labels)
    >>> # sil_score = silhouette_score(all_data, all_labels)
    
    Notes
    -----
    This algorithm is designed for streaming data and uses a sliding window approach.
    It combines KD-Trees for efficient range queries with Minimum Spanning Trees for
    macro-cluster formation.
    
    References
    ----------
    .. [1] MCMSTStream: Minimum spanning tree to KD-tree-based micro-clusters for 
           streaming data. Neural Computing and Applications (2024) 36:7025–7042
    """
    
    def __init__(self, W=200, n_micro=2, N=2, r=0.1, random_state=None, keep_history=False):
        """Initialize MCMSTStream clustering algorithm."""
        self.W = W
        self.n_micro = n_micro
        self.N = N
        self.r = r
        self.random_state = random_state
        self.keep_history = keep_history
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize data structures
        self._initialize_data_structures()
        
    def _initialize_data_structures(self):
        """Initialize all data structures for the algorithm."""
        # Data structures
        self.buffered_data = []  # Online phase buffer
        self.micro_clusters = []  # List of micro-clusters
        self.macro_clusters = []  # List of macro-clusters
        
        # History tracking
        self.history_labels_ = [] if self.keep_history else None
        self.history_data_ = [] if self.keep_history else None
        
        # Counters
        self.data_counter = 0
        self.micro_cluster_counter = 0
        self.macro_cluster_counter = 0
        
        # Performance optimization
        self._mc_tree = None  # KDTree for micro-clusters
        self._mc_tree_needs_update = True
        
        # Public attributes
        self.micro_clusters_ = []
        self.macro_clusters_ = []
        self.labels_ = np.array([])
        self.n_clusters_ = 0
        self.statistics_ = {}
        
    def fit(self, X, y=None):
        """
        Fit the clustering algorithm on a batch of data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        
        y : Ignored
            Not used, present for API consistency.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Reset the algorithm
        self._initialize_data_structures()
        
        # Process data points
        for i in range(X.shape[0]):
            self.partial_fit(X[i])
        
        return self
    
    def partial_fit(self, X):
        """
        Update the clustering with new data (streaming interface).
        
        Parameters
        ----------
        X : array-like of shape (n_features,) or (n_samples, n_features)
            New data point(s) to process.
        
        Returns
        -------
        labels : ndarray
            Cluster labels for data in current window.
        """
        X = np.asarray(X)
        
        # Handle single point
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Process each point
        labels_list = []
        for i in range(X.shape[0]):
            point = X[i]
            # Add point to buffer
            point_id = self._add_data_point(point)
            
            # Execute all clustering steps
            self._run_clustering_steps()
            
            # Get the label for this point
            label = self._get_label_for_point(point_id)
            labels_list.append(label)
            
            # Store in history if enabled
            if self.keep_history:
                self.history_data_.append(point.copy())
                self.history_labels_.append(label)
        
        # Update public attributes
        self._update_public_attributes()
        
        return np.array(labels_list)
    
    def _run_clustering_steps(self):
        """Execute all clustering algorithm steps."""
        self._define_micro_clusters()        # DefineMC
        self._assign_to_micro_clusters()     # AddtoMC
        self._define_macro_clusters()        # DefineMacroC
        self._assign_micro_to_macro()        # AddMCtoMacroC
        self._update_micro_clusters()        # UpdateMC
        self._update_macro_clusters()        # UpdateMacroC
        self._delete_micro_clusters()        # KillMCs
        self._delete_macro_clusters()        # KillMacroCs
        
        # Update KDTree
        self._mc_tree_needs_update = True
    
    def _get_label_for_point(self, point_id):
        """Get the cluster label for a specific point ID."""
        for data in self.buffered_data:
            if data['id'] == point_id:
                return data['macro_id']
        return 0
    
    def fit_predict(self, X, y=None):
        """
        Fit the algorithm and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        
        y : Ignored
            Not used, present for API consistency.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted cluster labels.
        """
        X = np.asarray(X)
        
        # Handle single point
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        labels = np.zeros(X.shape[0])
        
        # If no micro-clusters exist, all points are noise
        if not self.micro_clusters:
            return labels
        
        # Update micro-cluster KDTree if needed
        self._update_mc_tree()
        
        # Process each point
        for i in range(X.shape[0]):
            point = X[i]
            
            # Find distance to nearest micro-cluster center
            distance, nearest_idx = self._mc_tree.query(point.reshape(1, -1), k=1)
            distance = distance[0]
            nearest_idx = nearest_idx[0]
            
            # Check if within radius r
            if distance <= self.r and nearest_idx < len(self.micro_clusters):
                # Assign to the macro-cluster of the nearest micro-cluster
                mc = self.micro_clusters[nearest_idx]
                labels[i] = mc['macro_id']
            else:
                labels[i] = 0  # Noise/outlier
        
        return labels
    
    def _add_data_point(self, x):
        """Add new data point to sliding window."""
        point_id = self.data_counter
        
        self.buffered_data.append({
            'id': point_id,
            'features': np.array(x),
            'mc_id': 0,      # Not assigned to any micro-cluster
            'macro_id': 0,   # Not assigned to any macro-cluster
            'timestamp': self.data_counter
        })
        self.data_counter += 1
        
        # Maintain sliding window of size W
        if len(self.buffered_data) > self.W:
            removed = self.buffered_data.pop(0)
            if removed['mc_id'] > 0:
                self._update_mc_after_removal(removed['mc_id'])
        
        return point_id
    
    def get_all_labels(self):
        """
        Get all cluster labels for all processed data points.
        
        Returns
        -------
        labels : ndarray or None
            Array of all cluster labels if history is kept, None otherwise.
        """
        if self.keep_history:
            return np.array(self.history_labels_)
        else:
            print("History not kept. Initialize with keep_history=True to enable this feature.")
            return None
    
    def get_all_data(self):
        """
        Get all processed data points.
        
        Returns
        -------
        data : ndarray or None
            Array of all processed data points if history is kept, None otherwise.
        """
        if self.keep_history:
            return np.array(self.history_data_)
        else:
            print("History not kept. Initialize with keep_history=True to enable this feature.")
            return None
    
    def get_label_history(self):
        """
        Get the history of cluster labels with timestamps.
        
        Returns
        -------
        history : list of tuples or None
            List of (data_point, label, timestamp) if history is kept, None otherwise.
        """
        if self.keep_history:
            return list(zip(self.history_data_, self.history_labels_, range(len(self.history_labels_))))
        else:
            print("History not kept. Initialize with keep_history=True to enable this feature.")
            return None
    
    def _update_public_attributes(self):
        """Update public attributes for scikit-learn compatibility."""
        # Update micro_clusters_
        self.micro_clusters_ = [
            {
                'id': mc['id'],
                'center': mc['center'],
                'n_points': mc['n_points'],
                'macro_cluster_id': mc['macro_id']
            }
            for mc in self.micro_clusters
        ]
        
        # Update macro_clusters_
        self.macro_clusters_ = [
            {
                'id': macro['id'],
                'n_micro_clusters': macro['n_mcs'],
                'micro_cluster_ids': macro['mc_ids'],
                'active': macro['active']
            }
            for macro in self.macro_clusters if macro['active']
        ]
        
        # Update labels_ (only for current window)
        self.labels_ = np.array([data['macro_id'] for data in self.buffered_data])
        
        # Update n_clusters_
        self.n_clusters_ = len([macro for macro in self.macro_clusters if macro['active']])
        
        # Update statistics_
        self.statistics_ = self._get_statistics()
        
        # # Debug: Print current state
        # self._debug_print()
    
    def _debug_print(self):
        """Print debug information about current state."""
        if self.buffered_data:
            current_labels = [d['macro_id'] for d in self.buffered_data]
            unique_labels = np.unique(current_labels)
            print(f"Debug: {len(self.buffered_data)} points, labels: {unique_labels}")
            print(f"Debug: {len(self.micro_clusters)} micro-clusters")
            print(f"Debug: {len([m for m in self.macro_clusters if m['active']])} active macro-clusters")
    
    def _define_micro_clusters(self):
        """Define new micro-clusters (Algorithm 2: DefineMC)."""
        unassigned_data = [d for d in self.buffered_data if d['mc_id'] == 0]
        
        if len(unassigned_data) < self.N:
            return
        
        data_points = np.array([d['features'] for d in unassigned_data])
        kd_tree = KDTree(data_points)
        
        processed_indices = set()
        for i, data_point in enumerate(data_points):
            if i in processed_indices:
                continue
                
            indices = kd_tree.query_ball_point(data_point, self.r)
            
            if len(indices) >= self.N:
                mc_points = data_points[indices]
                center = np.mean(mc_points, axis=0)
                
                if self._is_far_from_existing_mcs(center):
                    self.micro_cluster_counter += 1
                    new_mc = {
                        'id': self.micro_cluster_counter,
                        'center': center,
                        'points': mc_points.tolist(),
                        'point_ids': [unassigned_data[idx]['id'] for idx in indices],
                        'n_points': len(indices),
                        'macro_id': 0  # Will be assigned later
                    }
                    self.micro_clusters.append(new_mc)
                    
                    # Assign points to this micro-cluster
                    for idx in indices:
                        data_id = unassigned_data[idx]['id']
                        self._get_data_by_id(data_id)['mc_id'] = self.micro_cluster_counter
                        # IMPORTANT: Also update macro_id if micro-cluster already has one
                        if new_mc['macro_id'] > 0:
                            self._get_data_by_id(data_id)['macro_id'] = new_mc['macro_id']
                    
                    processed_indices.update(indices)
                    self._mc_tree_needs_update = True
    
    def _assign_to_micro_clusters(self):
        """Assign new arrival data to existing micro-clusters (Algorithm 3: AddtoMC)."""
        if not self.micro_clusters:
            return
            
        self._update_mc_tree()
        unassigned_data = [d for d in self.buffered_data if d['mc_id'] == 0]
        
        if not unassigned_data:
            return
            
        data_points = np.array([d['features'] for d in unassigned_data])
        mc_centers = np.array([mc['center'] for mc in self.micro_clusters])
        mc_tree = KDTree(mc_centers)
        
        distances, indices = mc_tree.query(data_points, k=1)
        
        for i, (dist, mc_idx) in enumerate(zip(distances, indices)):
            if dist <= self.r:
                data_id = unassigned_data[i]['id']
                mc_id = self.micro_clusters[mc_idx]['id']
                
                self._get_data_by_id(data_id)['mc_id'] = mc_id
                mc = self.micro_clusters[mc_idx]
                mc['points'].append(unassigned_data[i]['features'])
                mc['point_ids'].append(data_id)
                mc['n_points'] += 1
                
                # IMPORTANT: Also assign macro-cluster label if micro-cluster has one
                if mc['macro_id'] > 0:
                    self._get_data_by_id(data_id)['macro_id'] = mc['macro_id']
    
    def _define_macro_clusters(self):
        """Define macro-clusters using MST (Algorithm 4: DefineMacroC)."""
        # Get micro-clusters not assigned to any macro-cluster
        unassigned_mcs = [mc for mc in self.micro_clusters if mc['macro_id'] == 0]
        
        if len(unassigned_mcs) < self.n_micro:
            return
            
        mc_centers = np.array([mc['center'] for mc in unassigned_mcs])
        
        if len(mc_centers) == 0:
            return
            
        # Calculate distances and apply threshold
        distances = squareform(pdist(mc_centers))
        distances[distances > 2 * self.r] = np.inf
        
        # Find connected components
        visited = set()
        macro_clusters_found = []
        
        for i in range(len(unassigned_mcs)):
            if i in visited:
                continue
                
            current_cluster = []
            stack = [i]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                    
                visited.add(current)
                current_cluster.append(current)
                
                # Find neighbors within 2r
                neighbors = np.where(distances[current] <= 2 * self.r)[0]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if len(current_cluster) >= self.n_micro:
                macro_clusters_found.append(current_cluster)
        
        # Create macro-clusters
        for cluster_indices in macro_clusters_found:
            self.macro_cluster_counter += 1
            macro_id = self.macro_cluster_counter
            new_macro = {
                'id': macro_id,
                'mc_ids': [unassigned_mcs[idx]['id'] for idx in cluster_indices],
                'n_mcs': len(cluster_indices),
                'active': True
            }
            self.macro_clusters.append(new_macro)
            
            # Assign micro-clusters to this macro-cluster
            for idx in cluster_indices:
                mc_id = unassigned_mcs[idx]['id']
                mc = self._get_mc_by_id(mc_id)
                if mc:
                    mc['macro_id'] = macro_id
                    
                    # Also update all points in this micro-cluster
                    for data_id in mc['point_ids']:
                        data = self._get_data_by_id(data_id)
                        if data:
                            data['macro_id'] = macro_id
    
    def _assign_micro_to_macro(self):
        """Assign micro-clusters to existing macro-clusters (Algorithm 6: AddMCtoMacroC)."""
        unassigned_mcs = [mc for mc in self.micro_clusters 
                         if mc['macro_id'] == 0 and mc['n_points'] >= self.N]
        
        if not unassigned_mcs or not self.macro_clusters:
            return
            
        assigned_mcs = [mc for mc in self.micro_clusters if mc['macro_id'] > 0]
        
        if not assigned_mcs:
            return
            
        assigned_centers = np.array([mc['center'] for mc in assigned_mcs])
        assigned_macro_ids = [mc['macro_id'] for mc in assigned_mcs]
        
        tree = KDTree(assigned_centers)
        
        for mc in unassigned_mcs:
            dist, idx = tree.query(mc['center'])
            
            if dist <= 2 * self.r:
                macro_id = assigned_macro_ids[idx]
                mc['macro_id'] = macro_id
                
                # Update all points in this micro-cluster
                for data_id in mc['point_ids']:
                    data = self._get_data_by_id(data_id)
                    if data:
                        data['macro_id'] = macro_id
                
                # Update macro-cluster
                macro = self._get_macro_by_id(macro_id)
                if macro:
                    macro['mc_ids'].append(mc['id'])
                    macro['n_mcs'] += 1
    
    def _update_micro_clusters(self):
        """Update micro-cluster information (Algorithm 7: UpdateMC)."""
        for mc in self.micro_clusters:
            mc_points = []
            point_ids = []
            
            for data in self.buffered_data:
                if data['mc_id'] == mc['id']:
                    mc_points.append(data['features'])
                    point_ids.append(data['id'])
            
            if mc_points:
                mc['center'] = np.mean(mc_points, axis=0)
                mc['points'] = mc_points
                mc['point_ids'] = point_ids
                mc['n_points'] = len(mc_points)
    
    def _update_macro_clusters(self):
        """Update macro-cluster information (Algorithm 8: UpdateMacroC)."""
        for macro in self.macro_clusters:
            if not macro['active']:
                continue
                
            mc_list = [self._get_mc_by_id(mc_id) for mc_id in macro['mc_ids']]
            mc_list = [mc for mc in mc_list if mc is not None]
            
            if len(mc_list) < self.n_micro:
                macro['active'] = False
                continue
                
            mc_centers = np.array([mc['center'] for mc in mc_list])
            
            if len(mc_centers) > 1:
                distances = squareform(pdist(mc_centers))
                distances[distances > 2 * self.r] = np.inf
                
                visited = set()
                new_mc_groups = []
                
                for i in range(len(mc_list)):
                    if i in visited:
                        continue
                        
                    component = []
                    stack = [i]
                    
                    while stack:
                        current = stack.pop()
                        if current in visited:
                            continue
                            
                        visited.add(current)
                        component.append(current)
                        
                        neighbors = np.where(distances[current] < np.inf)[0]
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                    
                    new_mc_groups.append(component)
                
                valid_components = []
                for component in new_mc_groups:
                    if len(component) >= self.n_micro:
                        valid_components.append(component)
                
                if len(valid_components) != 1:
                    macro['active'] = False
                    
                    for component in valid_components:
                        if len(component) >= self.n_micro:
                            self.macro_cluster_counter += 1
                            new_macro_id = self.macro_cluster_counter
                            new_macro = {
                                'id': new_macro_id,
                                'mc_ids': [mc_list[idx]['id'] for idx in component],
                                'n_mcs': len(component),
                                'active': True
                            }
                            self.macro_clusters.append(new_macro)
                            
                            for idx in component:
                                mc = mc_list[idx]
                                mc['macro_id'] = new_macro_id
                                
                                # Update all points in this micro-cluster
                                for data_id in mc['point_ids']:
                                    data = self._get_data_by_id(data_id)
                                    if data:
                                        data['macro_id'] = new_macro_id
    
    def _delete_micro_clusters(self):
        """Delete micro-clusters with insufficient points (Algorithm 9: KillMCs)."""
        to_delete = []
        
        for i, mc in enumerate(self.micro_clusters):
            if mc['n_points'] < self.N:
                to_delete.append(i)
                
                for data_id in mc['point_ids']:
                    data = self._get_data_by_id(data_id)
                    if data:
                        data['mc_id'] = 0
                        data['macro_id'] = 0
        
        for i in sorted(to_delete, reverse=True):
            deleted_mc = self.micro_clusters.pop(i)
            
            if deleted_mc['macro_id'] > 0:
                macro = self._get_macro_by_id(deleted_mc['macro_id'])
                if macro and deleted_mc['id'] in macro['mc_ids']:
                    macro['mc_ids'].remove(deleted_mc['id'])
                    macro['n_mcs'] -= 1
        
        if to_delete:
            self._mc_tree_needs_update = True
    
    def _delete_macro_clusters(self):
        """Delete macro-clusters with insufficient micro-clusters (Algorithm 10: KillMacroCs)."""
        to_delete = []
        
        for i, macro in enumerate(self.macro_clusters):
            if macro['n_mcs'] < self.n_micro:
                to_delete.append(i)
                
                for mc_id in macro['mc_ids']:
                    mc = self._get_mc_by_id(mc_id)
                    if mc:
                        mc['macro_id'] = 0
                        # Also update points in these micro-clusters
                        for data_id in mc['point_ids']:
                            data = self._get_data_by_id(data_id)
                            if data:
                                data['macro_id'] = 0
        
        for i in sorted(to_delete, reverse=True):
            self.macro_clusters.pop(i)
    
    def _update_mc_after_removal(self, mc_id):
        """Helper to update micro-cluster after point removal."""
        mc = self._get_mc_by_id(mc_id)
        if mc:
            mc['n_points'] = max(0, mc['n_points'] - 1)
            self._mc_tree_needs_update = True
    
    def _is_far_from_existing_mcs(self, center):
        """Check if a point is far enough from existing micro-clusters."""
        if not self.micro_clusters:
            return True
            
        self._update_mc_tree()
        distances, _ = self._mc_tree.query(center.reshape(1, -1), k=1)
        return distances[0] > 0.75 * self.r
    
    def _update_mc_tree(self):
        """Update KDTree for micro-clusters."""
        if self._mc_tree_needs_update and self.micro_clusters:
            centers = np.array([mc['center'] for mc in self.micro_clusters])
            self._mc_tree = KDTree(centers)
            self._mc_tree_needs_update = False
    
    def _get_data_by_id(self, data_id):
        """Get data point by ID."""
        for data in self.buffered_data:
            if data['id'] == data_id:
                return data
        return None
    
    def _get_mc_by_id(self, mc_id):
        """Get micro-cluster by ID."""
        for mc in self.micro_clusters:
            if mc['id'] == mc_id:
                return mc
        return None
    
    def _get_macro_by_id(self, macro_id):
        """Get macro-cluster by ID."""
        for macro in self.macro_clusters:
            if macro['id'] == macro_id:
                return macro
        return None
    
    def _get_statistics(self):
        """Get algorithm statistics."""
        active_macro_clusters = [m for m in self.macro_clusters if m['active']]
        
        # Calculate label distribution in current buffer
        current_labels = [d['macro_id'] for d in self.buffered_data]
        unique_labels = np.unique(current_labels)
        
        return {
            'n_data_points': len(self.buffered_data),
            'n_micro_clusters': len(self.micro_clusters),
            'n_macro_clusters': len(active_macro_clusters),
            'unique_labels': unique_labels,
            'window_size': self.W,
            'parameters': {
                'W': self.W,
                'n_micro': self.n_micro,
                'N': self.N,
                'r': self.r
            }
        }
    
    def evaluate(self, true_labels=None):
        """
        Evaluate clustering performance using various metrics.
        
        Parameters
        ----------
        true_labels : array-like, optional
            True labels for evaluation. If None, only internal metrics are calculated.
        
        Returns
        -------
        metrics : dict
            Dictionary containing various clustering metrics.
        """
        metrics = {}
        
        if not self.keep_history:
            print("Warning: History not kept. Metrics will be calculated only on current window.")
            current_data = np.array([d['features'] for d in self.buffered_data])
            current_labels = np.array([d['macro_id'] for d in self.buffered_data])
        else:
            current_data = np.array(self.history_data_)
            current_labels = np.array(self.history_labels_)
        
        # Only calculate metrics if we have data
        if len(current_data) == 0:
            return metrics
        
        # Debug information
        unique_labels = np.unique(current_labels)
        print(f"=== DEBUG INFO ===")
        print(f"Data points: {len(current_data)}")
        print(f"Unique labels: {unique_labels}")
        print(f"Label distribution: {np.bincount(current_labels.astype(int))}")
        print(f"Active macro-clusters: {self.n_clusters_}")
        print(f"Micro-clusters: {len(self.micro_clusters)}")
        
        # Check if micro-clusters have macro-cluster assignments
        for i, mc in enumerate(self.micro_clusters):
            if mc['macro_id'] == 0:
                print(f"  Micro-cluster {mc['id']}: {mc['n_points']} points, macro_id=0")
            else:
                print(f"  Micro-cluster {mc['id']}: {mc['n_points']} points, macro_id={mc['macro_id']}")
        
        # Internal metrics (don't require true labels)
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        try:
            if len(unique_labels) > 1:
                metrics['silhouette_score'] = silhouette_score(current_data, current_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(current_data, current_labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(current_data, current_labels)
                print(f"Internal metrics calculated successfully.")
            else:
                print(f"Warning: Only one cluster found. Internal metrics not calculated.")
        except Exception as e:
            print(f"Warning: Could not calculate internal metrics: {e}")
        
        # External metrics (require true labels)
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
            
            true_labels = np.asarray(true_labels)
            
            # Ensure lengths match
            min_len = min(len(current_labels), len(true_labels))
            if min_len > 0:
                metrics['adjusted_rand_score'] = adjusted_rand_score(
                    true_labels[:min_len], current_labels[:min_len]
                )
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(
                    true_labels[:min_len], current_labels[:min_len]
                )
                
                homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
                    true_labels[:min_len], current_labels[:min_len]
                )
                metrics['homogeneity'] = homogeneity
                metrics['completeness'] = completeness
                metrics['v_measure'] = v_measure
        
        # Algorithm statistics
        metrics.update({
            'n_clusters': self.n_clusters_,
            'n_micro_clusters': len(self.micro_clusters),
            'n_data_points': len(current_data),
            'noise_ratio': np.sum(current_labels == 0) / len(current_labels) if len(current_labels) > 0 else 0
        })
        
        return metrics
    
    def visualize(self, figsize=(10, 8), title="MCMSTStream Clustering"):
        """
        Visualize current clustering state.
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size.
        
        title : str, default="MCMSTStream Clustering"
            Plot title.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        
        ax : matplotlib.axes.Axes
            The created axes.
        """
        import matplotlib.patches as patches
        from matplotlib import cm
        
        if len(self.buffered_data) == 0:
            print("No data available for visualization!")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        
        # Color palette
        color_palette = cm.tab10(np.linspace(0, 1, 10))
        
        # Plot data points
        data_points = []
        colors = []
        labels = []
        
        for data in self.buffered_data:
            if len(data['features']) >= 2:
                data_points.append(data['features'][:2])
                label = data['macro_id']
                labels.append(label)
                
                if label == 0:
                    colors.append('lightgray')
                else:
                    cluster_color = color_palette[label % len(color_palette)]
                    colors.append(cluster_color)
        
        data_points = np.array(data_points)
        
        if len(data_points) > 0:
            scatter = ax.scatter(data_points[:, 0], data_points[:, 1],
                      c=colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Debug info
            unique_labels = np.unique(labels)
            # print(f"Visualization: {len(data_points)} points, labels: {unique_labels}")
            
            # Set axis limits
            x_min, x_max = data_points[:, 0].min(), data_points[:, 0].max()
            y_min, y_max = data_points[:, 1].min(), data_points[:, 1].max()
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        
        # Draw MST connections
        for macro in self.macro_clusters:
            if macro['active'] and macro['n_mcs'] > 1:
                cluster_color = color_palette[macro['id'] % len(color_palette)]
                
                mc_ids = macro['mc_ids']
                micro_clusters = [self._get_mc_by_id(mc_id) for mc_id in mc_ids]
                micro_clusters = [mc for mc in micro_clusters if mc is not None]
                
                if len(micro_clusters) >= 2:
                    centers = np.array([mc['center'][:2] for mc in micro_clusters])
                    
                    distances = squareform(pdist(centers))
                    distances[distances > 2 * self.r] = np.inf
                    
                    mst = minimum_spanning_tree(distances)
                    mst_dense = mst.toarray()
                    
                    for i in range(len(mst_dense)):
                        for j in range(i + 1, len(mst_dense)):
                            if mst_dense[i, j] > 0:
                                ax.plot([centers[i, 0], centers[j, 0]],
                                       [centers[i, 1], centers[j, 1]],
                                       color=cluster_color, linewidth=2.5, 
                                       alpha=0.6, zorder=1)
        
        # Draw micro-cluster circles
        for mc in self.micro_clusters:
            if len(mc['center']) >= 2:
                center = mc['center'][:2]
                
                if mc['macro_id'] == 0:
                    circle_color = 'lightgray'
                    alpha = 0.1
                else:
                    cluster_color = color_palette[mc['macro_id'] % len(color_palette)]
                    circle_color = cluster_color
                    alpha = 0.15
                
                circle = patches.Circle(center, self.r, 
                                       fill=True, alpha=alpha, 
                                       color=circle_color, zorder=2)
                ax.add_patch(circle)
                
                # Micro-cluster center point
                if mc['macro_id'] == 0:
                    center_color = 'darkgray'
                    marker = 'o'
                else:
                    center_color = circle_color
                    marker = 'X'
                
                ax.scatter(center[0], center[1], c=center_color, 
                          s=80, marker=marker, edgecolor='black', 
                          linewidth=1, alpha=0.8, zorder=4)
        
        # Draw macro-cluster regions
        for macro in self.macro_clusters:
            if macro['active'] and macro['n_mcs'] >= self.n_micro:
                cluster_color = color_palette[macro['id'] % len(color_palette)]
                
                mc_centers = []
                for mc_id in macro['mc_ids']:
                    mc = self._get_mc_by_id(mc_id)
                    if mc and len(mc['center']) >= 2:
                        mc_centers.append(mc['center'][:2])
                
                if len(mc_centers) >= 2:
                    mc_centers = np.array(mc_centers)
                    
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(mc_centers)
                        poly = patches.Polygon(mc_centers[hull.vertices], 
                                               closed=True, 
                                               fill=False, alpha=0.7,
                                               color=cluster_color, linewidth=2,
                                               linestyle='-', zorder=3)
                        ax.add_patch(poly)
                        
                        # Show macro-cluster ID
                        centroid = mc_centers.mean(axis=0)
                        ax.text(centroid[0], centroid[1], f'C{macro["id"]}', 
                               fontsize=12, fontweight='bold',
                               color=cluster_color, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7, 
                                        edgecolor=cluster_color),
                               zorder=5)
                    except:
                        # Fallback to bounding box
                        x_min, x_max = mc_centers[:, 0].min(), mc_centers[:, 0].max()
                        y_min, y_max = mc_centers[:, 1].min(), mc_centers[:, 1].max()
                        rect = patches.Rectangle((x_min, y_min), 
                                                 x_max - x_min, y_max - y_min,
                                                 fill=False, alpha=0.7,
                                                 color=cluster_color, linewidth=2,
                                                 linestyle='-', zorder=3)
                        ax.add_patch(rect)
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightgray', markersize=8, 
                      label='Unassigned Data'),
            plt.Line2D([0], [0], marker='X', color='w', 
                      markerfacecolor='darkgray', markersize=10, 
                      label='Micro-cluster Center'),
            patches.Circle((0, 0), 0.5, fill=True, alpha=0.2, 
                          facecolor='gray', label='Micro-cluster Radius'),
            plt.Line2D([0], [0], color='blue', linewidth=2, 
                      alpha=0.6, label='MST Connection'),
            patches.Polygon([[0, 0], [1, 0], [1, 1]], fill=False, 
                           alpha=0.7, color='green', linewidth=2, 
                           label='Macro-cluster Region')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Axis labels and title
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        
        # Show statistics
        stats = self._get_statistics()
        stats_text = (
            f'Data Points: {stats["n_data_points"]}\n'
            f'Micro-clusters: {stats["n_micro_clusters"]}\n'
            f'Macro-clusters: {stats["n_macro_clusters"]}\n'
            f'W: {self.W}\n'
            f'r: {self.r:.2f}\n'
            f'N: {self.N}\n'
            f'n_micro: {self.n_micro:}\n'
            f'Unique Labels: {stats["unique_labels"]}'
        )
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.8, edgecolor='gray'))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'W': self.W,
            'n_micro': self.n_micro,
            'N': self.N,
            'r': self.r,
            'random_state': self.random_state,
            'keep_history': self.keep_history
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
        return self
    
    
    
    
    
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from kd_ar_stream import load_exclastar

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