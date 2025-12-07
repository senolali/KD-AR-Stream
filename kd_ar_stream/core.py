import numpy as np
from scipy.spatial import KDTree
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings
import matplotlib.pyplot as plt


class WindowType(Enum):
    """Type of window for data summarization."""
    TIME_BASED = "time"
    AMOUNT_BASED = "amount"


@dataclass
class KDARStreamConfig:
    """Configuration parameters for KD-AR Stream algorithm."""
    N: int = 10  # Minimum number of points to form a cluster
    r: float = 0.5  # Initial cluster radius
    r_threshold: float = 0.1  # Radius increase/decrease threshold
    r_max: float = 1.5  # Maximum cluster radius
    window_type: WindowType = WindowType.AMOUNT_BASED
    window_size: int = 200  # For amount-based: number of points in window
    verbose: bool = True
    enable_evolutionary_ops: bool = True
    evolutionary_check_interval: int = 50
    
    def validate(self):
        """Validate configuration parameters."""
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.r_max <= self.r:
            raise ValueError("r_max must be greater than r")

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")


class Cluster:
    """Represents a single cluster in the KD-AR Stream algorithm."""
    
    def __init__(self, cluster_id: int, center: np.ndarray, points: np.ndarray, 
                 timestamp: float, radius: float):
        self.id = cluster_id
        self.center = center.copy()
        self.points = points.copy()  # Aktif noktalar
        if self.points.ndim == 1:
            self.points = self.points.reshape(1, -1)
        self.shell_radius = radius
        self.kernel_radius = self._calculate_kernel_radius()
        self.timestamp = timestamp
        self.is_active = True
        self.data_count = len(points)
        self.last_update_time = timestamp
        self.deleted_points = []  # Silinen noktalar
    
    def _calculate_kernel_radius(self) -> float:
        """Calculate kernel radius as average of standard deviations."""
        if len(self.points) < 2:
            return self.shell_radius / 2
        return float(np.mean(np.std(self.points, axis=0)))
    
    def _calculate_shell_radius(self) -> float:
        """Calculate shell radius as maximum distance from center to points."""
        if len(self.points) == 0:
            return self.shell_radius / 2
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return float(np.max(distances)) if len(distances) > 0 else self.shell_radius / 2
    
    def get_active_points(self) -> np.ndarray:
        """Get only active points (excluding deleted ones)."""
        return self.points.copy()
    
    def update(self, new_points: Optional[np.ndarray] = None, 
               deleted_points: Optional[np.ndarray] = None):
        """Update cluster with new points and remove deleted ones."""
        # Silinen noktaları çıkar
        if deleted_points is not None and len(deleted_points) > 0:
            if deleted_points.ndim == 1:
                deleted_points = deleted_points.reshape(1, -1)
            
            # Aktif noktalardan silinenleri çıkar
            if len(self.points) > 0:
                # Basit bir yaklaşım: her silinen nokta için en yakın noktayı sil
                for del_point in deleted_points:
                    if len(self.points) > 0:
                        distances = np.linalg.norm(self.points - del_point, axis=1)
                        if len(distances) > 0:
                            min_idx = np.argmin(distances)
                            if distances[min_idx] < 1e-6:  # Çok yakınsa sil
                                self.points = np.delete(self.points, min_idx, axis=0)
            
            # Silinenleri history'e ekle
            self.deleted_points.extend(deleted_points)
        
        # Yeni noktaları ekle
        if new_points is not None and len(new_points) > 0:
            if new_points.ndim == 1:
                new_points = new_points.reshape(1, -1)
            self.points = np.vstack([self.points, new_points])
        
        # Veri sayısını güncelle
        self.data_count = len(self.points)
        
        # Aktif noktalar varsa merkezi güncelle
        if self.data_count > 0:
            self.center = np.mean(self.points, axis=0)
            
            # Yarıçapları güncelle
            self.shell_radius = self._calculate_shell_radius()
            self.kernel_radius = self._calculate_kernel_radius()
        else:
            # Eğer hiç aktif nokta kalmadıysa, küme inaktif olur
            self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary."""
        return {
            'id': self.id,
            'center': self.center.tolist(),
            'n_points': self.data_count,
            'n_active_points': len(self.points),
            'n_deleted_points': len(self.deleted_points),
            'shell_radius': self.shell_radius,
            'kernel_radius': self.kernel_radius,
            'is_active': self.is_active,
            'timestamp': self.timestamp
        }


class KDARStream:
    """
    Kd-tree and Adaptive Radius based Real-Time Data Stream Clustering Algorithm.
    N: int   # Minimum number of points to form a cluster
    r: float   # Initial cluster radius
    r_threshold: float  # Radius increase/decrease threshold
    r_max: float   # Maximum cluster radius
    window_type: WindowType {WindowType.AMOUNT_BASED,WindowType.TIME_BASED 
    window_size: int   # For amount-based: number of points in window
    verbose: bool {True, False}
    """
    def __init__(self, config: KDARStreamConfig):
        """Initialize KD-AR Stream algorithm."""
        config.validate()
        self.config = config
        self.d = 2  # Default, will be updated when data arrives
        self.d_set = False
        
        # Data structures
        self.clusters: List[Cluster] = []
        self.buffered_data = np.empty((0, 5))  
        self.next_cluster_id = 1
        self.processed_count = 0
        
        self.point_labels = {}  
        self.point_timestamps = {}  
        
        if self.config.verbose:
            print(f"KD-AR Stream initialized with N={config.N}, r={config.r}")
    
    def partial_fit(self, X: np.ndarray, timestamp: Optional[float] = None, 
                   original_indices: Optional[np.ndarray] = None):
        """
        Process new data points from the stream.
        """
        if timestamp is None:
            timestamp = time.time()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if not self.d_set:
            self.d = X.shape[1]
            self.buffered_data = np.empty((0, self.d + 3))
            self.d_set = True
        
        n_points = len(X)
        
        # Generate timestamps if not provided
        if np.isscalar(timestamp):
            timestamps = np.full(n_points, timestamp)
        else:
            timestamps = np.array(timestamp)
            if len(timestamps) != n_points:
                timestamps = np.full(n_points, timestamps[0])
        
        # Generate original indices if not provided
        if original_indices is None:
            start_idx = self.processed_count
            original_indices = np.arange(start_idx, start_idx + n_points)
        
        # Process each point
        for i in range(n_points):
            self._process_single_point(
                X[i], 
                timestamps[i], 
                int(original_indices[i])
            )
            self.processed_count += 1

            if (self.config.enable_evolutionary_ops and 
                self.processed_count % self.config.evolutionary_check_interval == 0):
                self._perform_evolutionary_operations()
    
    def _process_single_point(self, point: np.ndarray, timestamp: float, original_idx: int):
        """Process a single data point."""
        # Add to buffer with tracking info
        new_entry = np.hstack([
            point,
            timestamp,
            -1, 
            original_idx
        ])
        self.buffered_data = np.vstack([self.buffered_data, new_entry])
    
        # Apply window constraint
        self._apply_window_constraint()
    
        # Try to assign to existing cluster
        assigned = self._assign_to_existing_cluster(point, timestamp, original_idx)
    
        if assigned:
            pass
        else:
            # Check for new cluster formation
            self._check_for_new_cluster(point, timestamp)
        
        if self.processed_count % 10 == 0:
            self._perform_evolutionary_operations()
    
    def _apply_window_constraint(self):
        """Apply window constraint to buffer."""
        if len(self.buffered_data) == 0:
            return
        
        removed_points_by_cluster = {}
        
        if self.config.window_type == WindowType.TIME_BASED:
            current_time = time.time()
            time_threshold = current_time - self.config.window_size
            mask = self.buffered_data[:, self.d] >= time_threshold
            
            removed_data = self.buffered_data[~mask]
            
        else:  # AMOUNT_BASED
            if len(self.buffered_data) > self.config.window_size:
                removed_data = self.buffered_data[:-self.config.window_size]
                self.buffered_data = self.buffered_data[-self.config.window_size:]
            else:
                removed_data = np.empty((0, self.buffered_data.shape[1]))
        
        for data_point in removed_data:
            cluster_id = int(data_point[self.d + 1])
            if cluster_id != -1:
                if cluster_id not in removed_points_by_cluster:
                    removed_points_by_cluster[cluster_id] = []
                removed_points_by_cluster[cluster_id].append(data_point[:self.d])
        
        for cluster_id, points in removed_points_by_cluster.items():
            cluster = self._get_cluster_by_id(cluster_id)
            if cluster is not None:
                points_array = np.array(points)
                cluster.update(deleted_points=points_array)
    
    def _get_cluster_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None
    
    def _assign_to_existing_cluster(self, point: np.ndarray, timestamp: float, 
                              original_idx: int) -> bool:
        """Try to assign point to an existing cluster."""
        if len(self.clusters) == 0:
            return False
        
        best_cluster = None
        min_distance = float('inf')
        
        for cluster in self.clusters:
            # Check both active and passive clusters
            if cluster.data_count == 0:
                continue
            
            distance = np.linalg.norm(point - cluster.center)
            

            if distance <= cluster.shell_radius:
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster

            elif distance <= cluster.shell_radius + self.config.r_threshold:
                potential_radius = min(cluster.shell_radius + self.config.r_threshold, 
                                      self.config.r_max)
                if distance <= potential_radius and distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster
        
        if best_cluster is not None:
            best_cluster.update(new_points=point.reshape(1, -1))
            best_cluster.last_update_time = timestamp
            
            point_idx = np.where(
                (self.buffered_data[:, :self.d] == point).all(axis=1) & 
                (self.buffered_data[:, self.d + 2] == original_idx)
            )[0]
            
            if len(point_idx) > 0:
                self.buffered_data[point_idx[0], self.d + 1] = best_cluster.id
            
            self.point_labels[original_idx] = best_cluster.id
            self.point_timestamps[original_idx] = timestamp
            
            return True
        
        return False
    
    def _check_for_new_cluster(self, point: np.ndarray, timestamp: float):
        """Check if a new cluster should be formed."""
        # Get unassigned points
        unassigned_mask = self.buffered_data[:, self.d + 1] == -1
        unassigned_points = self.buffered_data[unassigned_mask, :self.d]
        
        if len(unassigned_points) < self.config.N:
            return
        
        try:
            # Build KD-tree 
            tree = KDTree(unassigned_points)
            
            n_to_check = min(20, len(unassigned_points))
            indices_to_check = np.random.choice(len(unassigned_points), n_to_check, replace=False)
            
            for idx in indices_to_check:
                test_point = unassigned_points[idx]
                neighbor_indices = tree.query_ball_point(test_point, self.config.r)
                
                if len(neighbor_indices) >= self.config.N:
                    candidate_points = unassigned_points[neighbor_indices]
                    candidate_center = np.mean(candidate_points, axis=0)
                    
                    valid_candidate = True
                    for cluster in self.clusters:
                        if cluster.is_active and cluster.data_count > 0:
                            distance = np.linalg.norm(candidate_center - cluster.center)
                            min_allowed_distance = cluster.shell_radius + self.config.r_max
                            if distance < min_allowed_distance:
                                valid_candidate = False
                                break
                    
                    if valid_candidate:
                        new_cluster = Cluster(
                            cluster_id=self.next_cluster_id,
                            center=candidate_center,
                            points=candidate_points,
                            timestamp=timestamp,
                            radius=self.config.r
                        )
                        
                        self.clusters.append(new_cluster)
                        self.next_cluster_id += 1
                        
                        # Update Buffer
                        unassigned_indices = np.where(unassigned_mask)[0]
                        for neighbor_idx in neighbor_indices:
                            original_idx = unassigned_indices[neighbor_idx]
                            self.buffered_data[original_idx, self.d + 1] = new_cluster.id
                            
                            # Update Dictionary
                            self.point_labels[original_idx] = new_cluster.id
                            self.point_timestamps[original_idx] = timestamp
                        
                        if self.config.verbose:
                            print(f"NEW CLUSTER {new_cluster.id} formed with {len(candidate_points)} points!")
                        
                        break
                        
        except Exception as e:
            if self.config.verbose:
                print(f"Error in cluster formation: {e}")
    
    def _perform_evolutionary_operations(self):
        """Perform merge, split, and activation operations."""
        if len(self.clusters) == 0:
            return
        
        # 1. Merge clusters
        self._merge_clusters()
        
        # 2. Split clusters
        self._split_clusters()
        
        # 3. Update cluster status
        self._update_cluster_status()
        
        # 4. REASSIGN POINTS AFTER EVOLUTION
        self._reassign_points_to_clusters()
    
    def _merge_clusters(self):
        """Merge overlapping clusters."""
        i = 0
        while i < len(self.clusters):
            j = i + 1
            merged = False
            while j < len(self.clusters):
                c1 = self.clusters[i]
                c2 = self.clusters[j]
                
                if c1.is_active and c2.is_active:
                    distance = np.linalg.norm(c1.center - c2.center)
                    
                    # Merge condition from paper
                    if (distance <= c1.shell_radius + c2.kernel_radius or
                        distance <= c1.kernel_radius + c2.shell_radius):
                        
                        # Keep the cluster with more points
                        if c1.data_count >= c2.data_count:
                            merged_cluster = c1
                            removed_cluster_id = c2.id
                            if c2.data_count > 0:
                                c1.update(new_points=c2.points)
                        else:
                            merged_cluster = c2
                            removed_cluster_id = c1.id
                            if c1.data_count > 0:
                                c2.update(new_points=c1.points)
                        
                        # Remove the other cluster
                        self.clusters.pop(j)
                        
                        # Update buffer
                        mask = self.buffered_data[:, self.d + 1] == removed_cluster_id
                        self.buffered_data[mask, self.d + 1] = merged_cluster.id
                        
                        # Update Dictionary
                        for idx in list(self.point_labels.keys()):
                            if self.point_labels[idx] == removed_cluster_id:
                                self.point_labels[idx] = merged_cluster.id
                        
                        if self.config.verbose:
                            print(f"MERGED: Clusters {c1.id} and {c2.id} into {merged_cluster.id}")
                        
                        merged = True
                        continue
                
                j += 1
            
            if not merged:
                i += 1
    
    def _split_clusters(self):
        """Split clusters that have become too large or heterogeneous."""
        new_clusters = []
        clusters_to_remove = []
        
        for cluster in self.clusters:
            if not cluster.is_active or cluster.data_count < 2 * self.config.N:
                continue
            
            try:
                # Build KD-tree for cluster points
                tree = KDTree(cluster.points)
                
                # Try to find a split
                for i in range(min(10, len(cluster.points))):
                    test_point = cluster.points[i]
                    
                    # Find points within radius r
                    indices = tree.query_ball_point(test_point, self.config.r)
                    
                    if len(indices) >= self.config.N:
                        candidate_points = cluster.points[indices]
                        remaining_indices = [idx for idx in range(len(cluster.points)) 
                                           if idx not in indices]
                        
                        if len(remaining_indices) >= self.config.N:
                            remaining_points = cluster.points[remaining_indices]
                            
                            # Calculate properties
                            center1 = np.mean(candidate_points, axis=0)
                            center2 = np.mean(remaining_points, axis=0)
                            
                            # Calculate radii
                            dists1 = np.linalg.norm(candidate_points - center1, axis=1)
                            dists2 = np.linalg.norm(remaining_points - center2, axis=1)
                            radius1 = float(np.max(dists1)) if len(dists1) > 0 else self.config.r
                            radius2 = float(np.max(dists2)) if len(dists2) > 0 else self.config.r
                            
                            distance = np.linalg.norm(center1 - center2)
                            
                            if distance > radius1 + radius2:
                                new_cluster1 = Cluster(
                                    cluster_id=self.next_cluster_id,
                                    center=center1,
                                    points=candidate_points,
                                    timestamp=cluster.timestamp,
                                    radius=radius1
                                )
                                
                                new_cluster2 = Cluster(
                                    cluster_id=self.next_cluster_id + 1,
                                    center=center2,
                                    points=remaining_points,
                                    timestamp=cluster.timestamp,
                                    radius=radius2
                                )
                                
                                new_clusters.extend([new_cluster1, new_cluster2])
                                self.next_cluster_id += 2
                                clusters_to_remove.append(cluster)
                                
                                for i in range(len(self.buffered_data)):
                                    if int(self.buffered_data[i, self.d + 1]) == cluster.id:
                                        point = self.buffered_data[i, :self.d]
                                        dist1 = np.linalg.norm(point - center1)
                                        dist2 = np.linalg.norm(point - center2)
                                        new_cluster_id = new_cluster1.id if dist1 < dist2 else new_cluster2.id
                                        self.buffered_data[i, self.d + 1] = new_cluster_id
                                        
                                        original_idx = int(self.buffered_data[i, self.d + 2])
                                        self.point_labels[original_idx] = new_cluster_id
                                
                                if self.config.verbose:
                                    print(f"SPLIT: Cluster {cluster.id} into {new_cluster1.id} and {new_cluster2.id}")
                                
                                break
                                
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in cluster split: {e}")
        
        for cluster in clusters_to_remove:
            if cluster in self.clusters:
                self.clusters.remove(cluster)
        
        self.clusters.extend(new_clusters)
    
    def _reassign_points_to_clusters(self):
        """Reassign buffered points to clusters after cluster center or radius changes."""
        if len(self.clusters) == 0 or len(self.buffered_data) == 0:
            return
        
        for i in range(len(self.buffered_data)):
            point = self.buffered_data[i, :self.d]
            current_cluster_id = int(self.buffered_data[i, self.d + 1])
            
            best_cluster = None
            min_distance = float('inf')
            
            for cluster in self.clusters:
                distance = np.linalg.norm(point - cluster.center)
                if distance <= cluster.shell_radius and distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster
            
            if best_cluster is not None:
                if current_cluster_id != best_cluster.id:
                    self.buffered_data[i, self.d + 1] = best_cluster.id
                    
                    original_idx = int(self.buffered_data[i, self.d + 2])
                    self.point_labels[original_idx] = best_cluster.id
    
    def _update_cluster_status(self):
        """Activate or deactivate clusters based on point count."""
        for cluster in self.clusters:
            if cluster.data_count >= self.config.N:
                if not cluster.is_active:
                    cluster.is_active = True
                    if self.config.verbose and self.processed_count % 100 == 0:
                        print(f"ACTIVATED: Cluster {cluster.id}")
            else:
                if cluster.is_active:
                    cluster.is_active = False
                    if self.config.verbose and self.processed_count % 100 == 0:
                        print(f"DEACTIVATED: Cluster {cluster.id}")
    
    @property
    def labels_(self) -> np.ndarray:
        """Get cluster labels for all processed points."""
        labels = np.full(self.processed_count, -1, dtype=int)
        
        for i in range(self.processed_count):
            if i in self.point_labels:
                labels[i] = self.point_labels[i]
        
        return labels
    
    @property
    def n_clusters_(self) -> int:
        """Get number of active clusters."""
        return sum(1 for c in self.clusters if c.is_active)
    
    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get information about all clusters."""
        return [cluster.to_dict() for cluster in self.clusters]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics."""
        active_clusters = sum(1 for c in self.clusters if c.is_active)
        inactive_clusters = sum(1 for c in self.clusters if not c.is_active)
        
        unassigned_count = 0
        if len(self.buffered_data) > 0:
            unassigned_count = np.sum(self.buffered_data[:, self.d + 1] == -1)
        
        return {
            'processed_points': self.processed_count,
            'total_clusters': len(self.clusters),
            'active_clusters': active_clusters,
            'inactive_clusters': inactive_clusters,
            'buffered_points': len(self.buffered_data),
            'unassigned_points': int(unassigned_count),
            'labeled_points': len(self.point_labels)
        }
    
    def get_all_points_with_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all processed points with their cluster labels."""
        points = []
        labels = []
        
        for i in range(self.processed_count):
            if i in self.point_labels:
                for j in range(len(self.buffered_data)):
                    if int(self.buffered_data[j, self.d + 2]) == i:
                        points.append(self.buffered_data[j, :self.d])
                        labels.append(self.point_labels[i])
                        break
        
        return np.array(points), np.array(labels, dtype=int)
    
    def plot_data(self, index, index_value):
        """Plot current clusters with modern, minimal design."""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        
        MODERN_COLORS = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        
        if not hasattr(self, '_cluster_colors'):
            self._cluster_colors = {}
        
        for cluster in self.clusters:
            if cluster.id not in self._cluster_colors:
                color_idx = (cluster.id - 1) % len(MODERN_COLORS)
                self._cluster_colors[cluster.id] = MODERN_COLORS[color_idx]
        
        unassigned_mask = self.buffered_data[:, self.d + 1] == -1
        if np.any(unassigned_mask):
            unassigned_points = self.buffered_data[unassigned_mask, :self.d]
            ax.scatter(unassigned_points[:, 0], unassigned_points[:, 1],
                      c='#cccccc', s=20, alpha=0.2, marker='.', 
                      label=f'Unassigned ({np.sum(unassigned_mask)})',
                      edgecolors='none', zorder=1)
        
        cluster_groups = {}
        for i in range(len(self.buffered_data)):
            cluster_id = int(self.buffered_data[i, self.d + 1])
            if cluster_id > 0:
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(self.buffered_data[i, :self.d])
        
        for cluster_id, points in cluster_groups.items():
            if points:
                points_array = np.array(points)
                cluster = self._get_cluster_by_id(cluster_id)
                color = self._cluster_colors.get(cluster_id, '#888888')
                
                if cluster and cluster.is_active:
                    marker = 'o'
                    size = 40
                    alpha = 0.7
                    edgecolor = 'white'
                    linewidth = 0.8
                    label = f'Cluster {cluster_id}'
                else:
                    marker = 's'
                    size = 30
                    alpha = 0.4
                    edgecolor = color
                    linewidth = 0.5
                    label = f'Cluster {cluster_id} (inactive)'
                
                ax.scatter(points_array[:, 0], points_array[:, 1],
                          c=color, s=size, alpha=alpha, marker=marker,
                          edgecolors=edgecolor, linewidth=linewidth,
                          label=label, zorder=2)
        
        for cluster in self.clusters:
            if len(cluster.center) < 2:
                continue
            
            color = self._cluster_colors.get(cluster.id, '#888888')
            
            if cluster.is_active:
                shell_circle = plt.Circle(cluster.center[:2], cluster.shell_radius,
                                         color=color, fill=True, alpha=0.08,
                                         zorder=3)
                kernel_circle = plt.Circle(cluster.center[:2], cluster.kernel_radius,
                                          color=color, fill=True, alpha=0.15,
                                          zorder=3)
            else:
                shell_circle = plt.Circle(cluster.center[:2], cluster.shell_radius,
                                         color=color, fill=False, alpha=0.3,
                                         linestyle=':', linewidth=1.5, zorder=3)
                kernel_circle = plt.Circle(cluster.center[:2], cluster.kernel_radius,
                                          color=color, fill=False, alpha=0.5,
                                          linestyle='--', linewidth=1, zorder=3)
            
            ax.add_patch(shell_circle)
            ax.add_patch(kernel_circle)
            
            center_marker = 'D' if cluster.is_active else 'X'
            center_size = 100 if cluster.is_active else 70
            center_color = 'white' if cluster.is_active else color
            
            ax.scatter(cluster.center[0], cluster.center[1],
                      marker=center_marker, s=center_size,
                      c=color, edgecolors=center_color,
                      linewidth=2, zorder=4)
            
            info = f"{cluster.id}\n({cluster.data_count})"
            ax.text(cluster.center[0], cluster.center[1],
                   info, fontsize=8, fontweight='bold' if cluster.is_active else 'normal',
                   ha='center', va='center', color='white' if cluster.is_active else 'black',
                   zorder=5)
        
        ax.set_title(f"KD-AR Stream Clustering\n{index}: {index_value:.4f}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                     loc='upper left', bbox_to_anchor=(1.02, 1),
                     borderaxespad=0., framealpha=0.9)
        
        stats_text = (f"Statistics:\n"
                     f"• Processed: {self.processed_count}\n"
                     f"• Active clusters: {self.n_clusters_}\n"
                     f"• Total clusters: {len(self.clusters)}\n"
                     f"• Buffer size: {len(self.buffered_data)}\n"
                     f"• Parameters:\n"
                     f"  N={self.config.N}, r={self.config.r:.3f}\n"
                     f"  r_th={self.config.r_threshold:.3f}\n"
                     f"  r_max={self.config.r_max:.3f}")
        
        ax.text(1.02, 0.3, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        
        ax.grid(True, linestyle='-', alpha=0.1, linewidth=0.5)
        ax.set_axisbelow(True)
        
        ax.set_facecolor('#fafafa')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(0.5)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()