import gurobipy as gp
import numpy as np
import os
import time
from typing import List
from gurobipy import GRB, quicksum, LinExpr, nlfunc
from copy import deepcopy
from sklearn.cluster import KMeans

import logging
from constants.filepath_constants import RESULTS_DIR

LOGGER = logging.getLogger(__name__)

class DynamicKmeansController():
    """
    Dynamic KMeans clustering controller for streaming data.

    Core process:
    → New data batch arrives
    → Assign new points to existing clusters (or initialize)
    → Update cluster centers
    → Split clusters exceeding max_size into two
    → Compute and print current loss
    → Repeat for next batch

    Attributes:
        data (np.ndarray): full dataset for indexing.
        batch_size (int): number of points per incoming batch.
        min_size (int): minimum cluster size before merge (future extension).
        max_size (int): maximum cluster size before split.
        centroids (list[np.ndarray]): current cluster centers.
        cluster_data_indices (dict[int, list[int]]): mapping cluster IDs to data indices.
        assignments (dict[int, int]): mapping data index to its cluster ID.
        loss_history (list[float]): recorded loss after each batch.
        time_history (list[float]): cumulative processing time after each batch.
    """

    def __init__(self, data, batch_size=100, min_size=60, max_size=150):
        self.data = data
        self.batch_size = batch_size
        self.min_size = min_size
        self.max_size = max_size
        self.centroids = []
        self.cluster_data_indices = {}
        self.assignments = {}
        self.loss_history = []
        self.time_history = []
        self._current_time = 0.0

    def run(self):
        """Run dynamic clustering over the entire dataset in streaming batches."""
        n_points = self.data.shape[0]
        for batch_start in range(0, n_points, self.batch_size):
            batch_indices = list(range(batch_start, min(batch_start + self.batch_size, n_points)))
            batch = self.data[batch_indices]
            t0 = time.time()

            self._add_batch(batch, batch_indices)
            self._update_clusters()
            self._split_large_clusters()

            batch_loss = self._compute_loss()
            self._current_time += time.time() - t0

            self.loss_history.append(batch_loss)
            self.time_history.append(self._current_time)
            print(f"Batch {batch_start//self.batch_size + 1}: Loss = {batch_loss:.4f}, Time = {self._current_time:.4f}s")

    def _add_batch(self, batch, indices):
        """Assign a new batch of points to clusters or initialize clustering."""
        if not self.centroids:
            # Initial clustering on first batch
            k_init = max(1, len(batch) // ((self.min_size + self.max_size) // 2))
            km = KMeans(n_clusters=k_init, init='k-means++', random_state=42).fit(batch)
            self.centroids = list(km.cluster_centers_)
            for idx, lbl in zip(indices, km.labels_):
                self.assignments[idx] = lbl
            for cid in range(k_init):
                self.cluster_data_indices[cid] = [idx for idx in indices if self.assignments[idx] == cid]
        else:
            # Assign each new point to nearest centroid
            for idx, point in zip(indices, batch):
                distances = [np.linalg.norm(point - c) for c in self.centroids]
                lbl = int(np.argmin(distances))
                self.assignments[idx] = lbl
                self.cluster_data_indices.setdefault(lbl, []).append(idx)

    def _update_clusters(self):
        """Recompute centroids based on current cluster memberships."""
        for cid, members in self.cluster_data_indices.items():
            pts = np.array([self.data[i] for i in members])
            if pts.size > 0:
                self.centroids[cid] = pts.mean(axis=0)

    def _split_large_clusters(self):
        """Split any cluster larger than max_size into two subclusters."""
        next_cid = max(self.cluster_data_indices.keys(), default=-1) + 1
        oversized = [cid for cid, members in self.cluster_data_indices.items() if len(members) > self.max_size]
        for cid in oversized:
            members = self.cluster_data_indices.pop(cid)
            pts = np.array([self.data[i] for i in members])
            km2 = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(pts)
            centers2 = km2.cluster_centers_
            labels2 = km2.labels_

            # Update original and new cluster
            self.centroids[cid] = centers2[0]
            self.cluster_data_indices[cid] = [members[i] for i, l in enumerate(labels2) if l == 0]

            self.centroids.append(centers2[1])
            self.cluster_data_indices[next_cid] = [members[i] for i, l in enumerate(labels2) if l == 1]

            # Update assignments
            for idx, lbl in zip(members, labels2):
                self.assignments[idx] = cid if lbl == 0 else next_cid

            next_cid += 1

    def _compute_loss(self):
        """Compute total inertia (sum of squared distances) as loss."""
        total_loss = 0.0
        for cid, members in self.cluster_data_indices.items():
            center = self.centroids[cid]
            pts = np.array([self.data[i] for i in members])
            total_loss += np.sum((pts - center) ** 2)
        return total_loss