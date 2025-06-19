import numpy as np
import time
from sklearn.cluster import KMeans

import logging
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

    def __init__(self, data:np.ndarray, batch_size=100, min_size=60, max_size=150, ratio = 0.01,threshold = 0.01):
        self.data = data                                  # 保存整个数据集
        self.batch_size = batch_size                      # 每次处理的批大小
        # self.threshold = threshold
        # self.ratio = ratio
        self.min_size = min_size                          # 最小簇大小（留作合并条件） 
        self.max_size = max_size                          # 最大簇大小（超过需分裂）
        self.centroids = []                               # 存储当前簇中心列表
        self.cluster_data_indices = {}                    # 存储簇ID到样本索引的映射
        self.assignments = {}                             # 存储样本索引到簇ID的映射
        self.loss_history = []                            # 记录每批的损失值（inertia）
        self.time_history = []                            # 记录累计时间
        self._current_time = 0.0                          # 内部跟踪累计运行时间

    def getCentroidsNum(self): return len(self.centroids)

    def run(self):
        """Run dynamic clustering over the entire dataset in streaming batches."""
        n_points = self.data.shape[0]                     # 数据总样本数
        # 按批次遍历数据
        for batch_start in range(0, n_points, self.batch_size):
            # 计算本批样本全局索引范围
            batch_indices = list(range(batch_start, min(batch_start + self.batch_size, n_points)))
            batch = self.data[batch_indices]              # 提取本批数据
            t0 = time.time()                              # 计时开始

            # 增量聚类、更新中心、拆分过大簇
            self._add_batch(batch, batch_indices)
            self._update_clusters()
            self._split_large_clusters()

            batch_loss = self._compute_loss()             # 计算本批后总损失
            self._current_time += time.time() - t0        # 更新累计时间

            self.loss_history.append(batch_loss)          # 记录损失
            self.time_history.append(self._current_time)  # 记录时间
            print(f"Batch {batch_start//self.batch_size + 1}: Loss = {batch_loss:.4f}, Time = {self._current_time:.4f}s")

    def _add_batch(self, batch, indices):
        """Assign a new batch of points to clusters or initialize clustering."""
        if not self.centroids:                           # 如果还没有簇中心（首次调用）
            # 计算初始簇数：确保至少1个簇
            k_init = max(1, len(batch) // ((self.min_size + self.max_size) // 2))
            # k_init = max(1, int(self.batch_size * self.ratio))
            # 使用 KMeans 对首批数据做静态聚类初始化
            km = KMeans(n_clusters=k_init, init='k-means++', random_state=42).fit(batch)
            self.centroids = list(km.cluster_centers_)    # 保存初始簇中心
            # 记录每个样本的簇标签
            for idx, lbl in zip(indices, km.labels_):
                self.assignments[idx] = lbl
            # 构建簇到样本索引的映射
            for cid in range(k_init):
                self.cluster_data_indices[cid] = [idx for idx in indices if self.assignments[idx] == cid]
        else:
            # 对后续批次，遍历每个新点
            for idx, point in zip(indices, batch):
                # 计算该点到所有簇中心的欧氏距离
                distances = [np.linalg.norm(point - c) for c in self.centroids]
                lbl = int(np.argmin(distances))            # 选择最近的簇
                self.assignments[idx] = lbl                 # 更新样本-簇映射
                # 将样本索引追加到对应簇的列表中
                self.cluster_data_indices.setdefault(lbl, []).append(idx)

    def _update_clusters(self):
        """Recompute centroids based on current cluster memberships."""
        # 遍历所有簇
        for cid, members in self.cluster_data_indices.items():
            # 提取该簇对应的所有样本点
            pts = np.array([self.data[i] for i in members])
            if pts.size > 0:
                # 重新计算簇中心为样本均值
                self.centroids[cid] = pts.mean(axis=0)

    def _split_large_clusters(self):
        """Split any cluster larger than max_size into two subclusters."""
        # 下一个可用簇ID
        next_cid = max(self.cluster_data_indices.keys(), default=-1) + 1
        # 找出需要分裂的簇ID列表
        oversized = [cid for cid, members in self.cluster_data_indices.items() if len(members) > self.max_size]
        for cid in oversized:
            members = self.cluster_data_indices.pop(cid)  # 暂时移除该簇
            pts = np.array([self.data[i] for i in members])
            # 对过大簇内样本再次做2簇KMeans拆分
            km2 = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(pts)
            centers2 = km2.cluster_centers_
            labels2 = km2.labels_

            # 更新原簇中心及成员
            self.centroids[cid] = centers2[0]
            self.cluster_data_indices[cid] = [members[i] for i, l in enumerate(labels2) if l == 0]

            # 添加新簇中心及成员
            self.centroids.append(centers2[1])
            self.cluster_data_indices[next_cid] = [members[i] for i, l in enumerate(labels2) if l == 1]

            # 更新分裂后所有样本的簇归属
            for idx, lbl in zip(members, labels2):
                self.assignments[idx] = cid if lbl == 0 else next_cid

            next_cid += 1                            # 更新下一个可用簇ID

    def _compute_loss(self):
        """Compute total inertia (sum of squared distances) as loss."""
        total_loss = 0.0
        # 遍历每个簇计算簇内平方和
        for cid, members in self.cluster_data_indices.items():
            center = self.centroids[cid]                # 当前簇中心
            pts = np.array([self.data[i] for i in members])
            total_loss += np.sum((pts - center) ** 2)    # 累加平方误差
        return total_loss
    


class StaticKmeansController:
    """
    Static KMeans clustering controller using faiss for full-dataset clustering.

    Attributes:
        data (np.ndarray): Full dataset of shape (N, D), dtype float32.
        k (int): Number of clusters.
        niter (int): Number of iterations for faiss KMeans.
        seed (int): Random seed for reproducibility.
        centroids (np.ndarray): Final cluster centers of shape (k, D).
        assignments (dict[int, int]): Mapping from data index to assigned cluster ID.
        cluster_data_indices (dict[int, list[int]]): Mapping from cluster ID to list of data indices.
        loss (float): Final inertia (sum of squared distances to centroids).
        time_taken (float): Time spent on clustering (seconds).
    """

    def __init__(self, data: np.ndarray, k: int, niter: int = 100, seed: int = 42):
        self.data = data.astype('float32')
        self.k = k
        self.niter = niter
        self.seed = seed
        self.centroids = None
        self.assignments = {}
        self.cluster_data_indices = {}
        self.loss = None
        self.time_taken = None

    def run(self):
        """
        Run static KMeans clustering using faiss.Kmeans.
        Records centroids, assignments, cluster_data_indices, loss, and time_taken.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss library not installed. Please install faiss to use StaticKmeansController.")

        # Initialize faiss KMeans
        d = self.data.shape[1]
        kmeans = faiss.Kmeans(d, self.k, niter=self.niter, verbose=False, seed=self.seed)

        # Train and time
        start_time = time.time()
        kmeans.train(self.data)
        self.time_taken = time.time() - start_time

        # Retrieve centroids
        self.centroids = kmeans.centroids.copy()

        # Assign each point to the nearest centroid
        distances, labels = kmeans.index.search(self.data, 1)
        labels = labels.flatten()

        # Build assignments and cluster-to-indices mapping
        for idx, lbl in enumerate(labels):
            self.assignments[idx] = int(lbl)
            self.cluster_data_indices.setdefault(int(lbl), []).append(idx)

        # Compute inertia (loss)
        total_loss = 0.0
        for idx, lbl in self.assignments.items():
            center = self.centroids[lbl]
            point = self.data[idx]
            total_loss += np.sum((point - center) ** 2)
        self.loss = float(total_loss)

    def report(self):
        """
        Return a summary of clustering results.
        """
        return {
            'Method': 'Static faiss.Kmeans',
            'Time (s)': self.time_taken,
            'Loss': self.loss,
            'Num Clusters': self.k,
            'Cluster Sizes': {cid: len(idxs) for cid, idxs in self.cluster_data_indices.items()}
        }
