import logging
from optimizer import DynamicKmeansController, StaticKmeansController
from util import *
LOGGER = logging.getLogger(__name__)

# 2. 数据处理类
class ComparisonProcessor:
    """
    对比动态 vs 静态 聚类结果，存储以下信息：
      - center_distance: 两组簇中心平均最近距离
      - dynamic_loss, static_loss
      - dynamic_time, static_time
      - dynamic_history: loss & time 历史
    """
    def __init__(self, dynController:DynamicKmeansController, staController:StaticKmeansController):
        self.dyn = ComparisonProcessor.summarize_dynamic(dynController)
        self.sta = ComparisonProcessor.summarize_static(staController)
        self.results = {}
        self._compute()

    def _compute_center_distance(self):
        # 对于每个动态中心，找到距离最近的静态中心，计算平均距离
        dyn_c = self.dyn['centroids']
        sta_c = self.sta['centroids']
        dists = []
        for c in dyn_c:
            diff = sta_c - c
            dist_to_sta = np.linalg.norm(diff, axis=1)
            dists.append(np.min(dist_to_sta))
        return float(np.mean(dists))

    def _compute(self):
        self.results['center_distance'] = self._compute_center_distance()
        self.results['dynamic_loss'] = float(self.dyn['loss'])
        self.results['static_loss'] = float(self.sta['loss'])
        self.results['dynamic_time'] = float(self.dyn['time'])
        self.results['static_time'] = float(self.sta['time'])
        self.results['dynamic_loss_history'] = self.dyn['loss_history']
        self.results['dynamic_time_history'] = self.dyn['time_history']


    @staticmethod
    def summarize_dynamic(controller:DynamicKmeansController):
        """
        输出动态聚类摘要信息
        Returns a dict with:
        - num_clusters: 最终簇数
        - loss: 最终 loss
        - time: 累计运行时间
        - loss_history: 每批 loss 的列表
        - time_history: 每批累计时间的列表
        - centroids: 簇中心数组, shape=(k, D)
        """
        centroids = np.stack(controller.centroids, axis=0)
        return {
            'method': 'dynamic',
            'num_clusters': centroids.shape[0],
            'loss': controller.loss_history[-1],
            'time': controller.time_history[-1],
            'loss_history': np.array(controller.loss_history),
            'time_history': np.array(controller.time_history),
            'centroids': centroids
        }

    @staticmethod
    def summarize_static(controller:StaticKmeansController):
        """
        输出静态聚类摘要信息
        Returns a dict with:
        - num_clusters: 簇数 k
        - loss: 最终 loss
        - time: 聚类耗时
        - centroids: 簇中心数组, shape=(k, D)
        """
        return {
            'method': 'static',
            'num_clusters': controller.k,
            'loss': controller.loss,
            'time': controller.time_taken,
            'centroids': controller.centroids
        }



