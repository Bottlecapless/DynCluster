import matplotlib.pyplot as plt

import logging
from processer import ComparisonProcessor
from constants.filepath_constants import RESULTS_DIR

LOGGER = logging.getLogger(__name__)





# 3. 画图类
class ComparisonPlotter:
    """
    接收 ComparisonProcessor, 绘制：
      - loss 对比 (bar)
      - time 对比 (bar)
      - 中心距离 (单个文本展示)
      - 动态聚类 loss & time 历史曲线
    """
    def __init__(self, comp: ComparisonProcessor):
        self.comp = comp

    def plotAll(self):
        self.plot_dynamic_history()
        self.plot_loss_comparison()
        self.plot_time_comparison()

    def plot_loss_comparison(self):
        fig = plt.figure()
        labels = ['dynamic', 'static']
        values = [self.comp.results['dynamic_loss'], self.comp.results['static_loss']]
        plt.bar(labels, values)
        plt.ylabel('Loss')
        plt.title('Dynamic vs Static Loss Comparison')
        plt.show()

    def plot_time_comparison(self):
        fig = plt.figure()
        labels = ['dynamic', 'static']
        values = [self.comp.results['dynamic_time'], self.comp.results['static_time']]
        plt.bar(labels, values)
        plt.ylabel('Time (s)')
        plt.title('Dynamic vs Static Time Comparison')
        plt.show()

    def print_center_distance(self):
        dist = self.comp.results['center_distance']
        print(f"Average nearest-center distance between dynamic and static: {dist:.4f}")

    def plot_dynamic_history(self):
        # Loss history
        fig1 = plt.figure()
        plt.plot(self.comp.results['dynamic_loss_history'])
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Dynamic Clustering Loss over Batches')
        plt.show()
        # Time history
        fig2 = plt.figure()
        plt.plot(self.comp.results['dynamic_time_history'])
        plt.xlabel('Batch')
        plt.ylabel('Cumulative Time (s)')
        plt.title('Dynamic Clustering Time over Batches')
        plt.show()