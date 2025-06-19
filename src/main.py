"""
This project is a course project for Prof. Qinwu Xu's "Mathematical Theory of Machine Learning" (Spring 2025), School of Mathematics, Nanjing University. It mainly focuses on the problem of dynamic clustering.

The core process is as follows:
→ New data stream arrives  
→ Perform clustering on the new data  
→ Update cluster centers and adjust previous clustering results  
→ If any cluster becomes too large, it is split  
→ Adjust the clustering result accordingly  
→ Wait for the next batch of data

Author: Yingxiao Wang  
Email: wangbottlecap@gmail.com
"""

import logger
from constants.constants import *
from backgrounds import DataReader
from optimizer import DynamicKmeansController, StaticKmeansController
from processer import ComparisonProcessor
from plotter import ComparisonPlotter

def main():
    """
    
    """
    LOGGER = logger.setup_logging()
    LOGGER.info(f"This project is a course project for Prof. Qinwu Xu's Mathematical Theory of Machine Learning (Spring 2025), School of Mathematics, Nanjing University. It mainly focuses on the problem of dynamic clustering.")
    
    rd = DataReader()
    rd.read_fvecs()
    print(rd.data)

    dyn = DynamicKmeansController(data=rd.data)
    dyn.run()
    
    sta = StaticKmeansController(data=rd.data, k=dyn.getCentroidsNum())
    sta.run()

    processer = ComparisonProcessor(dyn, sta)
    plotter = ComparisonPlotter(processer)
    plotter.print_center_distance()
    plotter.plotAll()
    
    return


if __name__ == "__main__":
    main()