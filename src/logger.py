import os
import logging

from constants.filepath_constants import RESULTS_DIR

def setup_logging(fileName = 'app.log'):
    # set up logging file, format and location, information level

    # logFormat = "%(asctime)s - %(levelname)s - %(message)s"
    logFormat = "%(message)s"
    dateFormat = "%m/%d/%Y %H:%M:%S %p"

    # fileHandler
    path = os.path.join(RESULTS_DIR, fileName)
    file_handler = logging.FileHandler(path, mode='w')

    logging.basicConfig(
        format=logFormat, 
        datefmt=dateFormat,
        level=logging.DEBUG,
        handlers= [
            file_handler,
            # stream_handler
        ]
    )

    LOGGER = logging.getLogger()
    return LOGGER