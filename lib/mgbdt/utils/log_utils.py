import logging
import os
import os.path as osp
import time


logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")
logger = logging.getLogger("mgbdt")
logger.setLevel(logging.INFO)


def strftime(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


def set_logging_dir(logging_dir):
    if not osp.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_path = osp.join(logging_dir, strftime() + ".log")
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))
    logger.addHandler(fh)


def set_logging_dirTmp(logging_dir):
    if not osp.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_path = osp.join(logging_dir, "0.log")
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))
    logger.addHandler(fh)
