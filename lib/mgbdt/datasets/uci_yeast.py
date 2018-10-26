import numpy as np
import os.path as osp
import re

from .utils import get_dataset_base


def load_data():
    id2label = {}
    label2id = {}
    label_path = osp.abspath( osp.join(get_dataset_base(), "uci_yeast", "yeast.label") )
    with open(label_path) as f:
        for row in f:
            cols = row.strip().split(" ")
            id2label[int(cols[0])] = cols[1]
            label2id[cols[1]] = int(cols[0])

    data_path = osp.abspath( osp.join(get_dataset_base(), "uci_yeast", "yeast.data") )
    with open(data_path) as f:
        rows = f.readlines()
    n_datas = len(rows)
    X = np.zeros((n_datas, 8), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        cols = re.split(" +", row.strip())
        X[i, :] = list(map(float, cols[1: 1 + 8]))
        y[i] = label2id[cols[-1]]
    return X, y
