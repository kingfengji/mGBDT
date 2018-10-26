import numpy as np
import os
import os.path as osp


def set_seed(seed):
    from .log_utils import logger
    if seed is None:
        seed = np.random.randint(10000)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info("[set_seed] seed={}".format(seed))
    return seed


def save_hiddens(net, x_train, y_train, x_test, y_test, out_dir):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    for phase in ["train", "test"]:
        if phase == "train":
            hiddens = net.get_hiddens(x_train)
            y = y_train
        else:
            hiddens = net.get_hiddens(x_test)
            y = y_test
        des_path = osp.join(out_dir, "y.{}.npy".format(phase))
        print("Saving data (shape={}) in {} ...".format(y.shape, des_path))
        with open(des_path, "wb") as f:
            np.save(f, y)
        for i in range(len(hiddens)):
            des_path = osp.join(out_dir, "X_{}.{}.npy".format(i, phase))
            print("Saving data (shape={}) in {} ...".format(hiddens[i].shape, des_path))
            with open(des_path, "wb") as f:
                np.save(f, hiddens[i])


def load_dataset(src_dir, layer):
    y_train = np.load(osp.join(src_dir, "y.train.npy"))
    y_test = np.load(osp.join(src_dir, "y.test.npy"))
    x_train = np.load(osp.join(src_dir, "X_{}.train.npy".format(layer)))
    x_test = np.load(osp.join(src_dir, "X_{}.test.npy".format(layer)))
    return (x_train, y_train), (x_test, y_test)
