from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp
import sys
sys.path.insert(0, "lib")

from mgbdt import MGBDT
from mgbdt import MultiXGBModel
from mgbdt.utils.plot_utils import plot2d, plot3d
from mgbdt.utils.exp_utils import set_seed
from mgbdt.utils.log_utils import logger


if __name__ == "__main__":
    set_seed(0)
    des_dir = "outputs/circle"

    n_samples = 15000
    x_all, y_all = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.04, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=0, stratify=y_all)
    print("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    print("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))

    net = MGBDT(loss="CrossEntropyLoss", target_lr=1.0, epsilon=0.1)
    # F is the forward mapping, G is the inverse mapping
    # The first layer don't need invserse mapping, so set G=None for the first layer
    # 2 -> 5 -> 3 -> 2
    net.add_layer("tp_layer",
            F=MultiXGBModel(input_size=2, output_size=5, learning_rate=0.1, max_depth=5, num_boost_round=5),
            G=None)
    net.add_layer("tp_layer",
            F=MultiXGBModel(input_size=5, output_size=3, learning_rate=0.1, max_depth=5, num_boost_round=5),
            G=MultiXGBModel(input_size=3, output_size=5, learning_rate=0.1, max_depth=5, num_boost_round=5))
    net.add_layer("tp_layer",
            F=MultiXGBModel(input_size=3, output_size=2, learning_rate=0.1, max_depth=5, num_boost_round=5),
            G=MultiXGBModel(input_size=2, output_size=3, learning_rate=0.1, max_depth=5, num_boost_round=5))

    logger.info(net)
    net.init(x_train, n_rounds=5)
    net.fit(x_train, y_train, n_epochs=50, eval_sets=[(x_test, y_test)], eval_metric="accuracy")

    colors = np.asarray(["red", "blue"])
    # save the input dataset
    save_path = osp.join(des_dir, "input.jpg")
    plot2d(x_test, color=colors[y_test], save_path=save_path)

    # save the hidden output
    save_path = osp.join(des_dir, "pred2.jpg")
    hiddens = net.get_hiddens(x_test)
    plot3d(hiddens[2], color=colors[y_test], save_path=save_path)

    import IPython
    IPython.embed()
