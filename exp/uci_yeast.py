import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.insert(0, "lib")

from mgbdt import MGBDT
from mgbdt import MultiXGBModel, LinearModel
from mgbdt.datasets import uci_yeast
from mgbdt.utils.exp_utils import set_seed
from mgbdt.utils.log_utils import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", dest="n_layers", default=3, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    n_layers = args.n_layers
    set_seed(0)

    X, y = uci_yeast.load_data()
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    accs = []
    for ci, (train_index, test_index) in enumerate(skf.split(X, y)):
        logger.info("[progress] cv={}/{}".format(ci + 1, n_splits))
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
        logger.info("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))

        net = MGBDT(loss=None, target_lr=1.0, epsilon=0.3)
        net.add_layer("tp_layer",
                F=MultiXGBModel(input_size=8, output_size=16, learning_rate=0.1, max_depth=5, num_boost_round=5),
                G=None)
        for i in range(n_layers - 2):
            net.add_layer("tp_layer",
                    F=MultiXGBModel(input_size=16, output_size=16, learning_rate=0.1, max_depth=5, num_boost_round=5),
                    G=MultiXGBModel(input_size=16, output_size=16, learning_rate=0.1, max_depth=5, num_boost_round=5))
        net.add_layer("bp_layer",
                F=LinearModel(input_size=16, output_size=10, learning_rate=0.1, loss="CrossEntropyLoss"))
        logger.info("[net architecture]")
        logger.info(net)
        net.init(x_train, n_rounds=5)
        net.fit(x_train, y_train, n_epochs=50, eval_sets=[(x_test, y_test)], eval_metric="accuracy")

        y_proba = net.forward(x_test)
        acc = accuracy_score(y_test, y_proba.argmax(axis=1))
        accs.append(acc * 100)
        logger.info("accs={}".format(accs))

    logger.info("Mean(acc)={:.2f}%".format(np.mean(accs)))
    logger.info("Std(acc)={:.2f}%".format(np.std(accs)))
    import IPython
    IPython.embed()
