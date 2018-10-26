import sys
sys.path.insert(0, "lib")

from mgbdt import MGBDT
from mgbdt import MultiXGBModel, LinearModel
from mgbdt.datasets import uci_adult
from mgbdt.utils.exp_utils import set_seed
from mgbdt.utils.log_utils import logger


if __name__ == "__main__":
    set_seed(666)

    (x_train, y_train), (x_test, y_test) = uci_adult.load_data()
    logger.info("x_train.shape={}, y_train.shape={}".format(x_train.shape, y_train.shape))
    logger.info("x_test.shape={}, y_test.shape={}".format(x_test.shape, y_test.shape))

    net = MGBDT(loss=None, target_lr=1.0, epsilon=0.3)
    net.add_layer("tp_layer",
            F=MultiXGBModel(input_size=113, output_size=128, learning_rate=0.3, max_depth=5, num_boost_round=5),
            G=None)
    net.add_layer("tp_layer",
            F=MultiXGBModel(input_size=128, output_size=128, learning_rate=0.3, max_depth=5, num_boost_round=5),
            G=MultiXGBModel(input_size=128, output_size=128, learning_rate=0.3, max_depth=5, num_boost_round=5))
    net.add_layer("bp_layer",
            F=LinearModel(input_size=128, output_size=2, learning_rate=0.03, loss="CrossEntropyLoss"))
    logger.info("[net architecture]")
    logger.info(net)
    net.init(x_train, n_rounds=5)
    net.fit(x_train, y_train, n_epochs=40, eval_sets=[(x_test, y_test)], eval_metric="accuracy")

    import IPython
    IPython.embed()
