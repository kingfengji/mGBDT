from sklearn import datasets
import os.path as osp
import sys
sys.path.insert(0, "lib")

from mgbdt import MGBDTAntoEncoder
from mgbdt import MultiXGBModel
from mgbdt.utils.plot_utils import plot2d, plot3d
from mgbdt.utils.exp_utils import set_seed
from mgbdt.utils.log_utils import logger


if __name__ == "__main__":
    set_seed(0)
    des_dir = "outputs/scurve"

    n_points = 10000
    X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
    N_HIDDENS = 5

    # save input
    save_path = osp.join(des_dir, "input.jpg")
    plot3d(X, color=color, save_path=save_path)

    F = MultiXGBModel(input_size=3, output_size=N_HIDDENS, learning_rate=0.01, max_depth=3, num_boost_round=50)
    G = MultiXGBModel(input_size=N_HIDDENS, output_size=3, learning_rate=0.01, max_depth=3, num_boost_round=50)
    Ginv = MultiXGBModel(input_size=3, output_size=N_HIDDENS, learning_rate=0.01, max_depth=3, num_boost_round=50)
    net = MGBDTAntoEncoder(F, G, Ginv, target_lr=0.05, epsilon=0.3)
    logger.info("net architecture")
    logger.info(net)

    net.init(X, n_rounds=50)
    # net.init(X, n_rounds=5)
    net.fit(X, X, n_epochs=100)

    hiddens = net.get_hiddens(X)
    for i in range(0, N_HIDDENS):
        for j in range(i + 1, N_HIDDENS):
            des_path = osp.join(des_dir, "pred1.{}_{}.jpg".format(i + 1, j + 1))
            plot2d(hiddens[1][:, [i, j]], color=color, save_path=des_path)
    save_path = osp.join(des_dir, "pred2.jpg")
    plot3d(hiddens[2], color=color, save_path=save_path)

    import IPython
    IPython.embed()
