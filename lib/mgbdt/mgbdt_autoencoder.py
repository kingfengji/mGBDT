import numpy as np
from termcolor import colored

from mgbdt.model import MultiXGBModel
from mgbdt.utils.log_utils import logger


class MGBDTAntoEncoder:
    def __init__(self, F, G, Ginv, target_lr=1.0, epsilon=0.01, verbose=False):
        """
        F: the encoder layer
        G: the decoder layer
        """
        self.F = F
        self.G = G
        self.Ginv = Ginv
        self.target_lr = target_lr
        self.epsilon = epsilon
        self.n_layers = 2
        self.verbose = verbose

    def init(self, X, n_rounds=1, learning_rate=None, max_depth=None, batch=0):
        self.log("[init][start]")
        params = {}
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if max_depth is not None:
            params["max_depth"] = max_depth
        rand_out = np.random.randn(X.shape[0], self.F.output_size)
        self.log("[init] fit F: X -> random")
        self.F.fit(X, rand_out, num_boost_round=n_rounds, params=params.copy())
        self.log("[init] H = F(X)")
        H = self.F.predict(X)
        self.log("[init] fit G: H -> random")
        if isinstance(self.G, MultiXGBModel):
            self.G.fit(H, X, num_boost_round=n_rounds, params=params.copy())
        self.log("[init][end]")

    def forward(self, X):
        H = self.F.predict(X)
        out = self.G.predict(H)
        return out

    def get_hiddens(self, X):
        """
        Return
        ------
        hiddens: [X, H, Z]
            X is inputs
            H is the encoded data
            Z is the decoded data
        """
        layers = [None, self.F, self.G]
        hiddens = [None for _ in range(self.n_layers + 1)]
        hiddens[0] = X
        for i in range(1, self.n_layers + 1):
            hiddens[i] = layers[i].predict(hiddens[i - 1])
        return hiddens

    def fit_inverse_mapping(self, X, y):
        if self.Ginv is None:
            return
        F, G, Ginv = self.F, self.G, self.Ginv
        # 1.1 Compute Hiddens
        self.log("1.1 Compute hidden units")
        H = F.predict(X)
        # 1.1 Traning G inverse
        self.log("1.2 Fit inverse mapping")
        H_noise = H + self.gen_noise(H)
        Ginv.fit(G.predict(H_noise), H_noise)
        return H

    def fit_forward_mapping(self, X, y):
        self.log("[fit_forward_mapping][start] X.shape={}, y.shape={}".format(X.shape, y.shape))
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        F, G, Ginv = self.F, self.G, self.Ginv
        if Ginv is None:
            Ginv = F
        # 2.1 Compute hidden units in prediction
        self.log("2.1 Compute hidden units")
        # h is the encoded data, z is the decoded data
        H = self.F.predict(X)
        Z = self.G.predict(H + self.gen_noise(H))
        # 2.2 Compute the targets
        self.log("2.2 Compute the targets")
        Zt = Z - self.target_lr * (Z - y) * 2
        Ht = Ginv.predict(Zt)
        # 2.3 Training feedward mapping
        self.log("2.3 Fitting f, g")
        F.fit(X + self.gen_noise(X), Ht)
        G.fit(H, Zt)
        self.log("[fit_forward_mapping][end]")
        return [X, H, Z], [None, Ht, Zt]

    def fit(self, X, y, n_epochs=1, eval_sets=None, batch=0, callback=None, n_eval_epochs=1):
        if eval_sets is None:
            eval_sets = ()
        self.log("[epoch={}/{}] training_loss={:.6f}".format(0, n_epochs, self.score(X, y)), "green")
        if batch == 0:
            batch = len(X)
        for _epoch in range(1, n_epochs + 1):
            if batch < len(X):
                perm = np.random.permutation(len(X))
            else:
                perm = list(range(len(X)))
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                self.fit_inverse_mapping(X[rand_idx], y[rand_idx])
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                hiddens, targets = self.fit_forward_mapping(X[rand_idx], y[rand_idx])
            if n_eval_epochs > 0 and _epoch % n_eval_epochs == 0:
                self.log("[epoch={}/{}] training_loss={:.6f}".format(_epoch, n_epochs, self.score(X, y)), "green")
                for (x_test, y_test) in eval_sets:
                    self.log("[epoch={}/{}] testing_loss={:.6f}".format(_epoch, n_epochs, self.score(x_test, y_test)), "green")
            if callback is not None:
                callback(self, _epoch, X, y, eval_sets, hiddens=hiddens, targets=targets)
            si += batch

    def score(self, X, y):
        out = self.forward(X)
        return np.mean((out - y)**2)

    def mse(self, a, b):
        return np.mean((a - b)**2)

    def gen_noise(self, data):
        return np.random.randn(*data.shape) * self.epsilon

    def log(self, msg, color=None):
        if color is None:
            if self.verbose:
                logger.info(msg)
        else:
            logger.info(colored("{}".format(msg), color))

    def __repr__(self):
        res = "\n(MGBDTAntoEncoder STRUCTURE BEGIN)\n"
        res += "target_lr={:.3f}, epsilon={:.3f}\n".format(self.target_lr, self.epsilon)
        res += "F={}\n".format(self.F)
        res += "G={}\n".format(self.G)
        res += "Ginv={}\n".format(self.Ginv)
        res += "(MGBDTAntoEncoder STRUCTURE END)"
        return res
