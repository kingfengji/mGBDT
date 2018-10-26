import numpy as np
from joblib import Parallel, delayed

from .online_xgb import OnlineXGB


class MultiXGBModel:
    def __init__(self, input_size, output_size, learning_rate, max_depth=5, num_boost_round=1, force_no_parallel=False, **kwargs):
        """
        model: (XGBoost)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_boost_round = num_boost_round
        self.force_no_parallel = force_no_parallel
        self.models = []
        for i in range(self.output_size):
            single_model = OnlineXGB(max_depth=max_depth, silent=True, n_jobs=-1, learning_rate=learning_rate, **kwargs)
            self.models.append(single_model)

    def __repr__(self):
        return "MultiXGBModel(input_size={}, output_size={}, learning_rate={:.3f}, max_depth={}, num_boost_round={})".format(
                self.input_size, self.output_size, self.learning_rate, self.max_depth, self.num_boost_round)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit(self, X, y, num_boost_round=None, params=None):
        assert X.shape[1] == self.input_size
        if self.force_no_parallel:
            self._fit_serial(X, y, num_boost_round, params)
        else:
            self._fit_parallel(X, y, num_boost_round, params)

    def predict(self, X):
        assert X.shape[1] == self.input_size
        out = self._predict_serial(X)
        if not self.force_no_parallel and (X.shape[0] <= 10000 or X.shape[1] <= 10):
            out = self._predict_parallel(X)
        else:
            out = self._predict_serial(X)
        return out

    def _fit_serial(self, X, y, num_boost_round, params):
        if num_boost_round is None:
            num_boost_round = self.num_boost_round
        for i in range(self.output_size):
            self.models[i].n_jobs = -1
        for i in range(self.output_size):
            self.models[i].fit_increment(X, y[:, i], num_boost_round=num_boost_round, params=None)

    def _fit_parallel(self, X, y, num_boost_round, params):
        if num_boost_round is None:
            num_boost_round = self.num_boost_round
        for i in range(self.output_size):
            self.models[i].n_jobs = 1
        Parallel(n_jobs=-1, verbose=False, backend="threading")(
                delayed(model.fit_increment)(X, y[:, i], num_boost_round=num_boost_round, params=None)
                for i, model in enumerate(self.models))

    def _predict_serial(self, X):
        for i in range(self.output_size):
            self.models[i].n_jobs = -1
        pred = np.empty((X.shape[0], self.output_size), dtype=np.float64)
        for i in range(self.output_size):
            pred[:, i] = self.models[i].predict(X)
        return pred

    def _predict_parallel(self, X):
        for i in range(self.output_size):
            self.models[i].n_jobs = 1
        pred = Parallel(n_jobs=-1, verbose=False, backend="threading")(
                delayed(model.predict)(X)
                for i, model in enumerate(self.models))
        pred = np.asarray(pred, dtype=np.float64).T
        return pred
