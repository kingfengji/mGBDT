import numpy as np
from xgboost.core import DMatrix
from xgboost.training import train
from xgboost.sklearn import XGBModel, _objective_decorator


class OnlineXGB(XGBModel):
    def fit_increment(self, X, y, num_boost_round=1, params=None):
        trainDmatrix = DMatrix(X, label=y, nthread=self.n_jobs, missing=self.missing)
        extra_params = params
        params = self.get_xgb_params()
        if extra_params is not None:
            for k, v in extra_params.items():
                params[k] = v
        params.pop("n_estimators")

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:linear"
        else:
            obj = None

        if self._Booster is None:
            self._Booster = train(
                    params=params,
                    dtrain=trainDmatrix,
                    num_boost_round=num_boost_round,
                    obj=obj)
        else:
            self._Booster = train(
                    params=params,
                    dtrain=trainDmatrix,
                    num_boost_round=num_boost_round,
                    obj=obj,
                    xgb_model=self._Booster)
        return self

    def predict(self, X):
        if self._Booster is None:
            return np.full((X.shape[0],), self.base_score)
        return super(OnlineXGB, self).predict(X)
