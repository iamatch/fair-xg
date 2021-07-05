import numpy as np

import scipy.stats as stats


class MetricsEvaluator:

    def __init__(self):
        pass

    @staticmethod
    def np_cast(y):
        return np.array(list(y))

    @staticmethod
    def round_d(d, nb_decimals):
        d = {k: round(v, nb_decimals) for k, v in d.items()}
        return d

    @staticmethod
    def compute_marge(p, n, alpha=0.05):
        if n:
            return stats.norm.ppf(1 - alpha) * np.sqrt(p * (1 - p) / n)
        else:
            return np.nan

    @staticmethod
    def compute_acc(y, y_pred):
        n = y.shape[0]
        n_true = np.equal(y, y_pred).sum()
        if n:
            p = n_true / n
        else:
            p = np.nan
        return p, n, n_true

    @staticmethod
    def compute_tpr(y, y_pred, v=1):
        indexes = np.where(y == v)[0]
        y = y[indexes]
        y_pred = y_pred[indexes]
        return MetricsEvaluator.compute_acc(y, y_pred)

    @staticmethod
    def compute_tnr(y, y_pred):
        return MetricsEvaluator.compute_tpr(y, y_pred, v=0)

    @staticmethod
    def compute_ppv(y, y_pred):
        return MetricsEvaluator.compute_tpr(y_pred, y)

    @staticmethod
    def compute_npv(y, y_pred):
        return MetricsEvaluator.compute_tpr(y_pred, y, v=0)

    @staticmethod
    def compute_metric(y, y_pred, metric_name, alpha):

        y = MetricsEvaluator.np_cast(y)
        y_pred = MetricsEvaluator.np_cast(y_pred)

        p, n, n_true = getattr(MetricsEvaluator, 'compute_' + metric_name)(y, y_pred)

        m = MetricsEvaluator.compute_marge(p, n, alpha)
        d = {
            'n': n,
            'n_true': n_true,
            'value': p,
            'ci': m,
            'alpha': alpha
        }
        return MetricsEvaluator.round_d(d, 3)
