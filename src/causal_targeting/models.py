"""Three focused models with train-only preprocessing and explicit score definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, CATEGORICAL, CONTINUOUS, baseline_features


class TreeEncoder:
    """Native categorical splits, with train-only top-254 pooling.

    Pool rare/unseen categories into code 0; retain 254 most frequent tokens as
    1..254. Anonymous floats are category identities, not ordered quantities.
    """
    def fit(self, frame):
        self.levels = {}
        for col in CATEGORICAL:
            counts = frame[col].value_counts().rename_axis('token').reset_index(name='n')
            counts = counts.sort_values(['n', 'token'], ascending=[False, True])
            self.levels[col] = {v: j + 1 for j, v in enumerate(counts.token.iloc[:254])}
        return self

    def transform(self, frame):
        x = baseline_features(frame).astype(float)
        for col in CATEGORICAL:
            x[col] = x[col].map(self.levels[col]).fillna(0)
        return x.to_numpy()


class TLearner:
    def __init__(self, params, seed):
        self.params, self.seed = params, seed

    def fit(self, frame, outcome='conversion'):
        self.encoder = TreeEncoder().fit(frame)
        x = self.encoder.transform(frame)
        t = frame.treatment.to_numpy()
        y = frame[outcome].to_numpy()
        self.models = {}
        for arm in (0, 1):
            model = HistGradientBoostingClassifier(
                **self.params, random_state=self.seed, early_stopping=False,
                categorical_features=[col in CATEGORICAL for col in FEATURES])
            model.fit(x[t == arm], y[t == arm])
            self.models[arm] = model
        return self

    def predict(self, frame):
        x = self.encoder.transform(frame)
        return tuple(self.models[a].predict_proba(x)[:, 1] for a in (0, 1))


class LogisticInteractions:
    def __init__(self, c=1.0):
        self.c = c

    @staticmethod
    def design(z, t):
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        return sparse.hstack([z, sparse.csr_matrix(t), z.multiply(t)], format='csr')

    def fit(self, frame, outcome='conversion'):
        self.preprocessor = ColumnTransformer([
            ('continuous', StandardScaler(), CONTINUOUS),
            ('categorical', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                          max_categories=128, sparse_output=True), CATEGORICAL)
        ], sparse_threshold=1.0)
        z = self.preprocessor.fit_transform(baseline_features(frame))
        x = self.design(z, frame.treatment.to_numpy())
        self.model = LogisticRegression(C=self.c, solver='lbfgs', max_iter=300, tol=1e-6)
        self.model.fit(x, frame[outcome].to_numpy())
        self.n_iter = int(self.model.n_iter_[0])
        return self

    def predict(self, frame):
        z = self.preprocessor.transform(baseline_features(frame))
        return tuple(self.model.predict_proba(self.design(z, np.full(len(frame), a)))[:, 1]
                     for a in (0, 1))

    def coefficients(self):
        names = list(self.preprocessor.get_feature_names_out())
        names = names + ['treatment'] + ['treatment:' + n for n in names]
        return pd.DataFrame({'term': ['intercept'] + names,
                             'log_odds_coefficient': np.r_[self.model.intercept_, self.model.coef_[0]]})


def fit_assignment_model(frame, seed):
    """Diagnostic nuisance fit, separate from outcome models; no test outcomes."""
    encoder = TreeEncoder().fit(frame)
    x = encoder.transform(frame)
    model = HistGradientBoostingClassifier(
        max_iter=60, max_leaf_nodes=15, min_samples_leaf=500,
        l2_regularization=50, random_state=seed, early_stopping=False,
        categorical_features=[c in CATEGORICAL for c in FEATURES])
    model.fit(x, frame.treatment)
    return encoder, model
