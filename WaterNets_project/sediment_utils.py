import numpy as np


class SimpleScaler:
    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
