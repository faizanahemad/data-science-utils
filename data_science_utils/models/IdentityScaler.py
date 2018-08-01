class IdentityScaler():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        return self

    def transform(self, X, y='deprecated', copy=None):
        return X

    def inverse_transform(self, X, copy=None):
        return X

    def fit_transform(self, X):
        return X