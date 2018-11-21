from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_science_utils import dataframe as df_utils
import numpy as np


class CategoricalFeatureTransformer:
    def __init__(self, colidx, strategy="pca", n_components=32, n_iter=25, label_encode_combine=True, nan_fill="",
                 prefix="cat_"):
        """
        For pca strategy n_components,n_iter parameters are used. n_components determine
        how many columns resulting transformation will have

        :param strategy determines which strategy to take for reducing categorical variables
            Supported values are pca and label_encode

        :param n_components Valid for strategy="pca"

        :param n_iter Valid for strategy="pca"

        :param label_encode_combine Decides whether we combine all categorical column into 1 or not.
        """
        self.strategy = strategy
        self.pipeline = None
        self.n_components = n_components
        self.n_iter = n_iter
        self.colidx = colidx
        assert type(self.colidx) == list
        if type(self.colidx[0]) == int:
            raise NotImplementedError()
        self.prefix = prefix
        self.nan_fill = nan_fill

    def fit(self, X, y=None):
        if type(X) == pd.DataFrame:
            if type(self.colidx[0]) == str:
                X = X[self.colidx].fillna(self.nan_fill)
            else:
                raise ValueError("Please provide colidx parameter")
        else:
            raise NotImplementedError()

        if self.strategy == "pca":
            assert X.shape[1] > 1
            pca = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter)
            enc = OneHotEncoder(handle_unknown='ignore')
            scaler = StandardScaler()
            ft = FunctionTransformer()
            pipeline = make_pipeline(enc, pca, scaler, ft)
            pipeline.fit(X)
        elif self.strategy == "label_encode":
            raise NotImplementedError()
        else:
            raise ValueError("Unknown strategy %s is not supported" % (self.strategy))
        self.pipeline = pipeline

    def partial_fit(self, X, y=None):
        self.fit(X, y)

    def transform(self, X, y='deprecated', copy=None):
        if type(X) == pd.DataFrame:
            if type(self.colidx[0]) == str:
                Input = X[self.colidx].fillna(self.nan_fill)
        else:
            raise NotImplementedError()
        results = self.pipeline.transform(Input)
        results = np.array(results)
        if len(results.shape) == 1:
            results = results.reshape(-1, 1)
        if type(X) == pd.DataFrame:
            columns = list(map(lambda x: self.prefix + str(x), range(0, self.n_components)))
            results = pd.DataFrame(results, columns=columns)
            results.index = X.index
            X = X.copy()
            df_utils.drop_columns_safely(X, self.colidx, inplace=True)
            X[results.columns] = results
            return X
        else:
            return np.concatenate((X, results), axis=1)

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)