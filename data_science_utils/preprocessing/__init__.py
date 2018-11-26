from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
def reduce_dimensions_by_ohe_svd(frame,n_components=2,n_iter=10):
    """

    :param frame: dataframe to reduce dimensions, should have only strings and no NaN/None
    :param n_components: Dimensions after reduction
    :param n_iter: Number of Iterations
    :return: pipeline which can reduce dimension of new data. Call as `new_dims = pipeline.transform(new_df)`
    """
    pca = TruncatedSVD(n_components=n_components,n_iter=n_iter)
    enc = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()
    ft = FunctionTransformer()
    pipeline = make_pipeline(enc,pca,scaler,ft)
    pipeline.fit(frame)
    return pipeline


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

        enc = OneHotEncoder(handle_unknown='ignore')
        scaler = StandardScaler()
        ft = FunctionTransformer()
        if self.strategy == "pca":
            assert X.shape[1] > 1
            pca = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter)
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
            X[results.columns] = results
            return X
        else:
            return np.concatenate((X, results), axis=1)

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)



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

class TargetBasedStatCategoricals:
    def __init__(self,colnames,target,stat_fn=np.mean,suffix="_agg",nan_fill=0):
        """
        """
        import multiprocessing
        self.cpus = int((multiprocessing.cpu_count()/2)-1)
        self.colnames=colnames
        self.target=target
        self.stat_fn=stat_fn
        self.nan_fill=nan_fill
        self.group_values=None
        self.suffix = suffix
    def fit(self, X, y=None):
        if not type(X)==pd.DataFrame:
            raise ValueError()
        X=X.copy()
        if self.nan_fill is not None:
            X[self.target] = X[self.target].fillna(self.nan_fill)
        gpv = X.groupby(self.colnames)[[self.target]].agg(self.stat_fn).reset_index(level=self.colnames)
        self.group_values = gpv
    def partial_fit(self, X, y=None):
        self.fit(X,y)

    def transform(self, X, y='deprecated', copy=None):
        import pandas as pd
        if not type(X)==pd.DataFrame:
            raise ValueError()
        X = X.copy()
        Input = X[self.colnames]
        result = Input.merge(self.group_values,on=self.colnames,how="left")
        result.rename(columns={self.target:"_".join(self.group_values)+self.suffix},inplace=True)
        result.index = X.index
        df_utils.drop_columns_safely(result,self.group_values,inplace=True)
        X[result.columns] = result
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X,y=None):
        self.fit(X,y)
        return self.transform(X,y)


import pandas as pd
import numpy as np
from gensim import models, corpora
from data_science_utils import dataframe as df_utils


class NamedColumnSelector:
    def __init__(self, include_columns=None,exclude_columns=None):
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns

    def fit(self, X, y='ignored'):
        pass

    def partial_fit(self, X, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        if type(X) != pd.DataFrame:
            raise ValueError()
        if self.include_columns is not None:
            X=X[self.include_columns]
        if self.exclude_columns is not None:
            df_utils.drop_columns_safely(X,self.exclude_columns,inplace=True)
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X,y)

