
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
    def __init__(self,colnames,target,stat_fn=np.mean,suffix="_agg",fill_na=0):
        """
        """
        import multiprocessing
        self.cpus = int((multiprocessing.cpu_count()/2)-1)
        self.colnames=colnames
        self.target=target
        self.stat_fn=stat_fn
        self.fill_na=fill_na
        self.group_values=None
        self.suffix = suffix
    def fit(self, X, y=None):
        if not type(X)==pd.DataFrame:
            raise ValueError()
        X=X.copy()
        if self.fill_na is not None:
            X[self.target] = X[self.target].fillna(self.fill_na)
        gpv = X.groupby(self.colnames)[[self.target]].agg(self.stat_fn).reset_index(level=self.colnames)
        self.group_values = gpv
    def partial_fit(self, X, y=None):
        self.fit(X,y)

    def transform(self, X, y='deprecated', copy=None):
        import pandas as pd
        if not type(X)==pd.DataFrame:
            raise ValueError()
        Input = X[self.colnames]
        result = Input.merge(self.group_values,on=self.colnames,how="left")
        result.rename(columns={self.target:"_".join(self.group_values)+self.suffix},inplace=True)
        df_utils.drop_columns_safely(result,self.group_values,inplace=True)
        X[result.columns] = result
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X,y=None):
        self.fit(X,y)
        return self.transform(X,y)