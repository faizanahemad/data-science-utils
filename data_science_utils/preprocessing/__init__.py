from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import more_itertools
from data_science_utils.dataframe import get_specific_cols
import gc

from data_science_utils.nlp import FasttextTfIdfTransformer


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
    def __init__(self,colnames,target,suffix="_agg",nan_fill=0,inplace=True):
        """

        :param colnames: categorical column names
        :param target: target column
        :param stat: weight_of_evidence or mean, default: weight_of_evidence
            See [weight_of_evidence](https://pkghosh.wordpress.com/2017/10/09/combating-high-cardinality-features-in-supervised-machine-learning/)
        :param suffix: new columns's suffix
        :param nan_fill: filler value for nan
        :param inplace:
        """
        import multiprocessing
        self.cpus = int((multiprocessing.cpu_count()/2)-1)
        self.colnames=colnames
        self.target=target
        self.nan_fill=nan_fill
        self.group_values=None
        self.suffix = suffix
        self.inplace=inplace
    def fit(self, X, y=None):
        if not type(X)==pd.DataFrame:
            raise ValueError()
        if not self.inplace:
            X = X.copy()
        if self.nan_fill is not None:
            X[self.target] = X[self.target].fillna(self.nan_fill)

        gpv = X.groupby(self.colnames)[[self.target]].agg(["mean","std"]).reset_index(level=self.colnames)
        overall_means = X[self.target].mean()
        counts = X.groupby(self.cols)[self.target].agg(['count'])

        gpv.columns = list(map(lambda x: x[0] + x[1], list(gpv.columns)))
        mean_col = self.target + "mean"
        overall_means.index = mean_col
        std_cols = self.target + "std"
        woe_cols = self.target + "woe"
        gpv[woe_cols] = np.clip(np.log(gpv[mean_col] / overall_means), -1e8, 1e8)
        gpv['count'] = counts.values
        self.group_values = gpv
    def partial_fit(self, X, y=None):
        self.fit(X,y)

    def transform(self, X, y='deprecated', copy=None):
        import pandas as pd
        if not type(X)==pd.DataFrame:
            raise ValueError()
        if not self.inplace:
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
    def __init__(self, columns=None,prefixes=None,suffixes=None):
        self.columns = columns
        self.prefixes = prefixes
        self.suffixes = suffixes

    def get_cols_(self, X):
        cols = list(self.columns)
        if self.prefixes is not None:
            for pf in self.prefixes:
                cols.extend(get_specific_cols(X, prefix=pf))
        if self.suffixes is not None:
            for pf in self.suffixes:
                cols.extend(get_specific_cols(X, suffix=pf))
        cols = list(set(cols))
        return cols

    def fit(self, X, y='ignored'):
        pass

    def partial_fit(self, X, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        if type(X) != pd.DataFrame:
            raise ValueError()
        X = X[self.get_cols_(X)]
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X,y)



from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import BaseWrapper
from keras.layers import Dense, Activation
from keras import optimizers


class NeuralCategoricalFeatureTransformer:

    def __init__(self, cols,
                 include_input_as_output=True, target_columns=None,
                 n_layers=2, n_components=32, n_iter=150, nan_fill="", verbose=0,
                 prefix="nncat_",
                 save_file=None,inplace=True,
                 skip_fit=False,skip_transform=False):
        """
        """
        self.model = None
        self.n_components = n_components
        self.n_iter = n_iter
        self.cols = cols
        assert type(self.cols) == list
        if type(self.cols[0]) == int:
            raise NotImplementedError()
        self.n_layers = n_layers
        self.target_columns = target_columns
        self.nan_fill = nan_fill
        self.include_input_as_output = include_input_as_output
        if not include_input_as_output and target_columns is None:
            raise ValueError("We need either input columns as targets or a different target column atleast")
        self.enc = None
        self.verbose = verbose
        self.prefix = prefix
        self.save_file = save_file
        self.inplace=inplace
        self.skip_fit = skip_fit
        self.skip_transform = skip_transform
        if save_file is not None:
            raise NotImplementedError()

    def fit(self, X, y=None):
        print("Neural Categorical fit start at: %s" % (str(pd.datetime.now())))
        if self.skip_fit:
            return self
        if type(X) != pd.DataFrame:
            raise NotImplementedError()

        X = X.copy()
        if type(self.cols[0]) == str:
            X[self.cols] = X[self.cols].fillna(self.nan_fill)
        else:
            raise ValueError("Please provide colidx parameter")

        only_numeric_target = True
        if self.target_columns is not None:
            for colname, dtype in X.head()[self.target_columns].dtypes.reset_index(level=0).values:
                if dtype == 'object':
                    only_numeric_target = False
        else:
            only_numeric_target = False
        only_string_input = True
        for colname, dtype in X.head()[self.cols].dtypes.reset_index(level=0).values:
            if dtype != 'object':
                only_string_input = False

        validation_split = 0.4
        if only_numeric_target and only_string_input:
            cnt_col = self.target_columns[0]
            target_columns = list(self.target_columns)
            overall_means = X[self.target_columns].mean()
            Xt = X.copy()
            # for all cols
            counts = X.groupby(self.cols)[cnt_col].agg(['count'])

            X = X.groupby(self.cols)[self.target_columns].agg(['mean','std']).reset_index()
            X.columns = list(map(lambda x:x[0]+x[1],list(X.columns)))
            mean_cols = [t + "mean" for t in self.target_columns]
            overall_means.index = mean_cols
            std_cols = [t + "std" for t in self.target_columns]
            woe_cols = [t + "woe" for t in self.target_columns]
            X[woe_cols] = np.clip(np.log(X[mean_cols]/overall_means),-1e8,1e8)
            X['count'] = counts.values
            self.target_columns = mean_cols + std_cols + ['count'] + woe_cols
            # for each col
            for col in self.cols:
                gps = Xt.groupby([col])
                counts = gps[cnt_col].agg(['count'])
                gp = gps[target_columns].agg(['mean', 'std']).reset_index()
                gp.columns = list(map(lambda x: x[0] + x[1], list(gp.columns)))
                mean_cols = [t + "mean" for t in target_columns]
                std_cols = [t + "std" for t in target_columns]

                woe_cols = [t + "_" + col + "_" + "woe" for t in target_columns]
                gp[woe_cols] = np.clip(np.log(gp[mean_cols] / overall_means), -1e8, 1e8)
                gp[col + "_" + 'count'] = counts.values
                gp.rename(columns=dict(zip(mean_cols, [col + "_" + t for t in mean_cols])), inplace=True)
                mean_cols = [col + "_" + t for t in mean_cols]
                gp.rename(columns=dict(zip(std_cols, [col + "_" + t for t in std_cols])), inplace=True)
                std_cols = [col + "_" + t for t in std_cols]
                cols = woe_cols + [col + "_" + 'count'] + mean_cols + std_cols
                X = X.merge(gp,on=col)
                self.target_columns = cols + self.target_columns



        X = X.sample(frac=1)
        if self.target_columns is None:
            extra_col = list(set(X.columns) - set(self.cols))[0]
            X = X[self.cols+[extra_col]]
            X = X.groupby(self.cols).agg(['count']).reset_index()
            X.columns = list(map(lambda x: x[0] + x[1], list(X.columns)))
            cols = list(X.columns)
            cols[-1] = "count"
            X.columns = cols
            self.target_columns = ['count']

        Inp = X[self.cols]
        ouput_cols = list(self.target_columns)
        if self.include_input_as_output:
            ouput_cols = list(self.cols) + (self.target_columns if self.target_columns is not None else [])
        if self.target_columns is not None:
            X[self.target_columns] = X[self.target_columns].fillna(X[self.target_columns].mean())
        Output = X[ouput_cols]



        # we will add count of each group,
        # we will add std-dev of target columns
        # we will add weight of evidence

        loss = "mean_squared_error"
        es = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=0, )
        es2 = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=6, verbose=0, )
        scaler = MinMaxScaler()

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

        numeric_cols_target = []
        string_cols_target = []
        for colname, dtype in Output.dtypes.reset_index(level=0).values:
            if dtype != "object":
                numeric_cols_target.append(colname)
            else:
                string_cols_target.append(colname)
        assert X.shape[1] > 1

        out_enc = OneHotEncoder(sparse=False)

        if len(numeric_cols_target)>0:
            Output[numeric_cols_target] = scaler.fit_transform(Output[numeric_cols_target])
        if len(string_cols_target)>0:
            ohe_ouputs = out_enc.fit_transform(Output[string_cols_target])
            outout_string_dummies = pd.DataFrame(ohe_ouputs)
            outout_string_dummies.index = Output.index
            if len(numeric_cols_target) > 0:
                outout_string_dummies[numeric_cols_target] = Output[numeric_cols_target]
        if len(string_cols_target)>0:
            Output = outout_string_dummies
        Inp = enc.fit_transform(Inp)

        # total_memory = 2*8*X.shape[0]*X.shape[0]/1000000000
        # print("Total Memory in GB %.3f"%total_memory)
        # https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue
        # https://stackoverflow.com/questions/41538692/using-sparse-matrices-with-keras-and-tensorflow

        # Output.memory_usage(deep=True).sum()/1000000000
        # Inp.nbytes/1000000000
        # 8*Inp.shape[0]*Inp.shape[1]/1000000000

        input_layer = Input(shape=(Inp.shape[1],))
        encoded = Dense(self.n_components * 2, activation='elu')(input_layer)

        encoded = Dense(self.n_components, activation='elu')(encoded)

        decoded = Dense(self.n_components * 2, activation='elu')(encoded)
        decoded = Dense(Output.shape[1], activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)

        adam = optimizers.Adam(lr=0.01, clipnorm=4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.05, amsgrad=False)
        autoencoder.compile(optimizer=adam, loss=loss)
        print("Shape of Input to Neural Network: %s, Output shape: %s"%(Inp.shape,Output.shape))
        validation_size=int(Inp.shape[0]*validation_split)
        train_size = Inp.shape[0] - validation_size
        autoencoder.fit(Inp[:train_size], Output[:train_size],
                        epochs=self.n_iter,
                        batch_size=4096,
                        shuffle=True,
                        validation_data=(Inp[train_size:], Output[train_size:]),
                        verbose=self.verbose,
                        callbacks=[es])
        autoencoder.fit(Inp[train_size:], Output[train_size:],
                        epochs=self.n_iter,
                        batch_size=4096,
                        shuffle=True,
                        validation_data=(Inp[:train_size], Output[:train_size]),
                        verbose=self.verbose,
                        callbacks=[es2])
        self.model = encoder
        self.enc = enc
        gc.collect()
        print("Neural Categorical fit done at: %s" % (str(pd.datetime.now())))
        return self

    def partial_fit(self, X, y=None):
        self.fit(X, y)

    def transform(self, X, y='deprecated', copy=None):
        if self.skip_transform:
            return X
        if type(X) == pd.DataFrame:
            if type(self.cols[0]) == str:
                Inp = X[self.cols].fillna(self.nan_fill)
        else:
            raise NotImplementedError()
        Inp = self.enc.transform(Inp)
        results = pd.DataFrame(self.model.predict(Inp))
        results.index = X.index

        columns = list(map(lambda x: self.prefix + str(x), range(0, results.shape[1])))
        results.columns = columns
        if not self.inplace:
            X = X.copy()
        X[results.columns] = results
        gc.collect()
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        print("Neural Categorical fit-transforms start at: %s" % (str(pd.datetime.now())))
        self.fit(X, y)
        res = self.transform(X, y)
        print("Neural Categorical fit-transforms done at: %s" % (str(pd.datetime.now())))
        return res

