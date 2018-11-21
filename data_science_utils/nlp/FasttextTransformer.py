from gensim.test.utils import common_texts
from gensim.models import FastText
import multiprocessing
import pandas as pd

from data_science_utils.misc import deep_map


class FasttextTransformer:
    def __init__(self, size=128, window=3, min_count=1, iter=20, min_n=2, max_n=5, word_ngrams=1,
                 workers=int(multiprocessing.cpu_count() / 2), ft_prefix="ft_", token_column=None, model=None):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.iter = iter
        self.min_n = min_n
        self.max_n = max_n
        self.word_ngrams = word_ngrams
        self.workers = workers
        self.token_column = token_column
        self.model = model
        assert type(self.token_column) == str
        self.ft_prefix = ft_prefix

    def fit(self, X, y='ignored'):
        if type(X) == pd.DataFrame:
            X = X[self.token_column].values

        if self.model is None:
            self.model = FastText(sentences=X, size=self.size, window=self.window, min_count=self.min_count,
                                  iter=self.iter, min_n=self.min_n, max_n=self.max_n, word_ngrams=self.word_ngrams,
                                  workers=self.workers)

    def partial_fit(self, X, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        if type(X) == pd.DataFrame:
            Input = X[self.token_column].values
        else:
            raise ValueError()
        tnsfr = lambda t: self.model.wv[t]
        X = X.copy()
        results = deep_map(tnsfr, Input)

        X[self.token_column] = results
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)