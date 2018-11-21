from gensim.test.utils import common_texts
from gensim.models import FastText
import multiprocessing
import pandas as pd
import numpy as np
from gensim import models, corpora
from data_science_utils import dataframe as df_utils


class FasttextTfIdfTransformer:
    def __init__(self, size=128, window=3, min_count=1, iter=20, min_n=2, max_n=5, word_ngrams=2,
                 workers=int(multiprocessing.cpu_count() / 2), ft_prefix="ft_", token_column=None,
                 model=None, dictionary=None, tfidf=None):
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
        self.tfidf = tfidf
        assert type(self.token_column) == str
        self.ft_prefix = ft_prefix
        self.dictionary = dictionary

    def fit(self, X, y='ignored'):
        from gensim.models import TfidfModel
        if type(X) == pd.DataFrame:
            X = X[self.token_column].values

        if self.model is None:
            self.model = FastText(sentences=X, size=self.size, window=self.window, min_count=self.min_count,
                                  iter=self.iter, min_n=self.min_n, max_n=self.max_n, word_ngrams=self.word_ngrams,
                                  workers=self.workers)
        if self.dictionary is None:
            dictionary = corpora.Dictionary(X)
            self.dictionary = dictionary
            self.dictionary.filter_extremes(no_below=3, no_above=0.25)
        if self.tfidf is None:
            bows = list(map(self.dictionary.doc2bow, X))
            tfidf = TfidfModel(bows, normalize=True)
            self.tfidf = tfidf

    def partial_fit(self, X, y=None):
        self.fit(X, y='ignored')

    def transform(self, X, y='ignored'):
        if type(X) == pd.DataFrame:
            Input = X[self.token_column].values
        else:
            raise ValueError()
        X = X.copy()

        def tokens2tfidf(token_array, tfidf, dictionary):
            id2tfidf = {k: v for k, v in tfidf[dictionary.doc2bow(token_array)]}
            token2tfidf = {dictionary.id2token[k]: v for k, v in id2tfidf.items()}
            return [token2tfidf[token] if token in token2tfidf else 0 for token in token_array]

        def tokens2vec(token_array, fasttext_model):
            if len(token_array) == 0:
                return np.full(self.size, 0)
            return [fasttext_model.wv[token] for token in token_array]

        t2tfn = lambda tokens: tokens2tfidf(tokens, self.tfidf, self.dictionary)
        tfidfs = list(map(t2tfn, Input))
        ft_fn = lambda tokens: tokens2vec(tokens, self.model)
        ft_vecs = list(map(ft_fn, Input))

        def doc2vec(ftv, tfidf_rep):
            if np.sum(ftv) == 0:
                return np.full(self.size, 0)
            if np.sum(tfidf_rep) == 0:
                return np.average(ftv, axis=0)
            return np.average(ftv, axis=0, weights=tfidf_rep)

        results = list(map(lambda x: doc2vec(x[0], x[1]), zip(ft_vecs, tfidfs)))
        text_df = pd.DataFrame(list(map(list, results)))
        text_df.columns = [self.ft_prefix + str(i) for i in range(0, self.size)]
        X[list(text_df.columns)] = text_df
        df_utils.drop_columns_safely(X, [self.token_column], inplace=True)
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y='ignored'):
        self.fit(X)
        return self.transform(X)