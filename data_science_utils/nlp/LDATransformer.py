from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
from nltk import bigrams
from nltk import trigrams
from nltk.stem import WordNetLemmatizer
from data_science_utils import dataframe as df_utils


class LDATransformer:
    def __init__(self, token_column="tokens", lda_prefix="lda_", no_below=10, no_above=0.2,
                 iterations=100, num_topics=100, passes=10):
        """
        For pca strategy n_components,n_iter parameters are used. n_components determine
        how many columns resulting transformation will have

        :param strategy determines which strategy to take for reducing categorical variables
            Supported values are pca and label_encode

        :param n_components Valid for strategy="pca"

        :param n_iter Valid for strategy="pca"

        :param label_encode_combine Decides whether we combine all categorical column into 1 or not.
        """

        self.token_column = token_column
        self.lda_prefix = lda_prefix
        import multiprocessing
        self.cpus = int((multiprocessing.cpu_count() / 2) - 1)
        self.dictionary = None
        self.model = None
        self.no_below = no_below
        self.no_above = no_above
        self.iterations = iterations
        self.num_topics = num_topics
        self.passes = passes

    def fit(self, X, y=None):
        tokens = list(X[self.token_column].values)
        dictionary = corpora.Dictionary(tokens)
        self.dictionary = dictionary
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        print('Number of unique tokens after filtering: %d' % len(dictionary))
        X = X.copy()
        X['bow'] = X[self.token_column].apply(dictionary.doc2bow)
        from gensim.models.ldamulticore import LdaMulticore
        eval_every = None
        temp = dictionary[0]
        id2word = dictionary.id2token
        corpus = list(X['bow'].values)
        num_topics = 120

        model = LdaMulticore(corpus=corpus, id2word=id2word, chunksize=750, \
                             eta='auto', \
                             iterations=self.iterations, num_topics=self.num_topics, \
                             passes=self.passes, eval_every=eval_every, workers=self.cpus)
        self.model = model

    def partial_fit(self, X, y=None):
        self.fit(X, y)

    def transform(self, X, y='deprecated', copy=None):
        import pandas as pd
        X = X.copy()
        dictionary = self.dictionary
        X['bow'] = X[self.token_column].apply(dictionary.doc2bow)

        def bow_to_topics(bow):
            return self.model[bow]

        X['lda_topics'] = X.bow.apply(bow_to_topics)

        lda_df = pd.DataFrame.from_records(X['lda_topics'].apply(lambda x: {k: v for k, v in x}).values)
        lda_df.index = X['lda_topics'].index
        lda_df.columns = [self.lda_prefix + str(i) for i in range(0, self.num_topics)]
        X[list(lda_df.columns)] = lda_df
        df_utils.drop_columns_safely(X, ['bow', 'lda_topics', self.token_column], inplace=True)
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)