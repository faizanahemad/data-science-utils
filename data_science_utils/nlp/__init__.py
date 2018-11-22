import re
import pandas as pd
import re
import ast
from gensim import models, corpora
import nltk
from fastnumbers import isfloat
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import MmCorpus
from gensim.models.ldamodel import LdaModel
from nltk import bigrams
from nltk import trigrams
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet
from nltk import ngrams
import more_itertools

NUM = "_NUM_"
UNIT_OF_LENGTH = "_UNIT_OF_LENGTH_"
UNIT_OF_VOLUME = "_UNIT_OF_VOLUME_"
UNIT_OF_TIME = "_UNIT_OF_TIME_"
UNIT_OF_MASS = "_UNIT_OF_MASS_"
UNIT_OF_ELECTRICITY = "_UNIT_OF_ELECTRICITY_"

stopwords_list = stopwords.words('english')


def get_number_base_words():
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
               "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
               "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
               "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four",
               "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty", "forty-one",
               "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight",
               "forty-nine", "fifty", "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six",
               "fifty-seven", "fifty-eight", "fifty-nine", "sixty", "sixty-one", "sixty-two", "sixty-three",
               "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy",
               "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six",
               "seventy-seven", "seventy-eight", "seventy-nine", "eighty", "eighty-one", "eighty-two", "eighty-three",
               "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine", "ninety",
               "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven",
               "ninety-eight", "ninety-nine", ]

    nums = list(range(0, 100))
    n2 = [l.replace("-", "") for l in numbers]
    n3 = [l.replace("-", " ") for l in numbers]
    scales = ["hundred", "thousand", "lakh", "million", "crore", "billion", "trillion"]
    scale_nums = [1e2, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12]

    d1 = {tupl[0]: tupl[1] for tupl in zip(numbers, nums)}
    d2 = {tupl[0]: tupl[1] for tupl in zip(n2, nums)}
    d3 = {tupl[0]: tupl[1] for tupl in zip(n3, nums)}
    d4 = {tupl[0]: int(tupl[1]) for tupl in zip(scales, scale_nums)}
    nw = {**d4, **d1, **d2, **d3}
    return nw


def get_number_words():
    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
               "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
               "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
               "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four",
               "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty", "forty-one",
               "forty-two", "forty-three", "forty-four", "forty-five", "forty-six", "forty-seven", "forty-eight",
               "forty-nine", "fifty", "fifty-one", "fifty-two", "fifty-three", "fifty-four", "fifty-five", "fifty-six",
               "fifty-seven", "fifty-eight", "fifty-nine", "sixty", "sixty-one", "sixty-two", "sixty-three",
               "sixty-four", "sixty-five", "sixty-six", "sixty-seven", "sixty-eight", "sixty-nine", "seventy",
               "seventy-one", "seventy-two", "seventy-three", "seventy-four", "seventy-five", "seventy-six",
               "seventy-seven", "seventy-eight", "seventy-nine", "eighty", "eighty-one", "eighty-two", "eighty-three",
               "eighty-four", "eighty-five", "eighty-six", "eighty-seven", "eighty-eight", "eighty-nine", "ninety",
               "ninety-one", "ninety-two", "ninety-three", "ninety-four", "ninety-five", "ninety-six", "ninety-seven",
               "ninety-eight", "ninety-nine", ]
    nums = list(range(0, 100))
    n2 = [l.replace("-", "") for l in numbers]
    n3 = [l.replace("-", " ") for l in numbers]
    scales = ["hundred", "thousand", "lakh", "million", "crore", "billion", "trillion"]
    scale_nums = [1e2, 1e3, 1e5, 1e6, 1e7, 1e9, 1e12]
    nw = {}
    for i in range(len(scales)):
        for j in range(len(numbers)):
            number = numbers[j]
            scale = scales[i]
            num = nums[j]
            scs = scale_nums[i]
            value = int(num * scs)
            nw[number + " " + scale] = value
            nw[number + "-" + scale] = value
            nw[number + "" + scale] = value

            nw[n2[j] + " " + scale] = value
            nw[n2[j] + "-" + scale] = value
            nw[n2[j] + "" + scale] = value

            nw[n3[j] + " " + scale] = value
            nw[n3[j] + "-" + scale] = value
            nw[n3[j] + "" + scale] = value

    d1 = {tupl[0]: tupl[1] for tupl in zip(numbers, nums)}
    d2 = {tupl[0]: tupl[1] for tupl in zip(n2, nums)}
    d3 = {tupl[0]: tupl[1] for tupl in zip(n3, nums)}
    nw = {**nw, **d1, **d2, **d3}
    return nw


def remove_html_tags(text):
    """Remove html tags from a string"""
    if type(text) is pd.core.series.Series or type(text) is str:
        text = text.replace("'", " ").replace('"', " ")
        clean = re.compile('<.*?>')
        return re.sub(clean, ' ', text)
    return text


def clean_text(text):
    EMPTY = ''
    text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("\r", " ").replace("\t", " ").lower()
    text = re.sub('<pre><code>.*?</code></pre>', EMPTY, text)
    text = re.sub('<code>.*?</code>', EMPTY, text)

    def replace_link(match):
        EMPTY = ''
        return EMPTY if re.match('[a-z]+://', match.group(1)) else match.group(1)

    text = re.sub('<a[^>]+>(.*)</a>', replace_link, text)
    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('S'):
        return 's'
    else:
        return wordnet.NOUN


def is_stopword(word):
    return word in stopwords_list


def translate(text, kw, ignore_case=False):
    """Translate by changing keys mentioned in kw to values in passed text"""
    search_keys = map(lambda x: re.escape(x), kw.keys())
    if ignore_case:
        kw = {k.lower(): kw[k] for k in kw}
        regex = re.compile('|'.join(search_keys), re.IGNORECASE)
        res = regex.sub(lambda m: kw[m.group().lower()], text)
    else:
        regex = re.compile('|'.join(search_keys))
        res = regex.sub(lambda m: kw[m.group()], text)

    return res


def replace_numbers(text):
    if text is None or type(text) is not str:
        return text
    base_words = get_number_base_words()
    text = translate(text, {" "+k+" ": " _NUM_ " for k, v in base_words.items()})
    text = re.sub(r"[0-9]+.[0-9]+", "_NUM_", text)
    return re.sub(r"[0-9]+", "_NUM_", text)


def __get_translator_from_representation(representations, unit):
    l2 = [l.replace("-", "") for l in representations]
    l3 = [l.replace("-", " ") for l in representations]
    representations.extend(l2)
    representations.extend(l3)
    representations = set(representations)
    representations = list(
        more_itertools.flatten([[NUM + l + " ", NUM + " " + l + " ", NUM + "-" + l + " "] for l in representations]))
    translator = {k: NUM + " " + unit + " " for k in representations}
    return translator


def get_measurement_translators():
    length_representations = ["m", "cm", "mm", "km", "meter", "metre", "centi-meter", "milli-meter", "centi-metre",
                              "milli-metre","mili-meter","mili-metre","mili-meters","mili-metres"
                              "kilo-meter", "kilo-metre", "meters", "metres", "centi-meters", "milli-meters",
                              "centi-metres", "milli-metres","mtr",
                              "kilo-meters", "kilo-metres", "mile", "miles","inch","inches","in","cms","foot","feet","ft"]
    length_translator = __get_translator_from_representation(length_representations, UNIT_OF_LENGTH)
    volume_representations = ["ml", "l","lt","ltrs","ltr", "liter", "litre", "liters", "litres", "meter-cube", "m3", "cm3", "metre-cube",
                              "cubic-meter", "cubic-metre","lit","mili-liters","mili-liter","milli-litres","milli-liters",
                              "milli-litre", "milli-liter","barrel","barrels","cc"]
    volume_translator = __get_translator_from_representation(volume_representations, UNIT_OF_VOLUME)
    time_representations = ["s", "secs", "second", "seconds", "min", "minute", "minutes", "hour", "hours", "hr", "hrs",
                            "day", "days", "d", "week", "weeks", "month", "months", "year", "y", "years","yrs","yr","monhs"]
    time_translator = __get_translator_from_representation(time_representations, UNIT_OF_TIME)
    mass_representations = ["mg", "milli-gram", "milli-grams", "gram", "gm", "grams", "gms", "g", "kilo", "kilos", "kg",
                            "kgs","lb","ounce","ounces",
                            "kilo-gram", "kilo-grams", "ton", "tonnes", "quintal", "quintals","oz","pound","pounds"]
    mass_translator = __get_translator_from_representation(mass_representations, UNIT_OF_MASS)
    electricity_representations = ["watt", "w", "kw", "kilo-watt", "kilo-watts", "v", "volt", "volts", "ampere", "a",
                                   "amperes", "kwh", "wh",
                                   "w-h"]
    electricity_translator = __get_translator_from_representation(electricity_representations, UNIT_OF_ELECTRICITY)

    translators = {**length_translator, **volume_translator, **time_translator, **mass_translator,
                   **electricity_translator}
    return {"translators": translators, "electricity_translator": electricity_translator,
            "mass_translator": mass_translator,
            "time_translator": time_translator, "volume_translator": volume_translator,
            "length_translator": length_translator}


def replace_measurement(text):
    text = " "+text + " "
    # adding above line here since we check space before and after adding units
    # so if the number+unit occur at start or end of string they may get missed
    if text is None or type(text) is not str:
        return text
    text = replace_numbers(text)
    translators = get_measurement_translators()

    volume_translator = translators["volume_translator"]
    electricity_translator = translators["electricity_translator"]
    mass_translator = translators["mass_translator"]
    time_translator = translators["time_translator"]
    length_translator = translators["length_translator"]

    text = translate(text, volume_translator)
    text = translate(text, electricity_translator)
    text = translate(text, time_translator)
    text = translate(text, mass_translator)
    text = translate(text, length_translator)
    return text


def tokenize_lemmatize(text, external_text_processing_funcs=[replace_numbers], lemmatizer=None, token_postprocessor=[]):
    if text is None or type(text) is not str:
        return []
    if external_text_processing_funcs is not None:
        for fn in external_text_processing_funcs:
            text = fn(text)
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tokens = list(map(lambda x: lemmatizer.lemmatize(x[0], get_wordnet_pos(x[1])), pos_tags))
    if token_postprocessor is not None:
        for fn in token_postprocessor:
            tokens = [fn(token) for token in tokens]
    return tokens


def ngram_stopword(tokens, word_length_filter=3, ngram_limit=3):
    tokens = list(map(lambda x: re.sub('[^ a-zA-Z0-9]', '', x), tokens))
    tokens = list(map(lambda x: x.strip(), tokens))
    if ngram_limit is not None and ngram_limit >= 2:
        grams = list(more_itertools.flatten([ngrams(tokens, i) for i in range(2, ngram_limit + 1)]))
    else:
        grams = []
    all_words = []
    for w in tokens:
        if len(w) >= word_length_filter and not is_stopword(w):
            all_words.append(w)
    for w in grams:
        is_acceptable = not any([True for spw in w if len(spw) < word_length_filter or is_stopword(spw)])
        if is_acceptable:
            all_words.append(' '.join(w))
    return all_words


def combined_text_processing(text, external_text_processing_funcs=[replace_numbers], lemmatizer=None,
                             word_length_filter=3, ngram_limit=3,token_postprocessor=[]):
    tokens = tokenize_lemmatize(text, external_text_processing_funcs, lemmatizer,token_postprocessor)
    tokens = ngram_stopword(tokens, word_length_filter, ngram_limit)
    return tokens



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
            self.dictionary.filter_extremes(no_below=self.min_count)
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
        temp = self.dictionary[0]
        def tokens2tfidf(token_array, tfidf, dictionary):
            tmp = self.dictionary[0]
            id2tfidf = {k: v for k, v in tfidf[dictionary.doc2bow(token_array)]}
            token2tfidf = {dictionary.id2token[k]: v for k, v in id2tfidf.items()}
            return [token2tfidf[token] if token in token2tfidf else 0 for token in token_array]

        def tokens2vec(token_array, fasttext_model):
            if len(token_array) == 0:
                return np.full(self.size, 0)
            return [fasttext_model.wv[token] if token in fasttext_model.wv else np.full(self.size, 0) for token in token_array]

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

