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
    text = translate(text, {k: "_NUM_" for k, v in base_words.items()})
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
                              "milli-metre",
                              "kilo-meter", "kilo-metre", "meters", "metres", "centi-meters", "milli-meters",
                              "centi-metres", "milli-metres",
                              "kilo-meters", "kilo-metres", "mile", "miles"]
    length_translator = __get_translator_from_representation(length_representations, UNIT_OF_LENGTH)
    volume_representations = ["ml", "l", "liter", "litre", "liters", "litres", "meter-cube", "m3", "cm3", "metre-cube",
                              "cubic-meter", "cubic-metre"]
    volume_translator = __get_translator_from_representation(volume_representations, UNIT_OF_VOLUME)
    time_representations = ["s", "secs", "second", "seconds", "min", "minute", "minutes", "hour", "hours", "hr", "hrs",
                            "day", "days", "d", "week", "weeks", "month", "months", "year", "y", "years"]
    time_translator = __get_translator_from_representation(time_representations, UNIT_OF_TIME)
    mass_representations = ["mg", "milli-gram", "milli-grams", "gram", "gm", "grams", "gms", "g", "kilo", "kilos", "kg",
                            "kgs",
                            "kilo-gram", "kilo-grams", "ton", "tonnes", "quintal", "quintals"]
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
    text = text + " "
    # adding above line here since we check space before adding units
    # so if the number+unit occur at end of string they may get missed
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


def tokenize_lemmatize(text, external_text_processing_funcs=[replace_numbers], lemmatizer=None):
    if external_text_processing_funcs is not None:
        for fn in external_text_processing_funcs:
            text = fn(text)
    if text is None:
        return []
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    tokens = list(map(lambda x: lemmatizer.lemmatize(x[0], get_wordnet_pos(x[1])), pos_tags))
    return tokens


def ngram_stopword(tokens, word_length_filter=3, ngram_limit=3):
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
                             word_length_filter=3, ngram_limit=3):
    tokens = tokenize_lemmatize(text, external_text_processing_funcs, lemmatizer)
    tokens = ngram_stopword(tokens, word_length_filter, ngram_limit)
    return tokens



