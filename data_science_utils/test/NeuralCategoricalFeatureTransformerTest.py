import sys, os

sys.path.append(os.getcwd())

import os.path
import sys
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from preprocessing import NeuralCategoricalFeatureTransformer

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
import pandas as pd
import numpy as np

# df = pd.DataFrame({"a":["c1","c1","c1","c2","c2","c2","c1"],"b":["d1","d2","d3","d1","d2","d3","d1"],"x":[2,1,3,4,5,6,7],"y":[12,11,8,5,3,1,9]})
#
# nn_cat = NeuralCategoricalFeatureTransformer(cols=["a","b"],target_columns=["x","y"],verbose=1,n_components=16,n_iter=500)
# nn_cat.fit(df)
# tdf = nn_cat.transform(df)
# print(tdf)

# df = pd.DataFrame({"a":["c1","c1","c1","c2","c2","c2","c1","c1","c1"],
#                    "b":["d1","d2","d3","d1","d2","d3","d1","d2","d3"],
#                    "x":[1,0,1,1,1,0,1,0,0],
#                    "y": [1, 0, 1, 1, 1, 0, 1, 0, 0]})
#
# nn_cat = NeuralCategoricalFeatureTransformer(cols=["a","b"],target_columns=None,verbose=1,n_components=16,n_iter=500)
# nn_cat.fit(df)
# tdf = nn_cat.transform(df)
# print(tdf)


df = pd.DataFrame({"a":["c1","c1","c1","c2","c2","c2","c1","c1","c1"],
                   "b":["d1","d2","d3","d1","d2","d3","d1","d2","d3"],
                   "x":[1,0,1,1,1,0,1,0,0],
                   "y": [0, 1, 1, 1, 0, 0, 1, 0, 1]})

nn_cat = NeuralCategoricalFeatureTransformer(cols=["a","b"],target_columns=["x","y"],verbose=1,n_components=16,n_iter=500)
nn_cat.fit(df)
tdf = nn_cat.transform(df)
print(tdf)
