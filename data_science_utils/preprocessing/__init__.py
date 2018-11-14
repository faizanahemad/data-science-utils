from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
def reduce_dimensions_by_ohe_svd(frame,n_components=2,n_iter=10):
    pca = TruncatedSVD(n_components=n_components,n_iter=n_iter)
    enc = OneHotEncoder(handle_unknown='ignore')
    ft = FunctionTransformer()
    pipeline = make_pipeline(enc,pca,ft)
    pipeline.fit(frame)
    return pipeline