from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
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