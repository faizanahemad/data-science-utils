import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import gc

from data_science_utils.dataframe import get_specific_cols


def feature_importance(model,features):
    """

    :param model: Model object which has `feature_importances_`
    :param features: features/columns that were given to model, these must be in same order as given to model
    :return: DataFrame with sorted feature importances
    """
    if hasattr(model, 'feature_importances_'):
        fi=model.feature_importances_
    elif hasattr(model, 'coef_'):
        fi = model.coef_
    else:
        raise AttributeError('No attribute: feature_importances_ or  coef_')
    df_i=pd.DataFrame({"feature":features,"importance":fi})
    df_i["importance"] = df_i["importance"]*100
    return df_i.sort_values("importance",ascending=False)


def rmsle(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return mean_squared_error(np.log(y_true + 1),np.log(y_pred + 1)) ** 0.5

def rmse(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return mean_squared_error(y_true,y_pred) ** 0.5


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-4):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true),
                                              epsilon,
                                            None))
    return 100. * np.mean(diff)


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def baseline_regression_by_category(df,df_cv,categorical_column,predicted_column,df_test=None,id_col=None,nan=-1):
    df = df.fillna(nan)
    
    df_cv = df_cv.fillna(nan)
    df_mean = df.groupby([categorical_column])[predicted_column].agg(['mean','max','min','median'])
    df_mean[df_mean.index.names[0]] = df_mean.index
    df_merged = df_cv.merge(df_mean, on=categorical_column, how='inner')
    df_merged_test = pd.DataFrame({})
    if(df_test is not None and id_col is not None):
        df_test = df_test.fillna(nan)
        df_merged_test = df_test.merge(df_mean, on=categorical_column, how='inner')
        df_merged_test = df_merged_test[[id_col,'mean','max','min','median']]
    
    baseline_cv = {}
    for measure in ["mean","max","min","median"]:
        baseline_cv["rmsle_with_"+measure] = rmsle(df_cv[predicted_column],df_merged[measure])
        baseline_cv["rmse_with_"+measure] = rmse(df_cv[predicted_column],df_merged[measure])
    
    return (baseline_cv,df_merged_test)


def baseline_logistic_random(df,predicted_column,verbose=True):
    random_preds_train = np.random.randint(0, high=2, size=df.shape[0])
    if(verbose):
        print(classification_report(df[predicted_column],random_preds_train))
        
    return classification_report(df[predicted_column],random_preds_train)

def confusion_matrix_frame(y_true,y_pred,labels=None,sample_weight=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels=np.unique(np.concatenate((y_true,y_pred)))
    matrix = confusion_matrix(y_true,y_pred,labels,sample_weight)
    matrix = pd.DataFrame(matrix,index=labels,columns=labels)
    
    matrix["Actual Counts"]=matrix.sum(axis=1)
    predicted_counts = pd.DataFrame(matrix.sum(axis=0)).T
    matrix = pd.concat([matrix, predicted_counts], ignore_index=True)
    
    new_index = list(labels)
    new_index.append("Predicted Counts")
    matrix.index = new_index
    matrix.index.names = ["Actual"]
    matrix.columns.names = ["Predicted"]
    
    
    actual_counts=matrix["Actual Counts"].values[:-1]
    predicted_counts = matrix[matrix.index=="Predicted Counts"].values[0][:-1]
    good_predictions = list()
    for label in labels:
        good_predictions.append(matrix[label].values[label])

    recall = 100*np.array(good_predictions)/actual_counts
    precision = 100*np.array(good_predictions)/predicted_counts
    recall = np.append(recall,[np.nan])
    matrix["Recall %"] = recall
    precision = pd.DataFrame(precision).T
    matrix = pd.concat([matrix, precision], ignore_index=True)
    new_index.append("Precision %")
    matrix.index = new_index
    matrix.index.names = ["Actual"]
    matrix.columns.names = ["Predicted"]
    matrix.fillna(-997,inplace=True)
    matrix = matrix.astype(int)
    matrix.replace(-997,np.nan,inplace=True)
    return matrix


def autoencoder_provide_reasons(actual,scaler,thres,autoencoder,features,top_error_cols=3):
    """

    returns errors,masked_actual,unmasked_actual_cols,description_df
    """
    scaled = actual.copy()
    
    scaled[features] = scaler.transform(actual[features].fillna(0))
    
    predictions = autoencoder.predict(scaled[features])
    mse = np.mean(np.power(scaled[features] - predictions, 2), axis=1)
    scaled['autoencoder_mse'] = mse
    scaled_copy = scaled[scaled['autoencoder_mse']>thres]
    actual = actual[scaled['autoencoder_mse']>thres]
    scaled = scaled_copy
    
    
    unmasked_actual_cols = actual.copy()
    errors = scaled.copy()
    actual = actual.copy()
    if(errors.shape[0]<1):
        print("No samples with error above given threshold, try decreasing threshold.")
    
    predictions = np.asarray(autoencoder.predict(scaled[features]))
    predictions = np.asarray(np.power(scaled[features] - predictions, 2))
    mse = np.mean(predictions, axis=1)
    se = np.sum(predictions, axis=1)
    mask = predictions<np.sort(predictions)[:,-top_error_cols:-top_error_cols+1].flatten().reshape((predictions.shape[0],1))
    predictions[mask]=0
    
    frac_se = np.sum(predictions, axis=1)
    frac_contribution_percent = (frac_se/se)*100
    errors[features] = (predictions/se.reshape((predictions.shape[0],1)))*100
    acv = np.asarray(actual[features].values)
    acv[mask] = 0
    actual[features] = acv
    
    biggest_contributor = errors[features].idxmax(axis=1)
    vals = np.asarray(errors[features].values)
    largest_contrib_mask = vals<np.sort(vals)[:,-1].flatten().reshape((vals.shape[0],1))
    vals[largest_contrib_mask]=0
    # We already calculated % in errors df
    largest_contrib_error_percent = np.sum(vals, axis=1)
    
    actual_vals = np.asarray(actual[features].values)
    actual_vals[largest_contrib_mask]=0
    largest_contrib_actual_value = np.sum(actual_vals, axis=1)
    
    description_df = pd.DataFrame({"fraction_contribution_percent":frac_contribution_percent,
                                  "mean_error":mse,"total_error":se,"fractional_contribution":frac_se,
                                  "largest_contributor":biggest_contributor,
                                  "largest_contrib_error_percent":largest_contrib_error_percent,
                                  "largest_contrib_actual_value":largest_contrib_actual_value})

    return (errors,actual,unmasked_actual_cols,description_df)


def generate_results(model,df_test,features,id_col,target,file):
    dft = df_test[features]
    results = df_test[[id_col]]
    results[target] = model.predict_proba(dft)[:,1]
    results.to_csv(file,index=False,columns=results.columns)


def cross_validate_classifier(model,X,y,scoring=['roc_auc','f1','f1_weighted','recall','precision','neg_log_loss'],cv=4,return_train_score=True):
    scores = cross_validate(model, X, y, scoring=scoring,cv=cv, return_train_score=return_train_score)
    for score in scoring:
        if return_train_score:
            scores['train_'+score+'_mean'] = scores['train_'+score].mean()
            scores['train_'+score+'_std'] = scores['train_'+score].std()
        scores['test_'+score+'_mean'] = scores['test_'+score].mean()
        scores['test_'+score+'_std'] = scores['test_'+score].std()
    return scores

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import time
def cross_validate_classifier_find_misclassified(build_model,X,y,scoring_fn,cv=4):
    X, y = shuffle(X, y)
    kf = KFold(n_splits=cv)
    results = {}
    i = 0
    for train_index, test_index in kf.split(X):
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        model = build_model()
        start = time.time()
        print("Starting Processing for GRoup %s of %s."%((i+1),cv))
        model.fit(X_train,y_train)
        y_score = model.predict_proba(X_test)
        res = scoring_fn(y_test, y_score, data=X_test)
        results[i] = res
        end = time.time()
        print("Group %s of %s done. Time taken = %.1f"%((i+1),cv,end-start))

        i = i+1
        gc.collect()
    gc.collect()
    return results




class ClassifierColumnCombiner():
    def __init__(self, columns, voting='hard', voting_strategy="or", weights=None, classification_threshold=0.5):
        self.voting = voting
        self.weights = weights
        if self.weights is None:
            self.weights = np.full(len(columns), 1)
        self.classification_threshold = classification_threshold
        self.voting_strategy = voting_strategy
        self.columns = columns

    def set_weights(self, weights):
        assert len(weights) == len(self.columns)
        self.weights = weights

    def fit(self, X, y):

        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

    def predict(self, X, voting=None, voting_strategy=None):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        if voting is None:
            voting = self.voting

        if voting_strategy is None:
            voting_strategy = self.voting_strategy
        if voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:
            predictions = self._predict(X)
            # 'hard' voting
            if voting_strategy == "or":
                maj = np.apply_along_axis(lambda x: 1 if any(x) else 0, axis=2, arr=predictions).reshape((-1, 1))
            elif voting_strategy == "majority":
                maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                          axis=1, arr=predictions)
            elif voting_strategy == "and":
                maj = np.apply_along_axis(lambda x: 1 if all(x) else 0, axis=2, arr=predictions).reshape((-1, 1))
            else:
                raise ValueError()
        return maj

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """

        weights = self.weights

        assert len(weights) == len(self.columns)
        avg = np.average(X[self.columns], axis=1, weights=weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([X[clf].values.reshape((-1, 1)) for clf in self.columns]).T


class BinaryClassifierToTransformer:
    def __init__(self, classifier, output_col, columns=[], prefixes=None, suffixes=None,
                 store_train_data=False,
                 store_transform_data=False,
                 scale_input=False, impute=False, raise_null=False,training_sampling_fn=None):
        self.classifier = classifier
        self.columns = columns
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.output_col = output_col
        assert len(columns) > 0 or prefixes is not None
        self.scale_input = scale_input
        self.scaler = StandardScaler()
        self.impute = impute
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.imp_inf = SimpleImputer(missing_values=np.inf, strategy='mean')
        self.raise_null = raise_null
        self.cols = None
        self.training_sampling_fn = training_sampling_fn
        self.train = None
        self.store_train_data = store_train_data
        self.store_transform_data = store_transform_data
        self.transform_data = None

    def check_null_(self, X):
        nans = np.isnan(X)
        infs = np.isinf(X)
        nan_summary = np.sum(np.logical_or(nans, infs))
        if nan_summary > 0:
            raise ValueError("nans/inf in frame = %s" % (nan_summary))

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

    def fit(self, X, y, sample_weight=None):
        if self.store_train_data:
            self.train = (X.copy(),y.copy(),sample_weight)
        cols = self.get_cols_(X)
        self.cols = cols
        if self.training_sampling_fn is not None:
            X,y,sample_weight = self.training_sampling_fn(X,y,sample_weight)

        X = X[cols]
        if self.impute:
            X = self.imp.fit_transform(X)
            X = self.imp_inf.fit_transform(X)
        if self.scale_input:
            X = self.scaler.fit_transform(X)
        if self.raise_null:
            self.check_null_(X)

        self.classifier.fit(X, y, sample_weight=sample_weight)
        gc.collect()
        return self

    def fit_stored(self):
        X,y,sample_weight = self.train
        return self.fit(X,y,sample_weight)

    def partial_fit(self, X, y):
        return self.fit(X, y)

    def transform(self, X, y='ignored'):
        if self.store_transform_data:
            self.transform_data = (X.copy())
        Inp = X
        cols = self.cols
        Inp = Inp[cols]
        if self.impute:
            Inp = self.imp.transform(Inp)
            Inp = self.imp_inf.transform(Inp)
        if self.scale_input:
            Inp = self.scaler.transform(Inp)

        if self.raise_null:
            self.check_null_(Inp)
        probas = self.classifier.predict_proba(Inp)[:, 1]
        X[self.output_col] = probas
        gc.collect()
        return X

    def inverse_transform(self, X, copy=None):
        raise NotImplementedError()

    def fit_transform(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X, y)
