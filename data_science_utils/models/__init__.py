import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error



def feature_importance(model,df,features):
    fi=model.feature_importances_
    fn=df[features].columns.values
    df_i=pd.DataFrame({"feature":fn,"importance":fi})
    df_i["importance"] = df_i["importance"]*100
    return df_i.sort_values("importance",ascending=False)


def rmsle(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def rmse(y_true,y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.square(y_pred - y_true).mean() ** 0.5

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


def baseline_logistic_random(df,predicted_column,df_test=None,id_col=None,verbose=True):
    random_preds_train = np.random.randint(0, high=2, size=df.shape[0])
    if(df_test is not None and id_col is not None):
        random_preds_test = np.random.randint(0, high=2, size=df_test.shape[0])
        df_test[predicted_column] = random_preds_test
        df_test = df_test[[id_col,predicted_column]]
    if(verbose):
        print(classification_report(df[predicted_column],random_preds_train))
        
    return (classification_report(df[predicted_column],random_preds_train),df_test)