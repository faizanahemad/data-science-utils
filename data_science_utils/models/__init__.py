import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix



def feature_importance(model,df,features):
    fi = None
    if hasattr(model, 'feature_importances_'):
        fi=model.feature_importances_
    elif hasattr(model, 'coef_'):
        fi = model.coef_
    else:
        raise AttributeError('No attribute: feature_importances_ or  coef_')
    fn=df[features].columns.values
    df_i=pd.DataFrame({"feature":fn,"importance":fi})
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