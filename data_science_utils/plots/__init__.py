from data_science_utils import dataframe as utils

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def scatter_plot_exclude_outliers(f_name,predicted_column,df,title=None,percentile=[0.01,0.99],logy=False,logx=False):
    df = utils.filter_dataframe_percentile(df,{f_name:percentile,predicted_column:percentile})
    df.plot.scatter(x=f_name, y=predicted_column,title=title,logy=logy,logx=logx)
    plt.figure();
    plt.show();


def plot_numeric_feature(f_name,predicted_column,df):
    figsize = mplt.rcParams['figure.figsize']
    mplt.rcParams['figure.figsize'] = (12,8)
    if(len(df)==0):
        print("Empty Dataframe - No plots possible")
        return


    df["log_%s" %f_name] = np.log(df[f_name]+1)
    df["square_root_%s" %f_name] = df[f_name]**0.5
    df["square_%s" %f_name] = df[f_name]**2
    df["cube_%s" %f_name] = df[f_name]**3

    scatter_plot_exclude_outliers(f_name, predicted_column,df,title="%s vs %s" %(predicted_column,f_name),logy=False)
    scatter_plot_exclude_outliers(f_name, predicted_column,df,title="Log %s vs %s" %(predicted_column,f_name),logy=True)
    scatter_plot_exclude_outliers("log_%s" %f_name, predicted_column,df,title="%s vs Log %s"%(predicted_column,f_name))
    scatter_plot_exclude_outliers("square_root_%s" %f_name, predicted_column,df,title="%s vs square_root %s"%(predicted_column,f_name))
    scatter_plot_exclude_outliers("square_%s" %f_name, predicted_column,df,title="%s vs square %s"%(predicted_column,f_name))
    scatter_plot_exclude_outliers("cube_%s" %f_name, predicted_column,df,title="%s vs cube %s"%(predicted_column,f_name))
    mplt.rcParams['figure.figsize'] = figsize
    plt.figure();
    plt.show();

def plot_numeric_features_filtered(f_name,predicted_column,df,filter_columns,strategy=None):
    colnames = [f_name]
    if strategy is None:
        colnames = [f_name]
    elif strategy=='prefix':
        colnames = df.columns[pd.Series(df.columns).str.startswith(f_name)]
    elif strategy=='suffix':
        colnames = df.columns[pd.Series(df.columns).str.endswith(f_name)]
    print("------Histograms for Distribution------")
    mplt.rcParams['figure.figsize'] = (6,4)
    for colname in colnames:
        df.hist(column=colname, bins=20)
    plt.show();
    df = utils.filter_dataframe(df,filter_columns)
    print("------Feature vs Predicted Column------")
    for colname in colnames:
        plot_numeric_feature(colname,predicted_column,df)


def plot_ts(df_test, columns=[], time_col='week', freq='7D',figsize=(24,8)):
    df_preds = df_test.copy()
    df_preds = df_preds.sort_values([time_col])
    idx = pd.date_range(df_preds[time_col].min(), df_preds[time_col].max(), freq=freq)
    df_preds.set_index(time_col, inplace=True)
    # scaler = RobustScaler()
    scaler = MinMaxScaler()
    df_preds[columns] = scaler.fit_transform(df_preds[columns])
    fg = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = figsize

    handles = []
    for column in columns:
        tsSparseActual = df_preds[column]
        tsActual = tsSparseActual.reindex(idx, fill_value=0)
        p1, = plt.plot(tsActual.index, tsActual, linestyle='--', marker='o', label=column)
        handles.append(p1)
    plt.legend(handles=handles)
    plt.show()
    plt.rcParams["figure.figsize"] = fg


def plot_ts_single_column(df_test, time_col, ewma_diff_plot=False, ewma_range=2, freq='7D', target='target',
                          figsize=(24, 8)):
    df_preds = df_test.copy()
    idx = pd.date_range(df_preds[time_col].min(), df_preds[time_col].max(), freq=freq)
    df_preds = df_preds.sort_values([time_col])
    df_preds.set_index(time_col, inplace=True)
    fg = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = figsize
    tsSparseActual = df_preds[target]
    tsActual = tsSparseActual.reindex(idx, fill_value=1)
    df_preds['ewma'] = pd.ewma(tsActual.shift(1).fillna(tsActual.min()), span=ewma_range)
    tsSparse = df_preds['ewma']
    ts = tsSparse.reindex(idx, fill_value=1)
    tsDiff = (tsActual - ts) / ts
    p1, = plt.plot(tsActual.index, tsActual, linestyle='--', marker='o', label=target)

    handles = [p1]
    if ewma_diff_plot:
        p2, = plt.plot(ts.index, ts, linestyle='-.', marker='o', label="EWMA")
        handles = [p1, p2]
        plt.legend(handles=handles)
        plt.show()
        p3, = plt.plot(tsDiff.index, tsDiff, linestyle='-', marker='o', label="Diff")
        plt.show()
    else:
        plt.legend(handles=handles)
        plt.show()
    plt.rcParams["figure.figsize"] = fg
    return pd.DataFrame({"target": tsActual, "EWMA": ts, "Diff": tsDiff, time_col: df_preds.index})