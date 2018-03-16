from data_science_utils import df as utils

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt


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