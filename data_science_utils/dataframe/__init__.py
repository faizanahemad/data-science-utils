
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def _check_df(df):
    # Empty Dataframe check
    assert df.shape[0]>0 and df.shape[1]>0 , "DataFrame is Empty"
    # duplicate columns check
    assert len(df.columns.values)==len(set(df.columns.values)) , "DataFrame has duplicate columns"


def get_column_names(df,sorted=True):
    if(sorted):
        return list(np.sort(df.columns.values.tolist()))
    return df.columns.values.tolist()


def count_nulls(df):
    """Missing value count per column grouped by column name"""
    df_t = pd.DataFrame(df.isnull().sum()).rename(columns={0:"count"})
    df_t.index.names = ["Column"]
    return df_t.sort_values("count",ascending=False)

def count_distinct_values(df):
    _check_df(df)
    unique_counts = {}
    for idx in df.columns.values:
        #cnt=len(df[idx].unique())
        cnt = df[idx].nunique()
        unique_counts[idx]=cnt
    unique_ctr = pd.DataFrame([unique_counts]).T
    unique_ctr = unique_ctr.rename(columns={0: 'count'})
    unique_ctr.index.names = ["Column"]
    return unique_ctr.sort_values("count",ascending=False)


def __particular_values_per_column(df,values):
    _check_df(df)
    counts = {}
    for idx in df.columns.values:
        cnt=np.sum(df[idx].isin(values).values)
        counts[idx]=cnt
    ctr = pd.DataFrame([counts]).T
    ctr_2 = ctr.rename(columns={0: '# Values as %s'%values})
    return ctr_2

def get_column_datatypes(df):
    _check_df(df)
    dtype = {}
    for idx in df.columns.values:
        dt = df[idx].dtype
        dtype[idx]=dt
    ctr = pd.DataFrame([dtype]).T
    ctr = ctr.rename(columns={0: 'datatype'})
    ctr.index.names = ["Column"]
    return ctr


def most_common_value(df):
    # for simple cases use df.mode().T
    _check_df(df)
    total_rows = df.shape[0]
    columns,modes,counts = list(),list(),list()
    for column in df.columns:
        count = df[column].value_counts().max()
        mode = df[column].value_counts().index[0]
        columns.append(column)
        modes.append(mode)
        counts.append(count)
    description = pd.DataFrame(index=columns,data={"Most Frequent Value":modes,"Most Frequent Value Count":counts})
    description["Most Frequent Value %"] = 100 * description["Most Frequent Value Count"]/total_rows
    description.index.names = ["Column"]
    return description

def column_summaries(df):
    _check_df(df)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val/len(df)
    particular_ctr = __particular_values_per_column(df,[0])
    unique_ctr = count_distinct_values(df)
    statistical_summary = df.describe().T
    datatypes = get_column_datatypes(df)
    most_common_values = most_common_value(df)
    skewed = pd.DataFrame(df.skew()).rename(columns={0: 'skew'})
    mis_val_table = pd.concat([mis_val, mis_val_percent, unique_ctr, particular_ctr,datatypes,skewed,statistical_summary,most_common_values], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% missing of Total Values'})
    return mis_val_table_ren_columns

def filter_dataframe_values(df,filter_columns):
    _check_df(df)
    df_filtered = df
    for feature in filter_columns:
        values = filter_columns[feature]
        if(len(values)==1):
            # if only one value present assume upper range given
            df_filtered = df_filtered[df_filtered[feature]<=values[0]]
        elif(len(values)==2):
            df_filtered = df_filtered[(df_filtered[feature]>=values[0]) & (df_filtered[feature]<=values[1])]
    return df_filtered

def filter_dataframe_percentile(df, filter_columns):
    _check_df(df)
    df_filtered = df
    for feature in filter_columns:
        quantiles = filter_columns[feature]
        values = df[feature].quantile(quantiles).values
        if(len(values)==1):
            # if only one value present assume upper quantile
            df_filtered = df_filtered[df_filtered[feature]<=values[0]]
        elif(len(values)==2):
            df_filtered = df_filtered[(df_filtered[feature]>=values[0]) & (df_filtered[feature]<=values[1])]
    return df_filtered


def get_specific_cols(df,prefix=None,suffix=None,substring=None):
    features = list(df.columns.values)
    sl = list()
    if(substring is not None):
        for feature in np.sort(features):
            if(substring in feature):
                sl.append(feature)
    if(prefix is not None):
        for feature in np.sort(features):
            if(feature.startswith(prefix)):
                sl.append(feature)
    if(suffix is not None):
        for feature in np.sort(features):
            if(feature.endswith(suffix)):
                sl.append(feature)
    return list(np.sort(sl))


def drop_specific_cols(df,prefix=None,suffix=None,substring=None,inplace=False):
    _check_df(df)
    features = list(df.columns.values)
    sl = list()
    if(substring is not None):
        for feature in np.sort(features):
            if(substring in feature):
                sl.append(feature)
    if(prefix is not None):
        for feature in np.sort(features):
            if(feature.startswith(prefix)):
                sl.append(feature)
    if(suffix is not None):
        for feature in np.sort(features):
            if(feature.endswith(suffix)):
                sl.append(feature)
    return df.drop(sl,axis=1,inplace=inplace)




def drop_columns_safely(df,columns,inplace=False):
    assert len(columns)>0,"Column list passed for dropping is empty"
    _check_df(df)
    cur_cols=set(df.columns)
    drop_columns = list(set(columns).intersection(cur_cols))
    return df.drop(drop_columns,axis=1,inplace=inplace)





def find_correlated_pairs(df,thres):
    _check_df(df)
    df_corr=df.corr()
    correlated_pairs = list()
    processed_pair = set()
    np.fill_diagonal(df_corr.values, 0)
    for col in df_corr.columns:
        df1=df_corr[col]
        for index, value in df1.iteritems():
            col_pair1 = col+"#"+index
            col_pair2 = index+"#"+col
            try:
                if(abs(value)>thres 
                   and col_pair1 not in processed_pair 
                   and col_pair2 not in processed_pair):
                    correlated_pairs.append((col,index))
                    processed_pair.add(col_pair1)
                    processed_pair.add(col_pair2)
                    
            except:
                print()
    return correlated_pairs

def remove_correlated_pairs(df,thres,inplace=False):
    _check_df(df)
    df_nulls = count_nulls(df).transpose()
    correlated_pairs = find_correlated_pairs(df,thres)
    dropped_cols = set()
    for (p1,p2) in correlated_pairs:
        if(p1 not in dropped_cols and p2 not in dropped_cols):
            p1_nulls = df_nulls[p1].values[0]
            p2_nulls = df_nulls[p2].values[0]
            if(p1_nulls < p2_nulls):
                dropped_cols.add(p2)
            else:
                dropped_cols.add(p1)
    dropped_cols = list(np.sort(list(dropped_cols)))
    new_df = drop_columns_safely(df,dropped_cols,inplace)
    return (new_df,dropped_cols)

def detect_nan_columns(df):
    _check_df(df)
    columns = df.columns.values.tolist()
    cols = list()
    for colname in columns:
        if(np.sum(pd.isnull(df[colname]))>0):
            cols.append(colname)
    return list(np.sort(cols))

def fast_read_and_append(file_path,chunksize,fullsize=1e9,dtype=None):
    # in chunk reading be careful as pandas might infer a columns dtype as different for diff chunk
    # As such specifying a dtype while reading by giving params to read_csv maybe better
    # Label encoding will fail if half the rows for same column is int and rest are str
    # In case of that already happened then df_test["publisherId"] = df_test["publisherId"].apply(str)
    df = pd.DataFrame()
    iterations = 0
    total_needed_iters = math.ceil(fullsize/chunksize)
    for x in pd.read_csv(file_path, chunksize=chunksize,low_memory=False,dtype=dtype):
        print("iterations= %s out of %s" %  (iterations,total_needed_iters))
        df = pd.concat([df, x], ignore_index=True)
        iterations += 1
    return df

def add_polynomial_and_log_features(df,f_name):
    _check_df(df)
    cnames=[f_name]
    log_col = "log_%s"%f_name
    df[log_col] = np.log1p(df[f_name]+1)
    square_root_col = "square_root_%s"%f_name
    df[square_root_col] = df[f_name]**0.5
    square_col = "square_%s"%f_name
    df[square_col] = df[f_name]**2
    cube_col = "cube_%s"%f_name
    df[cube_col] = df[f_name]**3
    cnames += [log_col,square_col,cube_col,square_root_col]
    return cnames

        

    