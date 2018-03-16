import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder


def count_distinct_values(df, colname):
    df_t = pd.DataFrame(df[colname].value_counts(dropna=False))
    df_t.index.names = [colname]
    df_t.columns = ['Count']
    df_t.sort_index(inplace=True)
    return df_t


def label_encode_text_column(df,field,df_test=None,fillna="-1"):
    encoded_df = df[field].fillna(fillna)
    if(df_test is not None):
    	encoded_df = encoded_df.append(df_test[field].fillna(fillna))
    label_encoder = LabelEncoder()
    print(encoded_df.values)
    encoder = label_encoder.fit(encoded_df.values)
    
    if(df_test is not None):
    	encoded_t=encoder.transform(df_test[field].fillna(fillna).values)
    	df_test[field+"_encoded"] = encoded_t
    encoded_df=encoder.transform(df[field].fillna(fillna).values)
    df[field+"_encoded"] = encoded_df
    return encoder

def store_encoder_as_file(le,column_name,location):
    ids =np.arange(0,len(list(le.classes_))).astype(int)
    my_encoding = pd.DataFrame(list(le.classes_), ids, columns = [column_name])
    print(my_encoding.shape)
    my_encoding.to_csv("%s/%s-encoding.csv" % (location,column_name), index_label = ["id"])

def lagged_variance(series,lag):
    arr=series.values
    lags = arr[:lag]
    arr2 = np.delete(arr, [i for i in range(lag)], axis=0)
    arr2 = np.append(arr2,lags, axis=0)
    cov=np.corrcoef(arr,arr2)[0,1]
    logcov=np.corrcoef(np.log1p(arr),np.log1p(arr2))[0,1]
    return (cov,logcov)