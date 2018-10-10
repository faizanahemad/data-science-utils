import inspect
import time

import numpy as np
import pandas as pd

from fastnumbers import isfloat
from fastnumbers import fast_float
import re
import os
from multiprocessing import Pool
import functools


def print_function_code(func):
    print("".join(inspect.getsourcelines(func)[0]))


def get_timer(printer=print):
    ctr=1
    pv = -1e8
    if printer is None:
        printer = lambda x: x
    def timer(text=""):
        nonlocal ctr
        nonlocal pv
        tc = time.time()%10000
        diff = tc - pv
        diff = 0 if diff>1e6 else diff
        pv=tc
        printer("%s: %.3f, %.3f, %s "%(ctr,tc,diff,text))
        ctr=ctr+1
    return timer

def is_dataframe(df):
    if df is not None and type(df)==pd.core.frame.DataFrame:
        return True
    return False

def ffloat(string):
    if string is None:
        return np.nan
    if type(string)==float or type(string)==int or type(string)==np.int64 or type(string)==np.float64:
        return string
    string = re.sub('[^0-9\.]','',string.split(" ")[0])
    return fast_float(string,default=np.nan)

def ffloat_list(string_list):
    return list(map(ffloat,string_list))

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string


def parallel_map_reduce(initial_values,map_fn,reduce_fn=None,reduce_initializer=None,cores=os.cpu_count()-1):
    with Pool(processes=cores) as pool:
        result = pool.map(map_fn, initial_values)
    if reduce_fn is None:
        return result
    else:
        if reduce_initializer is not None:
            return functools.reduce(reduce_fn, result,reduce_initializer)
        else:
            return functools.reduce(reduce_fn, result)


