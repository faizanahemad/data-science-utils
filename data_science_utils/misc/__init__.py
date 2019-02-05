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


def print_code(func):
    print("".join(inspect.getsourcelines(func)[0]))


def get_timer(printer=print):
    ctr = 1
    start = time.time() % 1000000
    pv = start
    if printer is None:
        printer = lambda x: x

    def timer(text=""):
        nonlocal ctr
        nonlocal pv
        tc = time.time() % 1000000
        time_elapsed = tc - start
        diff = tc - pv
        diff = 0 if diff > 1e7 else diff
        pv = tc
        event_text = "{:<8}".format("Event %s" % ctr)
        text = " {:<10}".format(text)
        printer("%s|%s |%s" % (event_text, "{:<9}".format("%.2f" % diff), text))
        ctr = ctr + 1

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

def deep_map(fn,elems):
    if isinstance(elems,list) or isinstance(elems,np.ndarray):
        return list(map(lambda t:deep_map(fn,t), elems))
    else:
        return fn(elems)


def get_week_start_date(df,date_col,format=None):
    import datetime as dt
    date_col = pd.to_datetime(df[date_col],format=format)
    daysoffset = date_col.dt.weekday.apply(lambda x:dt.timedelta(days=x))
    week_start = date_col - daysoffset
    week_start = pd.to_datetime(week_start.dt.strftime('%Y-%m-%d'))
    return week_start


def save_list_per_line(lines, filename):
    # convert lines to a single blob of text
    lines = list(map(str,lines))
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)


def load_list_per_line(filename):
    with open(filename, 'r') as file:
        text = file.readlines()
        return text


