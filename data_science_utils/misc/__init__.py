import inspect
import time

import numpy as np
import pandas as pd

from fastnumbers import isfloat
from fastnumbers import fast_float

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
    return fast_float(string.split(" ")[0].replace(',','').replace('%',''),default=np.nan)

def ffloat_list(string_list):
    return list(map(ffloat,string_list))

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string


