from data_science_utils import dataframe
from data_science_utils import plots as plot_utils
import pandas as pd
from unittest import TestCase
import numpy as np
class TestDF(TestCase):
    def test_get_column_datatypes(self):
        s = dataframe.get_column_datatypes(pd.DataFrame({"A":[1, 2, 3], "B":[5, 6, 7]}))
        print(type(s))
        self.assertTrue(str(type(s))=="<class 'pandas.core.frame.DataFrame'>")
    def test_analyze_ts_results(self):
        test_true = np.random.randint(1,50,50)
        test_pred = np.random.randint(1, 50, 50)
        train_true = np.random.randint(1, 50, 100)
        train_pred = np.random.randint(1, 50, 100)
        plot_utils.analyze_ts_results(test_true, test_pred, train_true=train_true, train_pred=train_pred, timestamps=[], aep_line=20,
                           sample_percentile=80, plot=True, plot_error=False, xtick_interval=None, figsize=(24, 8))