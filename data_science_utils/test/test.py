from data_science_utils import df
import pandas as pd
from unittest import TestCase
class TestDF(TestCase):
    def test_get_column_datatypes(self):
        s = df.get_column_datatypes(pd.DataFrame({"A":[1,2,3],"B":[5,6,7]}))
        print(type(s))
        self.assertTrue(str(type(s))=="<class 'pandas.core.frame.DataFrame'>")