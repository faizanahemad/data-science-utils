import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


np.set_printoptions(threshold=np.nan)

import warnings

warnings.filterwarnings('ignore')
import argparse

import sys, os

sys.path.append(os.getcwd())

import os
import requests
from bs4 import BeautifulSoup

curr_module_parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_module_parent_dir)

parser = argparse.ArgumentParser(description='Args for mutual fund script')
parser.add_argument('--mfid', required=True, help='Url to Mutual Fund/ETFs money control page')

args = parser.parse_args()

mfid = args.mfid
url = "https://www.moneycontrol.com/india/mutualfunds/mfinfo/portfolio_holdings/"+mfid

page_response = requests.get(url, timeout=5)

page_content = BeautifulSoup(page_response.content, "html.parser")

portfolio_table = page_content.find('table', attrs={'class': 'tblporhd'})

print(portfolio_table)

