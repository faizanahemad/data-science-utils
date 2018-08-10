import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from stockstats import StockDataFrame

import warnings
import traceback

warnings.filterwarnings('ignore')
import argparse
import re
import sys, os

sys.path.append(os.getcwd())

import os
import requests
from requests.exceptions import ConnectionError

import bs4
from bs4 import BeautifulSoup
from fastnumbers import isfloat
from fastnumbers import fast_float
from multiprocessing.dummy import Pool as ThreadPool
import more_itertools
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import seaborn as sns
sns.set_style('whitegrid')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.dates as mdates
import seaborn as sns
import math
import gc
import ipaddress
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils.dataframe import column as column_utils
from data_science_utils.models.IdentityScaler import IdentityScaler as IdentityScaler


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report

import lightgbm as lgb

np.set_printoptions(threshold=np.nan)
import pickle


from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import missingno as msno
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import datetime
from scipy import signal
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn import linear_model
from sklearn.metrics import roc_auc_score


from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')


from data_science_utils.misc import ffloat
from data_science_utils.misc import is_dataframe
from data_science_utils.misc import ffloat_list
from data_science_utils.misc import remove_multiple_spaces


from datetime import date, timedelta

def prev_weekday(adate):
    if adate.weekday() <=4:
        return adate
    adate -= timedelta(days=1)
    while adate.weekday() > 4: # Mon-Fri are 0-4
        adate -= timedelta(days=1)
    return adate

def get_ci(p,t,r):
    return np.abs(np.fv(r/100,t,0,p))

def get_cumulative_amounts(p,t,r):
    psum = p
    for i in range(1,t):
        psum = psum + get_ci(p,i,r)
    return psum

def get_year_when_cumulative_profit_over_pe(pe,cpg):
    if np.isnan(pe) or np.isnan(cpg):
        return np.inf
    for i in range(1,int(np.ceil(pe))):
        if get_cumulative_amounts(1,i,cpg)>=pe:
            return i
    return int(np.ceil(pe))


def get_children(html_content):
    return [item for item in html_content.children if type(item) == bs4.element.Tag]


def get_portfolio(mfid):
    url = "https://www.moneycontrol.com/india/mutualfunds/mfinfo/portfolio_holdings/" + mfid
    page_response = requests.get(url, timeout=240)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    portfolio_table = page_content.find('table', attrs={'class': 'tblporhd'})
    fund_name = page_content.find('h1').text
    return portfolio_table, fund_name


def get_table(portfolio_table):
    portfolio_elems = get_children(portfolio_table)

    table_data = list()
    for row in portfolio_elems:
        row_data = list()
        row_elems = get_children(row)
        for elem in row_elems:
            text = elem.text.strip().replace("\n", "")
            if len(text) == 0:
                continue
            elem_descriptor = {'text': text}
            elem_children = get_children(elem)
            if len(elem_children) > 0:
                if elem_children[0].has_attr('href'):
                    elem_href = elem_children[0]['href']
                    elem_descriptor['href'] = elem_href

            row_data.append(elem_descriptor)
        table_data.append(row_data)
    return table_data


def get_table_simple(portfolio_table, is_table_tag=True):
    portfolio_elems = portfolio_table.find_all('tr') if is_table_tag else get_children(portfolio_table)
    table_data = list()
    for row in portfolio_elems:

        row_data = list()
        row_elems = get_children(row)
        for elem in row_elems:
            text = elem.text.strip().replace("\n", "")
            text = remove_multiple_spaces(text)
            if len(text) == 0:
                continue
            row_data.append(text)
        table_data.append(row_data)
    return table_data


def get_inner_texts_as_array(elem, filter_empty=True):
    children = get_children(elem)
    tarr = [child.text.strip().replace("\n", "") for child in children]
    if filter_empty:
        tarr = list(filter(lambda x: x is not None and len(x) > 0, tarr))
    return tarr


def get_shareholding_pattern(shareholding_url):
    page_response = requests.get(shareholding_url, timeout=240)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    tables = page_content.find_all('table')
    if len(tables) < 3:
        return {}
    table_content = page_content.find_all('table')[2]
    rows = table_content.find_all('tr')
    all_tds = page_content.find_all('td')
    idx = list(map(lambda x: x.text, all_tds)).index("Total (A)+(B)+(C)")

    promoters = get_inner_texts_as_array(
        list(filter(lambda x: "Total shareholding of Promoter and Promoter Group (A)" in x.text, rows))[0],
        filter_empty=False)
    public = get_inner_texts_as_array(list(filter(lambda x: "Total Public shareholding (B)" in x.text, rows))[0],
                                      filter_empty=False)
    all_shares = get_inner_texts_as_array(
        list(filter(lambda x: "Total (A)+(B)+(C)" in x.text, page_content.find_all('tr')))[0], filter_empty=False)
    promoters_pledging = ffloat(promoters[7])

    promoters = ffloat(promoters[5])
    public = ffloat(public[5])
    total_shares_count = ffloat(all_tds[idx + 2].text)
    total_pledging = ffloat(all_tds[idx + 7].text)

    return {"promoters": promoters, "public": public, "promoters_pledging": promoters_pledging,
            "total_shares_count": total_shares_count, "total_pledging": total_pledging}


def get_fundholding_pattern(fundholding_url):
    # Funds holding it or not Y
    # Total funds holding currently N
    # percent held by funds
    # buys last quarter
    # sells last quarter
    # no change last quarter
    # Total change in fund holding by money
    # Total change in fund holding by percent shares
    page_response = requests.get(fundholding_url, timeout=240)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    results = {}
    top_tab = page_content.text
    #     print(top_tab)
    if "Not held by Mutual Funds in the last 6 quarters" in top_tab:
        results['mf_holding'] = True
    else:
        results['mf_holding'] = False
    bought = np.nan
    sold = np.nan
    hold = np.nan
    if not results['mf_holding']:
        bl = top_tab.split("Bought by")
        if len(bl) == 2:
            bought = ffloat(bl[1].strip().split(" ")[0])

        sl = top_tab.split("Sold by")
        if len(sl) == 2:
            sold = ffloat(sl[1].strip().split(" ")[0])
        hl = top_tab.split("No change in")
        if len(hl) == 2:
            hold = ffloat(hl[1].strip().split(" ")[0])

    results['mf_bought'] = bought
    results['mf_sold'] = sold
    results['mf_hold'] = hold

    six_quarter = page_content.find('div', attrs={'id': 'div_0'}).find('table', attrs={'class': 'tblfund2'}).find_all('tr')[-1]
    six_quarter = ffloat_list(get_inner_texts_as_array(six_quarter)[1:])
    results['mf_share_count'] = six_quarter[0]
    results['mf_share_count_last_quarter_change'] = six_quarter[0] - six_quarter[1]
    results['mf_six_quarter_share_count'] = six_quarter
    return results


def get_ratios(url):
    page_response = requests.get(url, timeout=240)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    table_content = page_content.find_all('table', attrs={'class': 'table4'})[-1]
    if "Data Not Available" in table_content.text:
        return {}
    dates_html = get_children(get_children(get_children(table_content)[0])[1])[1]

    dates = get_inner_texts_as_array(dates_html)

    ratios_htmls = get_children(get_children(get_children(get_children(table_content)[0])[1])[2])[1:]

    rows = list(map(get_inner_texts_as_array, ratios_htmls))
    ratios = {}
    ratios['dates'] = dates

    for row in rows:
        if len(row) > 1:
            ratios[row[0]] = ffloat_list(row[1:])

    needed_keys = [('dates', 'ratios_dates'),
                   ('Diluted EPS (Rs.)', 'ratios_diluted_eps'),
                   ('Revenue from Operations/Share (Rs.)', 'ratios_revenue_per_share'),
                   ('PBT/Share (Rs.)', 'ratios_pbt_per_share'),
                   ('PBT Margin (%)', 'ratios_pbt_margin_per_share'),
                   ('Total Debt/Equity (X)', 'ratios_de'),
                   ('Asset Turnover Ratio (%)', 'ratios_asset_turnover_ratio'),
                   ('Current Ratio (X)', 'ratios_cr'),
                   ('EV/EBITDA (X)', 'ratios_ev_by_ebitda'),
                   ('Price/BV (X)', 'ratios_pb'),
                   ('MarketCap/Net Operating Revenue (X)','mcap/revenue'),
                   ('Price/Net Operating Revenue','price/revenue')]

    ratios = {your_key[1]: ratios[your_key[0]] if your_key[0] in ratios else [] for your_key in needed_keys}
    return ratios

def get_min_and_three_year_from_screener(table):
    min_value = np.inf
    three_year_value = np.inf
    for row in table:
        if len(row)==2:
            if row[0]=='3 Years:':
                three_year_value = ffloat(row[1].replace('%',''))
            cur_value = ffloat(row[1].replace('%',''))
            min_value = min(min_value,cur_value)
    return min_value,three_year_value

def get_quarterly_results(quarterly_results_table):
    qrt = get_table_simple(quarterly_results_table)
    qres = {}
    qres['dates'] = qrt[0]
    qres['sales'] = ffloat_list(qrt[1][1:])
    qres['operating_profit'] = ffloat_list(qrt[3][1:])
    qres['opm_percent'] = ffloat_list(qrt[4][1:])
    qres['interest'] = ffloat_list(qrt[7][1:])
    qres['pbt'] = ffloat_list(qrt[8][1:])
    return qres

def get_annual_results(annual_results):
    if annual_results is None:
        return {}
    qrt = get_table_simple(annual_results)
    qres = {}
    qres['dates'] = qrt[0]
    qres['sales'] = ffloat_list(qrt[1][1:])
    qres['operating_profit'] = ffloat_list(qrt[3][1:])
    qres['opm_percent'] = ffloat_list(qrt[4][1:])
    qres['interest'] = ffloat_list(qrt[6][1:])
    qres['pbt'] = ffloat_list(qrt[8][1:])
    qres['eps'] = ffloat_list(qrt[11][1:])
    return qres

def get_balance_sheet(balance_sheet):
    if balance_sheet is None:
        return {}
    qrt = get_table_simple(balance_sheet)
    qres = {}
    qres['dates'] = qrt[0]
    qres['borrowings'] = ffloat_list(qrt[3][1:])
    qres['fixed_assets'] = ffloat_list(qrt[6][1:])
    qres['total_assets'] = ffloat_list(qrt[10][1:])
    return qres

def get_cash_flows(cash_flows):
    if cash_flows is None:
        return {}
    qrt = get_table_simple(cash_flows)
    qres = {}
    qres['dates'] = qrt[0]
    qres['net_cash_flow'] = ffloat_list(qrt[4][1:])
    return qres


def get_past_prices(sc_id):
    bse_url = "https://www.moneycontrol.com/tech_charts/bse/his/%s.csv" % sc_id
    nse_url = "https://www.moneycontrol.com/tech_charts/nse/his/%s.csv" % sc_id

    past_prices_nse = pd.read_csv(nse_url, header=None, names=['open', 'high', 'low', 'close', 'volume', 1, 2, 3, 4])[
        ['open', 'high', 'low', 'close', 'volume']]
    past_prices_nse.index = pd.to_datetime(past_prices_nse.index)

    past_prices_bse = pd.read_csv(bse_url, header=None, names=['open', 'high', 'low', 'close', 'volume', 1, 2, 3, 4])[
        ['open', 'high', 'low', 'close', 'volume']]
    past_prices_bse.index = pd.to_datetime(past_prices_bse.index)

    ly = None
    two_year_ago = None
    three_year_ago = None
    five_year_ago = None
    past_prices = past_prices_bse
    for i in range(12):
        try:
            if ly is None:
                ly_t = pd.to_datetime(past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(364 + i, unit='d'))
                ly = past_prices.loc[[ly_t]]
            if two_year_ago is None:
                two_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(730 + i, unit='d'))
                two_year_ago = past_prices.loc[[two_year_ago_t]]
            if three_year_ago is None:
                three_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(1095 + i, unit='d'))
                three_year_ago = past_prices.loc[[three_year_ago_t]]
            if five_year_ago is None:
                five_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(1825 + i, unit='d'))
                five_year_ago = past_prices.loc[[five_year_ago_t]]
        except Exception as e:
            pass

    past_prices = past_prices_nse
    for i in range(12):
        try:
            if ly is None:
                ly_t = pd.to_datetime(past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(364 + i, unit='d'))
                ly = past_prices.loc[[ly_t]]
            if two_year_ago is None:
                two_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(730 + i, unit='d'))
                two_year_ago = past_prices.loc[[two_year_ago_t]]
            if three_year_ago is None:
                three_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(1095 + i, unit='d'))
                three_year_ago = past_prices.loc[[three_year_ago_t]]
            if five_year_ago is None:
                five_year_ago_t = pd.to_datetime(
                    past_prices.iloc[-1:].index.values[0] - pd.to_timedelta(1825 + i, unit='d'))
                five_year_ago = past_prices.loc[[five_year_ago_t]]
        except Exception as e:
            pass

    if len(past_prices_nse) >= len(past_prices_bse):
        past_prices = past_prices_nse
    else:
        past_prices = past_prices_bse
    stock = StockDataFrame.retype(past_prices)
    past_prices['rsi_15'] = stock['rsi_15']
    past_prices['rsi_45'] = stock['rsi_45']
    past_prices['rsi_75'] = stock['rsi_75']
    past_prices['rsi_130'] = stock['rsi_130']
    past_prices['boll_ub'] = stock['boll_ub']
    past_prices['boll_lb'] = stock['boll_lb']

    past_prices['boll_ub_gap'] = (past_prices['boll_lb'] - past_prices['close']) / past_prices['boll_lb']
    past_prices['boll_lb_gap'] = (past_prices['close'] - past_prices['boll_ub']) / past_prices['boll_ub']
    past_prices["weekly_change"] = past_prices[["close"]].rolling(6).agg({"close": lambda x: (x[-1] - x[0]) / x[0]})
    past_prices["monthly_change"] = past_prices[["close"]].rolling(21).agg({"close": lambda x: (x[-1] - x[0]) / x[0]})
    past_prices["3m_change"] = past_prices[["close"]].rolling(65).agg({"close": lambda x: (x[-1] - x[0]) / x[0]})
    past_prices["6m_change"] = past_prices[["close"]].rolling(130).agg({"close": lambda x: (x[-1] - x[0]) / x[0]})

    past_prices["ewm_7_close"] = past_prices['close'].ewm(span=7).mean()
    past_prices["ewm_30_close"] = past_prices['close'].ewm(span=30).mean()
    past_prices["ewm_120_close"] = past_prices['close'].ewm(span=120).mean()

    past_prices["ewm_7_close_diff"] = (past_prices['close'] - past_prices["ewm_7_close"]) / past_prices["ewm_7_close"]
    past_prices["ewm_30_close_diff"] = (past_prices['close'] - past_prices["ewm_30_close"]) / past_prices["ewm_30_close"]
    past_prices["ewm_120_close_diff"] = (past_prices['close'] - past_prices["ewm_120_close"]) / past_prices["ewm_120_close"]

    past_prices["std_7_close"] = past_prices['close'].ewm(span=7).std()
    past_prices["std_30_close"] = past_prices['close'].ewm(span=30).std()
    past_prices["std_120_close"] = past_prices['close'].ewm(span=120).std()

    past_prices["ewm_7_volume"] = past_prices['volume'].ewm(span=7).mean()
    past_prices["ewm_30_volume"] = past_prices['volume'].ewm(span=30).mean()
    past_prices["ewm_120_volume"] = past_prices['volume'].ewm(span=120).mean()

    past_prices["ewm_7_volume_diff"] = (past_prices['volume'] - past_prices["ewm_7_volume"]) / past_prices["ewm_7_volume"]
    past_prices["ewm_30_volume_diff"] = (past_prices['volume'] - past_prices["ewm_30_volume"]) / past_prices["ewm_30_volume"]
    past_prices["ewm_120_volume_diff"] = (past_prices['volume'] - past_prices["ewm_120_volume"]) / past_prices["ewm_120_volume"]

    past_prices["std_7_volume"] = past_prices['volume'].ewm(span=7).std()
    past_prices["std_30_volume"] = past_prices['volume'].ewm(span=30).std()
    past_prices["std_120_volume"] = past_prices['volume'].ewm(span=120).std()

    past_prices.fillna(0, inplace=True)
    df_utils.drop_columns_safely(past_prices, ["close_-1_d", 'close_-1_s', 'rs_45', 'rs_15', 'close_20_sma', 'close_20_mstd',
                                      'boll'], inplace=True)

    res = {"all_past_prices": past_prices, "last_year": ly, "two_year_ago": two_year_ago,
           "three_year_ago": three_year_ago, "five_year_ago": five_year_ago}
    return res


def get_scrip_info(url):
    original_url = url
    key_val_pairs = {}
    key_val_pairs["original_url"] = original_url
    if not url.startswith("http"):
        url = "https://www.moneycontrol.com" + url
    try:
        page_response = requests.get(url, timeout=240)
        if page_response.status_code > 299:
            print("Failed to fetch: %s" % url)
            page_response = requests.get(url, timeout=240)
        page_content = BeautifulSoup(page_response.content, "html.parser")
        scrip_name = None
        name_divs = page_content.find_all('div', attrs={'class': 'gry10'})
        for nd in name_divs:
            texts = list(map(lambda x: x.strip(), nd.text.split(" ")))
            if "NSE:" in texts:
                scrip_name = texts[texts.index("NSE:") + 1]
                scrip_name = re.sub('[^0-9a-zA-Z&\-]+', '', scrip_name)
        if scrip_name is None or len(scrip_name.strip()) == 0 or "ETF" in scrip_name:
            key_val_pairs['failure'] = True
            key_val_pairs['err'] = "%s is not named on NSE" % url
            # print(key_val_pairs['err'])
            return key_val_pairs

        content_div_text = page_content.find('div', attrs={'id': 'content_full'}).text
        if "not listed" in content_div_text or "not traded" in content_div_text:
            key_val_pairs['failure'] = True
            key_val_pairs['err'] = "%s is not listed on both BSE and NSE" % url
            # print(key_val_pairs['err'])
            return key_val_pairs
        price = ffloat(page_content.find('div', attrs={'id': 'Nse_Prc_tick_div'}).text.split(" ")[0].replace(',', ''))
        low = ffloat(page_content.find('span', attrs={'id': 'b_low_sh'}).text.split(" ")[0])
        high = ffloat(page_content.find('span', attrs={'id': 'b_high_sh'}).text.split(" ")[0])
        open_price = ffloat(page_content.find('div', attrs={'id': 'n_open'}).text.split(" ")[0].replace(',', ''))

        today_change = page_content.find('div',attrs={'id':'n_changetext'}).text.strip().split(" ")
        today_change_value = ffloat(today_change[0])
        today_change_percent = ffloat(today_change[1])

        name = page_content.find('h1', attrs={'class': 'company_name'}).text

        screener_url = "https://www.screener.in/company/%s/" % scrip_name
        screener_page_response = requests.get(screener_url, timeout=240)

        if screener_page_response.status_code > 299:
            key_val_pairs['failure'] = True
            key_val_pairs['err'] = "No Screener URL: %s" % screener_url
            # print(key_val_pairs['err'])
            return key_val_pairs
        screener_page_content = BeautifulSoup(screener_page_response.content, "html.parser")
        screener_name = \
        get_children(get_children(screener_page_content.find('nav', attrs={'id': 'fixed-scroll-aid-bar'}))[0])[
            0].text.strip()

        sector = get_children(screener_page_content.find('h1'))[0].text.replace("/", '').strip()
        yearly_high = page_content.find('span', attrs={'id': 'n_52high'}).text.strip()
        yearly_low = page_content.find('span', attrs={'id': 'n_52low'}).text.strip()
        html_data_content = page_content.find('div', attrs={'id': 'mktdet_1'})
        petable = get_table(get_children(html_data_content)[0])
        pbtable = get_table(get_children(html_data_content)[1])

        dma_table = get_table_simple(page_content.find('div', attrs={'id': 'acc_hd2'}).find_all('table')[2])
        # print(dma_table)
        thirty_dma = None
        fifty_dma = None
        one_fifty_dma = None
        two_hundred_dma = None
        if len(dma_table[1]) > 1:
            thirty_dma = dma_table[1][1]
        if len(dma_table[2]) > 1:
            fifty_dma = dma_table[2][1]
        if len(dma_table[3]) > 1:
            one_fifty_dma = dma_table[3][1]
        if len(dma_table[4]) > 1:
            two_hundred_dma = dma_table[4][1]

        side_nav = page_content.find('dl', attrs={'id': 'slider'})
        ratio_url = side_nav.find_all('dd')[2].find_all('a')[7]['href']
        ratio_url = "https://www.moneycontrol.com" + ratio_url
        ratios = get_ratios(ratio_url)

        volume = ffloat(page_content.find('span', attrs={'id': 'nse_volume'}).text)

        sc_id = page_content.find('input', attrs={'id': 'sc_id'}).get('value').lower()

        mf_url = "https://www.moneycontrol.com/mf/user_scheme/mfholddetail_sec.php?sc_did=%s" % sc_id
        shareholding_url = "https://www.moneycontrol.com" + side_nav.find_all('dd')[4].find_all('a')[0]['href']
        shareholdings = get_shareholding_pattern(shareholding_url)
        mfs = get_fundholding_pattern(mf_url)

        key_val_pairs = {**key_val_pairs, **mfs, **shareholdings, **ratios}

        past_prices = get_past_prices(sc_id)

        l_yp = None
        two_yp = None
        three_yp = None
        five_yp = None
        gain_loss_l_yp = None
        gain_loss_two_yp = None
        gain_loss_three_yp = None
        gain_loss_five_yp = None

        if is_dataframe(past_prices['last_year']):
            l_yp = past_prices['last_year']['close'].values[0]
            gain_loss_l_yp = (price - l_yp) * 100 / l_yp
        if is_dataframe(past_prices['two_year_ago']):
            two_yp = past_prices['two_year_ago']['close'].values[0]
            gain_loss_two_yp = (price - two_yp) * 100 / two_yp
        if is_dataframe(past_prices['three_year_ago']):
            three_yp = past_prices['three_year_ago']['close'].values[0]
            gain_loss_three_yp = (price - three_yp) * 100 / three_yp
        if is_dataframe(past_prices['five_year_ago']):
            five_yp = past_prices['five_year_ago']['close'].values[0]
            gain_loss_five_yp = (price - five_yp) * 100 / five_yp

        quarterly_results = get_quarterly_results(
            screener_page_content.find('section', attrs={'id': 'quarters'}).find('table'))

        annual_results_table = screener_page_content.find('section', attrs={'id': 'profit-loss'}).find('table', attrs={
            'class': 'data-table'})
        annual_results = None
        if annual_results_table is not None:
            annual_results = get_annual_results(annual_results_table)

        csg_table = get_table_simple(
            screener_page_content.find('section', attrs={'id': 'profit-loss'}).find_all('table', attrs={
                'class': 'ranges-table'})[0])
        min_csg, three_year_csg = get_min_and_three_year_from_screener(csg_table)
        cpg_table = get_table_simple(
            screener_page_content.find('section', attrs={'id': 'profit-loss'}).find_all('table', attrs={
                'class': 'ranges-table'})[1])
        min_cpg, three_year_cpg = get_min_and_three_year_from_screener(cpg_table)
        roe_table = get_table_simple(
            screener_page_content.find('section', attrs={'id': 'profit-loss'}).find_all('table', attrs={
                'class': 'ranges-table'})[2])
        min_roe, three_year_roe = get_min_and_three_year_from_screener(roe_table)

        balance_sheet = get_balance_sheet(
            screener_page_content.find('section', attrs={'id': 'balance-sheet'}).find('table'))
        cash_flows = get_cash_flows(screener_page_content.find('section', attrs={'id': 'cash-flow'}).find('table'))

        data_table = list()
        data_table.extend(petable)
        data_table.extend(pbtable)

        consolidated_html_data_content = page_content.find('div', attrs={'id': 'mktdet_2'})
        consolidated_petable = get_table(get_children(consolidated_html_data_content)[0])
        consolidated_pbtable = get_table(get_children(consolidated_html_data_content)[1])
        consolidated_data_table = list()
        consolidated_data_table.extend(consolidated_petable)
        consolidated_data_table.extend(consolidated_pbtable)

        for row in consolidated_data_table:

            k = row[0]['text']
            if len(row) < 2:
                v = None
            else:
                v = row[1]['text'].split(" ")[0].replace(',', '')
            key_val_pairs[k] = v

        for row in data_table:

            k = row[0]['text']
            if len(row) < 2:
                v = None
            else:
                v = row[1]['text'].split(" ")[0].replace(',', '')

            if k not in key_val_pairs or not isfloat(key_val_pairs[k]):
                key_val_pairs[k] = v

        key_val_pairs["pe"] = ffloat(key_val_pairs.pop('P/E'))
        key_val_pairs["book_value"] = ffloat(key_val_pairs.pop('BOOK VALUE (Rs)'))
        key_val_pairs["deliverables"] = ffloat(key_val_pairs.pop('DELIVERABLES (%)'))
        key_val_pairs["eps"] = ffloat(key_val_pairs.pop('EPS (TTM)'))
        key_val_pairs["industry_pe"] = ffloat(key_val_pairs.pop('INDUSTRY P/E'))
        if 'MARKET CAP (Rs Cr)' in key_val_pairs:
            key_val_pairs["market_cap"] = key_val_pairs.pop('MARKET CAP (Rs Cr)')
        elif '**MARKET CAP (Rs Cr)' in key_val_pairs:
            key_val_pairs["market_cap"] = key_val_pairs.pop('**MARKET CAP (Rs Cr)')
        key_val_pairs["market_cap"] = ffloat(key_val_pairs["market_cap"])
        key_val_pairs["pb"] = ffloat(key_val_pairs.pop('PRICE/BOOK'))
        key_val_pairs["pc"] = ffloat(key_val_pairs.pop('P/C'))
        key_val_pairs['price'] = ffloat(price)
        key_val_pairs['today_change_value'] = today_change_value
        key_val_pairs['today_change_percent'] = today_change_percent
        key_val_pairs['low'] = low
        key_val_pairs['high'] = high
        key_val_pairs['open'] = open_price
        key_val_pairs['volume'] = volume
        key_val_pairs["name"] = name
        key_val_pairs["scrip_name"] = scrip_name
        key_val_pairs["yearly_low"] = ffloat(yearly_low)
        key_val_pairs["yearly_high"] = ffloat(yearly_high)

        key_val_pairs["min_csg"] = min_csg
        key_val_pairs["three_year_csg"] = three_year_csg
        key_val_pairs["min_cpg"] = min_cpg
        key_val_pairs["three_year_cpg"] = three_year_cpg
        key_val_pairs["min_roe"] = min_roe
        key_val_pairs["three_year_roe"] = three_year_roe
        key_val_pairs["peg"] = ffloat(key_val_pairs["pe"]) / three_year_cpg
        if np.isnan(three_year_cpg):
            key_val_pairs["peg"] = ffloat(key_val_pairs["pe"]) / min_cpg
        key_val_pairs["min_recovery_year"] = get_year_when_cumulative_profit_over_pe(ffloat(key_val_pairs["pe"]),
                                                                                     three_year_cpg)
        key_val_pairs['sector'] = sector
        key_val_pairs['thirty_dma'] = ffloat(thirty_dma)
        key_val_pairs['fifty_dma'] = ffloat(fifty_dma)
        key_val_pairs['one_fifty_dma'] = ffloat(one_fifty_dma)
        key_val_pairs['two_hundred_dma'] = ffloat(two_hundred_dma)

        key_val_pairs['l_yp'] = l_yp
        key_val_pairs['two_yp'] = two_yp
        key_val_pairs['three_yp'] = three_yp
        key_val_pairs['five_yp'] = five_yp
        key_val_pairs['gain_loss_l_yp'] = gain_loss_l_yp
        key_val_pairs['gain_loss_two_yp'] = gain_loss_two_yp
        key_val_pairs['gain_loss_three_yp'] = gain_loss_three_yp
        key_val_pairs['gain_loss_five_yp'] = gain_loss_five_yp

        key_val_pairs['de'] = np.nan
        key_val_pairs['ev_by_ebitda'] = np.nan
        if "ratios_ev_by_ebitda" in key_val_pairs and len(key_val_pairs["ratios_ev_by_ebitda"]) > 0:
            key_val_pairs['ev_by_ebitda'] = key_val_pairs["ratios_ev_by_ebitda"][0]

        if "ratios_de" in key_val_pairs and len(key_val_pairs["ratios_de"]) > 0:
            key_val_pairs['de'] = key_val_pairs["ratios_de"][0]
        key_val_pairs['quarterly_results'] = quarterly_results
        key_val_pairs['annual_results'] = annual_results

        key_val_pairs['balance_sheet'] = balance_sheet
        key_val_pairs['cash_flows'] = cash_flows
        key_val_pairs['past_prices'] = past_prices
        key_val_pairs['failure'] = False

        del key_val_pairs['DIV (%)']
        del key_val_pairs['DIV YIELD.(%)']
        del key_val_pairs['FACE VALUE (Rs)']
        del key_val_pairs['Market Lot']
    except Exception as e:
        #         raise e
        traceback.print_exc()
        key_val_pairs['failure'] = True
        key_val_pairs['err'] = "Error for: %s" % original_url
        print(key_val_pairs['err'])
        return key_val_pairs

    return key_val_pairs




def get_scrip_info_by_nse_name(nse_name):
    url = "https://www.moneycontrol.com/mccode/common/autosuggesion.php?classic=true&query=%s&type=1&format=json"%nse_name
    page_response = requests.get(url, timeout=240)
    json_text = page_response.text
    data = json.loads(json_text)
    if len(data)>1:
        scrips = pd.DataFrame.from_records(data)["pdt_dis_nm"].values
        idx = list(map(lambda x:BeautifulSoup(x, "html.parser").find("span").text.split(",")[1].strip(),scrips)).index(nse_name)
        scrip_url = data[idx]['link_src']
    else:
        scrip_url = data[0]['link_src']
    return get_scrip_info(scrip_url)

def get_all_company_from_mf(mfid,threadpool_size=8):
    portfolio_table,fund_name = get_portfolio(mfid = mfid)
    table_data = get_table(portfolio_table)[1:]
    scrip_col = 0
    links = list(map(lambda x:x[scrip_col]['href'],table_data))
    pool = ThreadPool(threadpool_size)
    scrip_details = pool.map(get_scrip_info, links)
    scrip_details = list(filter(lambda x:x is not None,scrip_details))
    length1 = len(scrip_details)
    failed = list(filter(lambda x:x['failure'],scrip_details))
    scrip_details = list(filter(lambda x:not x['failure'],scrip_details))
    length2 = len(scrip_details)
    print("Scrips which failed to fetch = %s"%(length1-length2))
    print(failed)
    scrip_details = {scrip['scrip_name']:scrip for scrip in scrip_details}
    return scrip_details


def get_pe_filter(params={"mcap": [1e2, 1e3, 5e3, 1e4, 2e4], "pe": [1, 5, 10, 15, 20], "mcap_lower_limit": 1e2,
                          "pe_upper_limit": 25}):
    def filter_fn(stock_detail):
        x = params['mcap']
        y = params['pe']
        pe = ffloat(stock_detail['pe'])
        mcap = ffloat(stock_detail['market_cap'])

        if np.isnan(pe) or np.isnan(mcap):
            return False
        if pe > params['pe_upper_limit']:
            return False
        if mcap < params['mcap_lower_limit']:
            return False

        right = np.searchsorted(params['mcap'], mcap)

        if right == 0:
            right = right + 1

        if right == len(x):
            right = right - 1

        left = right - 1
        coefficients = np.polyfit([x[left], x[right]], [y[left], y[right]], 1)
        polynomial = np.poly1d(coefficients)
        pe_value = polynomial(mcap)
        if pe <= pe_value:
            return True
        return False

    return filter_fn


def get_pb_filter(params={"mcap": [1e2, 5e2, 1e3, 2e3, 6e3], "pb": [1, 2, 3, 4, 5], "pb_upper_limit": 5}):
    def filter_fn(stock_detail):
        x = params['mcap']
        y = params['pb']
        pb = ffloat(stock_detail['pb'])
        bv = ffloat(stock_detail['book_value'])
        mcap = ffloat(stock_detail['market_cap'])

        if np.isnan(pb) or np.isnan(bv) or pb > params['pb_upper_limit'] or bv < 0:
            return False

        if pb > params['pb_upper_limit']:
            return False

        right = np.searchsorted(params['mcap'], mcap)

        if right == 0:
            right = right + 1

        if right == len(x):
            right = right - 1

        left = right - 1
        coefficients = np.polyfit([x[left], x[right]], [y[left], y[right]], 1)
        polynomial = np.poly1d(coefficients)
        pb_value = polynomial(mcap)
        if pb <= pb_value:
            return True
        return False

    return filter_fn


def get_profitability_filter(params={"peg_lower_limit": 0, "peg_upper_limit": 3,
                                     "min_recovery_year": 15, "min_cpg_lower_limit": 0}):
    def filter_fn(stock_detail):
        peg = ffloat(stock_detail['peg'])
        min_cpg = ffloat(stock_detail['min_cpg'])
        min_recovery_year = ffloat(stock_detail['min_recovery_year'])

        if np.isnan(peg) or np.isnan(min_cpg) or np.isnan(min_recovery_year):
            return False

        if peg <= params['peg_upper_limit'] and peg >= params['peg_lower_limit'] and min_recovery_year <= params[
            'min_recovery_year'] and min_cpg > params['min_cpg_lower_limit']:
            return True
        return False

    return filter_fn


def get_generic_filter(param_name, lower_limit=None, upper_limit=None,
                       replacement_nan=None, replacement_not_present=None):
    def filter_fn(stock_detail):
        param = replacement_not_present
        if param_name in stock_detail:
            param = ffloat(stock_detail[param_name])

        if np.isnan(param):
            param = replacement_nan

        if param is None or np.isnan(param):
            return False

        if param <= upper_limit and param >= lower_limit:
            return True
        return False

    return filter_fn


def get_generic_filter_two_variables(x, y, xvar_name, yvar_name, accept_lower=True,
                                     pass_not_found=False):
    def filter_fn(stock_detail):
        if x is None or y is None or xvar_name is None or yvar_name is None:
            raise ValueError("Incorrect Parameters")

        if xvar_name not in stock_detail or yvar_name not in stock_detail:
            return pass_not_found

        xval = stock_detail[xvar_name]
        yval = stock_detail[yvar_name]

        right = np.searchsorted(xvar, xval)

        if right == 0:
            right = right + 1

        if right == len(x):
            right = right - 1

        left = right - 1
        coefficients = np.polyfit([x[left], x[right]], [y[left], y[right]], 1)
        polynomial = np.poly1d(coefficients)
        yt_value = polynomial(xval)
        if yval <= yt_value == accept_lower:
            return True

        return False

    return filter_fn


def get_stock_urls_from_listing_page(listing_page):
    page_response = requests.get(listing_page, timeout=240)
    page_content = BeautifulSoup(page_response.content, "html.parser")
    urls_table = page_content.find('table',attrs={'class':'pcq_tbl'})
    links = list(map(lambda x:get_children(x)[0]['href'],urls_table.find_all('td')))
    return links

def get_all_links(threadpool_size=8):
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letters = [letter for letter in abc]
    listing_page_urls = ['https://www.moneycontrol.com/india/stockpricequote/'+letter for letter in letters]
    pool = ThreadPool(threadpool_size)
    all_links = list(more_itertools.flatten(pool.map(get_stock_urls_from_listing_page, listing_page_urls)))
    return all_links


def get_all_company_details(accumulator={}, failures=[], size=10000, start=None, end=None,
                            threadpool_size=8, ignore_failures=True,
                            ignore_success=True, randomize=False):
    # filters is a list of functions returning T/F, They are always
    batch_size = 5 * threadpool_size
    all_links = list(set(get_all_links()))
    print("Total Number of links = %s" % (len(all_links)))
    if ignore_success:
        all_links = list(set(all_links) - set([scrip['original_url'] for scrip in accumulator.values()]))
    if ignore_failures:
        all_links = list(set(all_links) - set([scrip['original_url'] for scrip in failures]))
    print("Total Links after removing success and failures = %s" % (len(all_links)))
    all_links = sorted(all_links)
    if start is not None and end is not None:
        all_links = all_links[start:end]
    all_links = all_links[:size]
    if randomize:
        shuffle(all_links)

    print("Total Links To Process = %s" % (len(all_links)))
    pool = ThreadPool(threadpool_size)
    batches = int(np.ceil(len(all_links) / batch_size))

    for batch_num in range(batches):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, len(all_links))
        print("start = %s, end = %s" % (start, end))
        this_batch = all_links[start:end]
        scrip_details = pool.map(get_scrip_info, this_batch)
        scrip_details = list(filter(lambda x: x is not None, scrip_details))
        fails = list(filter(lambda x: x['failure'], scrip_details))
        scrip_details = list(filter(lambda x: not x['failure'], scrip_details))
        for scrip in scrip_details:
            accumulator[scrip['scrip_name']] = scrip
        failures.extend(fails)
        failures = {failure['original_url']: failure for failure in failures}
        failures = list(failures.values())


def filter_companies(all_scrips, filters=[]):
    scrip_details = list(all_scrips.values())

    for i in range(len(filters)):
        scrip_details = list(filter(filters[i], scrip_details))

    return scrip_details


def get_df_from_scrip_details(scrip_details):
    other_cols = ['name', 'scrip_name']
    numeric_cols = ['book_value', 'price', 'deliverables', 'eps', 'industry_pe',
                    'market_cap', 'pb', 'pc', 'pe', 'de',
                    'yearly_high', 'yearly_low', 'min_csg', 'three_year_csg', 'min_cpg', 'three_year_cpg',
                    'min_roe', 'three_year_roe', 'peg', 'min_recovery_year',
                    'l_yp', 'two_yp', 'three_yp', 'five_yp', 'gain_loss_l_yp', 'gain_loss_two_yp',
                    'gain_loss_three_yp']

    all_cols = other_cols + numeric_cols
    scrip_details = [{your_key: scrip[your_key] for your_key in all_cols} for scrip in scrip_details]
    scrip_details = pd.DataFrame.from_records(scrip_details)
    scrip_details[numeric_cols] = scrip_details[numeric_cols].applymap(ffloat)
    scrip_details = scrip_details[all_cols]
    return scrip_details


def score_company_on_filters(all_scrips, filters={}):
    all_scrips = list(all_scrips.values())
    other_cols = ['name', 'scrip_name']
    numeric_cols = ['price', 'industry_pe',
                    'market_cap', 'pb', 'pe', 'de',
                    'yearly_high', 'yearly_low', 'three_year_csg', 'three_year_cpg',
                    'peg', 'min_recovery_year',
                    'l_yp', 'three_yp', 'five_yp']
    all_cols = other_cols + list(filters.keys()) + numeric_cols
    scrip_details = []
    for scrip in all_scrips:
        for key in filters.keys():
            scrip[key] = filters[key](scrip)
        scrip_detail = {your_key: scrip[your_key] for your_key in all_cols}

        scrip_details.append(scrip_detail)

    scrip_details = pd.DataFrame.from_records(scrip_details)
    scrip_details[numeric_cols] = scrip_details[numeric_cols].applymap(ffloat)
    scrip_details = scrip_details[all_cols]
    return scrip_details


def generate_price_chart(stock_df, name, days=1095, ewmas=[], other_technical_indicators=[]):
    plt.figure(figsize=(16, 8))
    ts_df = stock_df.tail(days)
    handles = []
    p1, = plt.plot(ts_df.index, ts_df['close'], label="price")
    handles.append(p1)
    for ewma in ewmas:
        y = ts_df['close'].ewm(span=ewma).mean()
        p2, = plt.plot(ts_df.index, y, label="%s day ewma" % ewma)
        handles.append(p2)
    plt.legend(handles=handles)
    plt.title(name)
    plt.ylabel('Closing Price')
    plt.show()


def generate_price_volume_chart(stock_df, name, days=1095, ewmas=[], other_technical_indicators=[]):
    plt.figure(figsize=(16, 8))
    top = plt.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=6)
    bottom = plt.subplot2grid((6, 6), (4, 0), rowspan=2, colspan=6)
    ts_df = stock_df.tail(days)
    handles = []
    p1, = top.plot(ts_df.index, ts_df['close'], label="price")
    handles.append(p1)
    for ewma in ewmas:
        y = ts_df['close'].ewm(span=ewma).mean()
        p2, = top.plot(ts_df.index, y, label="%s day ewma" % ewma)
        handles.append(p2)
    top.legend(handles=handles)
    bottom.bar(ts_df.index, ts_df['volume'])
    bottom.set_ylim([ts_df['volume'].min(), ts_df['volume'].max()])
    #     sns.lineplot(x="index", y="close", data=ts_df.reset_index(),ax=top)
    #     sns.barplot(x="index", y="volume", data=ts_df.reset_index(),ax=bottom)

    # set the labels
    top.axes.get_xaxis().set_visible(False)
    top.set_title(name)
    top.set_ylabel('Closing Price')
    bottom.set_ylabel('Volume')

    plt.show()


def generate_returns_chart(stocks, days=1095):
    plt.figure(figsize=(16, 8))
    stocks = {key: stocks[key].tail(days).apply(lambda x: x / x[0]) for key in stocks.keys()}
    handles = []
    for key in stocks.keys():
        y = stocks[key]['close']
        p2, = plt.plot(stocks[key].index, y, label=key)
        handles.append(p2)
    plt.legend(handles=handles)
    plt.title("Comparative returns")
    plt.ylabel('Comparative Returns')

    plt.show()


def generate_rolling_returns_chart(stocks, days=1095, rolling=252):
    plt.figure(figsize=(16, 8))
    stocks = {key: stocks[key][['close']] for key in stocks.keys()}

    # Only take intersection of all indexes (dates) else rolling calculation will be screwed up
    indexes = None
    for key in stocks.keys():
        stock = stocks[key].tail(days + rolling)
        if indexes is None:
            indexes = set(stock.index)
        else:
            indexes = indexes.intersection(set(stock.index))
    for key in stocks.keys():
        stock = stocks[key]
        stock = stock[stock.index.isin(indexes)]
        stocks[key] = stock

    for df in stocks.values():
        df[["close"]] = df[["close"]].rolling(rolling).agg({"close": lambda x: (x[-1] - x[0]) * 100 / x[0]})

    stocks = {key: stocks[key].tail(days)[['close']] for key in stocks.keys()}
    handles = []
    for key in stocks.keys():
        y = stocks[key]['close']
        p2, = plt.plot(stocks[key].index, y, label=key)
        handles.append(p2)
    plt.legend(handles=handles)
    plt.title("Rolling returns")
    plt.ylabel('Rolling Returns')
    plt.show()
    for key in stocks.keys():
        stocks[key]['name'] = key

    all_stocks = pd.concat(list(stocks.values()))

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(x="name", y="close", data=all_stocks, ax=ax);
    ax.xaxis.set_tick_params(rotation=90)
    plt.title("Rolling Returns variation")
    plt.show()


def generate_percent_change_chart(stocks,days=1095):
    plt.figure(figsize=(16,8))
    stocks = {key:stocks[key].tail(days).pct_change()*100 for key in stocks.keys()}
    handles = []
    for key in stocks.keys():
        stocks[key]['name'] = key
        y = stocks[key]['close']
        p2, = plt.plot(stocks[key].index, y,label=key)
        handles.append(p2)
    all_stocks = pd.concat(list(stocks.values()))
    plt.legend(handles=handles)
    plt.title("Daily Percent Changes Chart")
    plt.ylabel('Daily Percent Changes')
    plt.show()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.boxplot(x="name", y="close", data=all_stocks,ax=ax);
    ax.xaxis.set_tick_params(rotation=90)
    plt.show()


def get_all_details_for_mf(scrip_links_table, percent_col=4, scrip_col=0, threadpool_size=8):
    percent_col = 4
    scrip_col = 0
    qty_col = 2
    total_value_crores_col = 3

    def scrip_detail_collector(row):
        scrip_url = row[scrip_col]['href']
        scrip_detail = get_scrip_info(scrip_url)
        try:

            scrip_detail['percent'] = row[percent_col]['text']
            scrip_detail['name'] = row[scrip_col]['text']
            scrip_detail['qty'] = row[qty_col]['text']
            scrip_detail['total_value_crores'] = row[total_value_crores_col]['text']
        except Exception as e:
            print(scrip_url)
        return scrip_detail

    pool = ThreadPool(threadpool_size)
    scrip_details = pool.map(scrip_detail_collector, scrip_links_table)
    scrip_details = list(filter(lambda x: x is not None, scrip_details))
    length1 = len(scrip_details)
    scrip_details = list(filter(lambda x: not x['failure'], scrip_details))
    length2 = len(scrip_details)
    print("Scrips which failed to fetch = %s" % (length1 - length2))
    scrip_details = pd.DataFrame.from_records(scrip_details)
    numeric_cols = ['book_value', 'price', 'deliverables', 'eps', 'industry_pe',
                    'market_cap', 'pb', 'pc', 'pe', 'percent', 'qty', 'total_value_crores',
                    'yearly_high', 'yearly_low', 'min_csg', 'three_year_csg', 'min_cpg', 'three_year_cpg',
                    'min_roe', 'three_year_roe', 'peg', 'min_recovery_year']
    scrip_details[numeric_cols] = scrip_details[numeric_cols].applymap(ffloat)
    return scrip_details


def fund_returns_analysis(fund_list, benchmark_index_prices={}, days=1095,rolling=252):
    fund_prices = {}
    for fund in fund_list:
        portfolio_table, fund_name = get_portfolio(mfid=fund)
        prices_df = pd.read_csv("https://www.moneycontrol.com/mf/mf_info/mf_graph.php?im_id=%s" % fund, header=None,
                                names=['open', 'high', 'low', 'close', 'volume'])[
            ['open', 'high', 'low', 'close', 'volume']]
        prices_df.index = pd.to_datetime(prices_df.index)
        fund_prices[fund_name] = prices_df
    generate_returns_chart({**fund_prices, **benchmark_index_prices}, days=days)
    generate_percent_change_chart({**fund_prices, **benchmark_index_prices}, days=days)
    generate_rolling_returns_chart({**fund_prices, **benchmark_index_prices}, days=days,rolling=rolling)

def comparative_analysis(fund_list,threadpool_size=8):
    fund_details = list()
    for fund in fund_list:
        portfolio_table,fund_name = get_portfolio(mfid = fund)
        table_data = get_table(portfolio_table)
        scrip_details = get_all_details_for_mf(table_data[1:],threadpool_size=threadpool_size)
        pe = np.dot(scrip_details['price'].fillna(0),scrip_details['percent'])/np.dot(scrip_details['eps'].fillna(0),scrip_details['percent'])
        three_year_cpg = np.dot(scrip_details['three_year_cpg'].fillna(0),scrip_details['percent']/100)
        peg = pe/three_year_cpg
        pb = np.dot(scrip_details['price'].fillna(0),scrip_details['percent'])/np.dot(scrip_details['book_value'].fillna(0),scrip_details['percent'])
        aum = np.sum(scrip_details['total_value_crores'])
        avg_market_cap = np.dot(scrip_details['market_cap'].fillna(0),scrip_details['percent']/100)
        min_recovery_year = get_year_when_cumulative_profit_over_pe(pe,three_year_cpg)
        prices_df = pd.read_csv("https://www.moneycontrol.com/mf/mf_info/mf_graph.php?im_id=%s"%fund,header=None,names=['open','high','low','close','volume'])[['open','high','low','close','volume']]
        prices_df.index = pd.to_datetime(prices_df.index)
        fund_detail = {"name":fund_name,"pe":pe,"peg":peg,"pb":pb,"aum":aum,"avg_market_cap":avg_market_cap,"three_year_cpg":three_year_cpg,"min_recovery_year":min_recovery_year,"past_prices":prices_df}
        fund_details.append(fund_detail)
    return pd.DataFrame.from_records(fund_details)






