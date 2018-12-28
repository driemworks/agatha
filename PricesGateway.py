import quandl
import pickle
import pandas as pd
import os
import datetime
import io
import requests

"""
quandl api key DrqX5vpnfbEwa8Lsxirx
"""
quandl.ApiConfig.api_key = 'DrqX5vpnfbEwa8Lsxirx'


def get_quandl_data(quandl_id):
    """
    download and cache quandl dataseries
    :param quandl_id: The quandl id
    :return: The data as a pandas dataframe
    """
    cache_path = 'resources/prices/' + '{}.pkl'.format(quandl_id).replace('/', '-')
    if os.path.exists(cache_path):
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    else:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def get_json_data(json_url, cache_path):
    if os.path.exists(cache_path):
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    else:
        with open(cache_path, 'wb') as f:
            print('Downloading {}'.format(json_url))
            df = pd.read_json(json_url)
            df.to_pickle(f)
            print('Cached {} at {}'.format(json_url, cache_path))
    return df


def get_quandl_stock_data(ticker, cache_path, start_date='1996-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    print('Retrieving data from {} to {} for ticker {}'.format(start_date, end_date, ticker))
    cache_path_formatted = cache_path.format(ticker, start_date, end_date)
    if os.path.exists(cache_path_formatted):
        f = open(cache_path, 'rb')
        data = pickle.load(f)
        print('Loaded {} data from cache'.format(ticker))

    else:
        data = quandl.get_table('WIKI/PRICES', ticker=[ticker],
                                qopts={'columns': ['ticker', 'date', 'close', 'adj_close']},
                                date={'gte': start_date, 'lte': end_date})
        # reverse order so newest is at end of list
        data = data[::-1]
        with open(cache_path_formatted, 'wb') as f:
            data.to_pickle(f)
            print('Cached {} data at {}'.format(ticker, cache_path))

    return data

base_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'

alpha_vantage_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}&datatype=csv'

def get_alphavantage_data(ticker, cache_path):
    if os.path.exists(cache_path):
        f = open(cache_path, 'rb')
        data = pickle.load(f)
        print('Loaded {} data from cache'.format(ticker))
    else:
        url = alpha_vantage_url.format(ticker, ' 1YB64755JWVBCSU3')
        content = requests.get(url).content
        data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # reverse order so newest is at end of list
        data = data[::-1]
        with open(cache_path, 'wb') as f:
            data.to_pickle(f)
            print('Cached {} data at {}'.format(ticker, cache_path))

    return data


def get_poloniex_data(currency_pair, start, end, period):
    """
    Make API call to poloniex to get chart data
    :return: the currency pair's dataframe
    """
    url = base_url.format(currency_pair, start, end, period)
    data_df = get_json_data(url, 'resources/prices/{}_{}_{}_{}.pkl'.format(currency_pair, start, end, period))
    data_df = data_df.set_index('date')
    return data_df

