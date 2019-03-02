import pickle
import pandas as pd
import os
import io
import requests

alpha_vantage_url = 'https://www.alphavantage.co/query?' \
					'function=TIME_SERIES_DAILY&symbol={}&outputsize=full&apikey={}&datatype=csv'


def get_alpha_vantage_data(api_key, ticker, cache_path):
    '''
    Download data from alphavantage, or retrieve from the cache_path if it exists
    :param api_key: The alphavantage api key
    :param ticker: The stock ticker
    :param cache_path: The cache path
    :return: The alphavantage data
    '''
    if os.path.exists(cache_path):
        f = open(cache_path, 'rb')
        data = pickle.load(f)
        print('Loaded {} data from cache'.format(ticker))
    else:
        url = alpha_vantage_url.format(ticker, api_key)
        content = requests.get(url).content
        data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # reverse order so newest is at end of list
        data = data[::-1]
        if not cache_path == None:
            with open(cache_path, 'wb') as f:
                data.to_pickle(f)
                print('Cached {} data at {}'.format(ticker, cache_path))

    return data
