import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.plotly as py

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sb
import plotly
import cufflinks as cf
cf.go_offline()

sb.set_style('whitegrid')

from pandas_datareader import data, wb
import datetime

def exercises():
    # df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv')
    df = pd.read_pickle('../../Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/all_banks')
    start = datetime.datetime(2006, 1, 1)
    end = datetime.datetime(2016, 1, 1)
    tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

    d = {}
    for ticker in tickers:
        try:
           d[ticker] = pd.read_csv(ticker + '.csv', index_col='Date')
        except:
            BAC = data.DataReader(ticker, 'yahoo', start=start, end=end)
            BAC.to_csv(ticker + '.csv')
            d[ticker] = pd.read_csv(ticker + '.csv', index_col='Date')
    tickers_d = [d['BAC'], d['C'], d['GS'], d['JPM'], d['MS'], d['WFC']]
    bank_stocks = pd.concat(tickers_d, keys=tickers, axis=1)
    bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']
    new_df = bank_stocks.xs(level='Stock Info', axis=1, key='Close').max()

    returns = pd.DataFrame()

    for ticker in tickers:
        # returns[ticker + ' Return'] = bank_stocks.xs(key=ticker, axis=1)['Close'].pct_change()
        returns[ticker + ' Return'] = bank_stocks[ticker]['Close'].pct_change()
    # print(returns)
    new_df_min = returns.idxmin()
    new_df_max = returns.idxmax()
    # sb.pairplot(returns[1:])
    # plt.show()

    new_df = returns.std()
    ms_returns = returns['MS Return'].loc['2015-01-01':'2015-12-31']
    c_returns = returns['MS Return'].loc['2008-01-01':'2008-12-31']
    # sb.distplot(ms_returns, bins=60)
    # sb.distplot(c_returns, bins=60)
    # plt.show()

    # banks_close = bank_stocks.xs(level='Stock Info', key='Close', axis=1)
    # banks_close.plot()

    # new_df = d['BAC']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot()
    # new_df = d['BAC']['Close'].loc['2008-01-01':'2009-01-01'].plot()
    # plt.show()

    banks_close_corr = bank_stocks.xs(level='Stock Info', key='Close', axis=1).corr()
    # sb.heatmap(banks_close_corr)
    # banks_close_corr.iplot(kind='heatmap')
    d['BAC'][['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle')
    # plt.show()
    # print(banks_close)
    print('bye')


if __name__ == '__main__':
    exercises()