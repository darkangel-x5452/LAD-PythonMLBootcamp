import numpy as np
import pandas as pd

def hello():
    arr = np.arange(0, 11)
    # df[['w', 'y']] # select columns
    # df.loc[['a', 'b']] # select rows
    # df.loc[['a', 'b'], ['w', 'y']] # select rows
    # df[df>0]
    #df[df['w']>0]
    # df[(df['w']>0) & (df['y']>0)] # needs & and not 'and' syntax
    # df.reset_index()
    # df.set_index('states')
    #pd.MultiIndex.from_tuples(jier_index)
    # df.xs # get cross section
    # df.dropna(thresh=2) # at least 2 nan rows
    # df.fillna(value='Fill value')
    # df[a].fillna(value=df['a'].mean())
    #df. groupby('company').sum().loc['FB'] # or .mean()
    # df.groupby('company').describe().transpose()['fb']
    # pd.concat([df1, df2, df3], axis=1)
    # pd.merge(leftdf, rightdf, how='inner', on='key')
    # pd.merge(leftdf, rightdf, on=['key1', 'key2'])
    # leftdf.join(rightdf)
    # df['col2'].nunique()
    # df['col2'].unique()
    # df['col2'].value_counts()
    # df[(df['col1']>2) & (df['col2']==444)]
    # df['col1'].apply(custom_function)
    # df['col1'].apply(lambda x:x*2)
    # df.sort_values('col1')
    # df.isnull()
    # df.pviot_table(values='d', index=['a', 'b'], columns=['c'])
    # pd.read_html
if __name__ == '__main__':
    hello()