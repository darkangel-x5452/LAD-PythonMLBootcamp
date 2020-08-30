import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def samples():
    df1 = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df1')
    df2 = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df2')
    df1_plot = df1.plot
    # df1['A'].hist(bins=30)
    # df1['A'].plot(kind='hist', bins=30)
    # df1['A'].plot.hist()
    # print(df1.head())
    # print(df1['Unnamed: 0'])
    # df1_plot.line(x='Unnamed: 0', y='B', figsize=(12,3), lw=1)
    # df1_plot.scatter(x='A', y='B', c='C', cmap='coolwarm')
    # df1_plot.scatter(x='A', y='B', s=df1['C']*100)
    # df2_plot = df2.plot
    # df2.plot.area(alpha=0.4)
    # df2_plot.bar(stacked=True)
    # df2.plot.box()
    df2['a'].plot.kde()
    df2.plot.density()

    # df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
    # df.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')





    plt.show()


def exercises():
    df3 = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/07-Pandas-Built-in-Data-Viz/df3')
    # df3.plot.scatter(x='a', y='b',  figsize=(12,3))
    # plt.xlim(-0.2, 1.2)
    # plt.xlim(-0.2, 1.2)

    # df3['a'].plot.hist()
    # plt.style.use('ggplot')
    # df3['a'].plot.hist(bins=30)
    # df3[['a', 'b']].plot.box()
    # df3['d'].plot.kde(lw=10,linestyle='--')
    df3.iloc[0:30].plot.area()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples()
    exercises()
