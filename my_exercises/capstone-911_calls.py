import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.plotly as py

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sb

def exercises():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv')
    print(df.head())

    new_df = df['twp'].value_counts().head(5)
    new_df = df['title'].nunique()

    """
    ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.**

*For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. *
    """
    df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
    # new_df = df['title'].apply(lambda x: x.split(':')[0])
    new_df = df['Reason'].value_counts().head(5)
    # sb.countplot(data=df, x=df['Reason'])
    # plt.show()

    new_df = type(df['timeStamp'].iloc[0])
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    # time = df['timeStamp'].iloc[0]
    df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
    df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
    df['Month'] = df['timeStamp'].apply(lambda time: time.month)


    dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df['Day of Week'] = df['Day of Week'].map(dmap)

    # sb.countplot(data=df, x='Day of Week', hue='Reason', palette='viridis')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
    # plt.show()

    # sb.countplot(data=df, x='Month', hue='Reason', palette='viridis')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.5)
    # plt.show()

    month_df = df.groupby('Month').count()
    # month_df['Reason'].plot()
    # plt.show()
    month_df['Month_index'] = month_df.index
    # sb.lmplot(data=month_df, x='Month_index', y='twp')
    # plt.show()

    df['Date'] = df['timeStamp'].apply(lambda time: time.date())
    date_df = df.groupby('Date').count()
    # date_df['twp'].plot()
    # plt.tight_layout()
    # plt.show()

    # traffic_df = df[df['Reason'] == 'Traffic']
    # traffic_grp = traffic_df.groupby('Date').count()
    # traffic_grp['twp'].plot()
    # plt.tight_layout()
    # plt.show()

    # fire_df = df[df['Reason'] == 'Fire']
    # fire_grp = fire_df.groupby('Date').count()
    # fire_grp['twp'].plot()
    # plt.show()
    # print()

    new_df = df.groupby(['Day of Week', 'Hour']).count()#.unstack(level=-1)
    new_df = new_df['Reason'].unstack()
    # sb.heatmap(data=new_df)
    # sb.clustermap(data=new_df)
    # plt.show()

    new_df = df.groupby(['Day of Week', 'Month']).count()['Reason'].unstack()
    # sb.heatmap(data=new_df)
    sb.clustermap(data=new_df)
    plt.show()
    print(new_df)
    print('bye')


if __name__ == '__main__':
    exercises()