# from sklearn.[family] import [model]
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
# from sklearn.cross_validation import train_test_split @ deprecated
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn import metrics
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def samples_1():
    data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8,random_state=101)
    # print(data[0])
    # print(data[1])
    # plt.scatter(data[0][:,0],data[0][:,1],c=data[1])

    kmeans = KMeans(n_clusters=8)
    kmeans.fit(data[0])
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))

    ax1.set_title('k means')
    ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_)
    ax2.set_title('real means')
    ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1])






    plt.show()
    print('bye')

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


def exercises():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data', index_col=0)
    print(df.info())
    # sns.scatterplot(x='Grad.Rate', y='Room.Board', hue='Private', data=df)
    sns.set_style(style='whitegrid')
    # sns.lmplot(x='Grad.Rate', y='Room.Board', hue='Private', data=df, fit_reg=False)
    # sns.lmplot(x='Outstate', y='F.Undergrad', hue='Private', data=df, fit_reg=False)

    # g = sns.FacetGrid(data=df, hue='Private')#, col='Outstate')
    # g = g.map(plt.hist, 'Outstate', bins=30, alpha=0.7)

    # g = sns.FacetGrid(data=df,hue='Private')
    # g = g.map(plt.hist,'Grad.Rate')

    # print(df[df['Grad.Rate'] > 100])
    df['Grad.Rate']['Cazenovia College'] = 100
    # print(df[df['Grad.Rate'] > 100])

    x_data = df.drop(labels='Private', axis=1)
    y = df['Private']

    km = KMeans(n_clusters=2)
    km.fit(x_data)
    # print(km.cluster_centers_)
    # print(km.labels_)

    df['Cluster'] = df['Private'].apply(lambda x: int(x == 'Yes'))
    # df['Cluster'] = df['Private'].apply(converter)
    # print(df['Cluster'])
    pred = km.labels_
    y_real = df['Cluster']
    print(pred)
    print(y_real)
    print(classification_report(y_true=y_real, y_pred=pred))
    print(confusion_matrix(y_true=y_real, y_pred=pred))

    # fig, (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))
    # ax1.scatter()



    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    exercises()

