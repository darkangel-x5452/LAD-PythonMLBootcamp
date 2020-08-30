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
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def samples_1():
    columns_names = ['user_id', 'item_id','rating', 'timestamp']
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/u.data', sep='\t', names=columns_names)

    movie_titles = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/Movie_Id_Titles')
    df = pd.merge(df, movie_titles, on='item_id')
    # print(df.head())

    sns.set_style(style='white')
    new_df = df.groupby('title')['rating'].mean()
    new_df = new_df.sort_values(ascending=False).head()
    new_df = df.groupby('title')['rating'].count().sort_values(ascending=False)
    # print(new_df)
    ratings_ser = df.groupby('title')['rating'].mean()
    ratings = pd.DataFrame(ratings_ser)
    # print(ratings.head())
    ratings_count = df.groupby('title')['rating'].count()
    ratings['num of ratings'] = pd.DataFrame(ratings_count)
    # print(ratings.head())

    # ratings['rating'].plot.hist(bins=70)
    # sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

    moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
    # print(moviemat.columns)
    starwars_ratings = moviemat['Star Wars (1977)']
    liarliar_ratings = moviemat['Liar Liar (1997)']
    # print(starwars_ratings)
    similiar_starwars = moviemat.corrwith(starwars_ratings)
    similiar_liar = moviemat.corrwith(liarliar_ratings)

    corr_starwars  = pd.DataFrame(similiar_starwars, columns=['Correlation'])
    corr_starwars.dropna(inplace=True)
    # print(corr_starwars)

    # print(corr_starwars.sort_values('Correlation', ascending=False))
    corr_starwars = corr_starwars.join(ratings['num of ratings'])
    test = corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False)
    print(test)

    corr_liar = pd.DataFrame(similiar_liar, columns=['Correlate'])
    corr_liar.dropna(inplace=True)
    corr_liar = corr_liar.join(ratings['num of ratings'])
    test = corr_liar[corr_liar['num of ratings'] > 100].sort_values(by='Correlate', ascending=False)
    print(test)






    # plt.legend()
    plt.show()
    print('bye')


def exercises():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data', index_col=0)
    print(df.info())



    plt.show()
    print('bye')


if __name__ == '__main__':
    samples_1()
    # exercises()

