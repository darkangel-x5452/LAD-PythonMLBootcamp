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
    cancer = load_breast_cancer()
    cancer_keys = cancer.keys()
    cancer_descr = cancer['DESCR']
    data = cancer['data']
    features = cancer['feature_names']
    df = pd.DataFrame(data=data, columns=features)

    scaler = StandardScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)
    # scaled.shape()
    # PCA
    pca = PCA(n_components=2)
    pca.fit(scaled)
    x_pca = pca.transform(scaled)
    # x_pca.shape()
    print(x_pca)
    # plt.scatter(x=x_pca[:,0],y=x_pca[:,1], c=cancer['target'])
    print(pca.components_)

    df_comp = pd.DataFrame(pca.components_, columns=features)
    print(df_comp)
    sns.heatmap(data=df_comp)




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

