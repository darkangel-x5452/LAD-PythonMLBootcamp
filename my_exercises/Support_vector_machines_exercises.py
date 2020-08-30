# K Nearest Neighbours

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
# The Iris Setosa
from IPython.display import Image



def samples_1():
    cancer = load_breast_cancer()
    # print(cancer['DESCR'])

    data_keys = cancer.keys()
    # print(data_keys)
    data = cancer['data']
    cols = cancer['feature_names']
    df_cancer = pd.DataFrame(data=data,columns=cols)
    print(df_cancer.info())
    x = df_cancer
    y = cancer['target']
    # print(x)
    # print(y)
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=101)
    model = SVC()
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    param_grid = {'C': [0.1,1,10,100,1000], 'gamma': [1,0.1, 0.001,0.0001]}
    grid = GridSearchCV(SVC(),param_grid,verbose=3)
    grid.fit(xtrain,ytrain)
    # grid.best_params_
    # grid.best_estimator_
    grid_pred = grid.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=grid_pred))
    print(confusion_matrix(y_true=ytest, y_pred=grid_pred))
    plt.show()
    print('bye')


def exercises():
    #IMAGES
    # # The Iris Setosa
    # url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
    # Image(url, width=300, height=300)
    # # The Iris Versicolor
    # url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
    # Image(url, width=300, height=300)
    # # The Iris Virginica
    # url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
    # Image(url, width=300, height=300)

    iris = sns.load_dataset('iris')
    print(iris.info())
    # sns.pairplot(data=iris, hue='species')

    # sns.boxplot(x='sepal_width',y='sepal_length',hue='species', data=iris)
    setosa = iris[iris['species'] == 'setosa']
    # sns.jointplot(x='sepal_width',y='sepal_length', data=setosa, kind='kde')
    # sns.kdeplot(data=setosa['sepal_width'], data2=setosa['sepal_length'], shade=True)

    x = iris.drop(labels='species', axis=1)
    y = iris['species']
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=101)
    model = SVC()
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    params = {'C': [0.1,1,10,100], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
    grid = GridSearchCV(estimator=SVC(), param_grid=params, verbose=1)
    grid.fit(xtrain,ytrain)
    pred_grid = grid.predict(xtest)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print(classification_report(y_true=ytest, y_pred=pred_grid))
    print(confusion_matrix(y_true=ytest, y_pred=pred_grid))

    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    exercises()

