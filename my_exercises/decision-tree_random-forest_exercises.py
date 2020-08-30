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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def samples_1():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/kyphosis.csv')
    print(df.info())
    # sns.pairplot(data=df, hue='Kyphosis')
    # DECISION TREE METHOD
    x = df.drop(labels='Kyphosis', axis=1)
    y = df['Kyphosis']
    xtrain,xtest,ytrain,ytest=train_test_split(x, y, test_size=0.30, random_state=42)

    dtree = DecisionTreeClassifier()
    dtree.fit(xtrain,ytrain)

    pred = dtree.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    # RANDOM FORREST METHOD (SHOULD BE BETTER THAN DTREE)
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(xtrain,ytrain)
    rfc_pred = rfc.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=rfc_pred))
    print(confusion_matrix(y_true=ytest, y_pred=rfc_pred))

    #TREE VIZ
    # in notes, but difficult to implement cause it requires graph_viz to be downloaded separately as it not automatic in library.



    plt.show()
    print('bye')


def exercises():
    #loan_data.csv
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv')
    print(df.info())
    df_cred_0 = df[df['credit.policy'] == 0]
    df_cred_1 = df[df['credit.policy'] == 1]
    # df_cred_0['fico'].plot.hist(bins=30,alpha=0.5, label='credit 0')
    # df_cred_1['fico'].plot.hist(bins=30, alpha=0.5, label='credit 1')


    df_cred_0 = df[df['not.fully.paid'] == 0]
    df_cred_1 = df[df['not.fully.paid'] == 1]
    # df_cred_0['fico'].plot.hist(bins=30, alpha=0.5, label='not.fully.paid 0')
    # df_cred_1['fico'].plot.hist(bins=30, alpha=0.5, label='not.fully.paid 1')

    # sns.countplot(x='purpose', hue='not.fully.paid', data=df)

    # sns.jointplot(x='fico',y='int.rate', data=df)

    # sns.lmplot(x='fico', y='int.rate', data=df, hue='credit.policy', col='not.fully.paid')

    cat_feats = ['purpose']
    final_data = pd.get_dummies(data=df, columns=cat_feats, drop_first=True)
    print(final_data.info())
    x = final_data.drop(labels='not.fully.paid', axis=1)
    y = final_data['not.fully.paid']
    xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.30, random_state=101)
    dtree = DecisionTreeClassifier()
    dtree.fit(xtrain,ytrain)
    pred = dtree.predict(xtest)
    print(classification_report(y_true=ytest,y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(xtrain, ytrain)
    pred_rfc = rfc.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred_rfc))
    print(confusion_matrix(y_true=ytest, y_pred=pred_rfc))

    # plt.legend()
    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    exercises()
