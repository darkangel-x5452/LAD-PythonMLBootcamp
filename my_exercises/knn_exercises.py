# K Nearest Neighbours

import matplotlib.pyplot as plt
# from sklearn.[family] import [model]
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.cross_validation import train_test_split @ deprecated
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def samples_1():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/Classified Data')
    print(df.info())
    # print(df.describe())

    scaler = StandardScaler()
    new_df = df.drop(labels=['TARGET CLASS'], axis=1)
    scaler.fit(new_df)
    scaled_features = scaler.transform(new_df)
    # print(scaled_features)
    df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    # print(df_feat.head())
    y = df['TARGET CLASS']

    x_train, x_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.30, random_state=101)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    # print(pred)

    class_rep = classification_report(y_true=y_test, y_pred=pred)
    print(class_rep)
    confuse = confusion_matrix(y_true=y_test, y_pred=pred)
    print(confuse)

    # SELECT A LOWER ERROR RATE
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        # print('pred_i', type(pred_i))
        # print('y_test', type(y_test))
        # print('pred=test', type(pred_i != y_test))
        # print('np mean', np.mean(pred_i != y_test))
        error_rate.append(np.mean(pred_i != y_test))
    # print(error_rate)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    # print(pred)
    class_rep = classification_report(y_true=y_test, y_pred=pred)
    print(class_rep)
    confuse = confusion_matrix(y_true=y_test, y_pred=pred)
    print(confuse)

    plt.show()
    print('bye')


def exercises():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data')
    print(df.info())
    # sns.pairplot(data=df, hue='TARGET CLASS')
    scaler = StandardScaler()
    new_df = df.drop(labels=['TARGET CLASS'], axis=1)
    scaler.fit(new_df)
    scaled = scaler.transform(new_df)
    y = df['TARGET CLASS']
    xtrain, xtest, ytrain, ytest = train_test_split(scaled, y)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(xtrain, ytrain)
    pred = knn.predict(xtest)

    class_rep = classification_report(y_true=ytest, y_pred=pred)
    confuse = confusion_matrix(y_true=ytest, y_pred=pred)
    print(class_rep, '\n', confuse)
    error_list = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xtrain, ytrain)
        pred_i = knn.predict(xtest)
        compare = pred_i != ytest
        error_val = np.mean(compare)
        error_list.append(error_val)

    plt.plot(range(1, 40), error_list)

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(xtrain, ytrain)
    pred = knn.predict(xtest)

    class_rep = classification_report(y_true=ytest, y_pred=pred)
    confuse = confusion_matrix(y_true=ytest, y_pred=pred)
    print(class_rep, '\n', confuse)

    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    exercises()
