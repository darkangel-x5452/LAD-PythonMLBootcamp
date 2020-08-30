# Confusion matrix

import cufflinks as cf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from sklearn.[family] import [model]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.cross_validation import train_test_split @ deprecated
from sklearn.model_selection import train_test_split


# https://www.kaggle.com/
def samples_1():
    train = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')
    nulls = train.isnull()
    # print(nulls)
    # sns.heatmap(data=nulls, yticklabels=False, cbar=False,cmap='viridis')
    sns.set_style('whitegrid')
    # sns.countplot(x='Survived', hue='Sex', data=train)
    # sns.countplot(x='Survived', hue='Pclass', data=train)
    # sns.distplot(train['Age'].dropna(), kde=False, bins=30)
    # train['Age'].plot.hist()
    # sns.countplot(x='SibSp', data=train)
    # train['Fare'].hist(bins=40, figsize=(10,4))

    cf.go_offline()
    train['Fare'].iplot(kind='hist', bins=30)



    plt.show()
    print('bye')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


def samples_2():
    train = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')
    nulls = train.isnull()
    # print(nulls)
    # sns.heatmap(data=nulls, yticklabels=False, cbar=False,cmap='viridis')
    # plt.figure(figsize=(10,7))
    # sns.boxplot(x='Pclass', y='Age', data=train)

    train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
    # print(train['Age'])
    # nulls = train.isnull()
    # sns.heatmap(nulls, yticklabels=False, cbar=False, cmap='viridis')
    train.drop('Cabin', axis=1, inplace=True)
    # nulls = train.isnull()
    # sns.heatmap(nulls, yticklabels=False, cbar=False, cmap='viridis')
    train.dropna(inplace=True)
    # nulls = train.isnull()
    # sns.heatmap(nulls, yticklabels=False, cbar=False, cmap='viridis')

    sex = pd.get_dummies(train['Sex'], drop_first=True)
    embark = pd.get_dummies(train['Embarked'], drop_first=True)
    train = pd.concat([train, sex, embark], axis=1)
    train.drop(inplace=True, labels=['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1)
    print(train.head())
    X = train.drop(labels=['Survived'], axis=1)
    y = train['Survived']
    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.30, random_state=101)
    logmodel = LogisticRegression(max_iter=100000)
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_true=y_test, y_pred=predictions))

    plt.show()
    print('bye')


def exercises():
    ad_data = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv')
    print(ad_data.head())
    print(ad_data.info())
    print(ad_data.describe())

    # sns.distplot(a=ad_data['Age'], kde=False, bins=30)
    # ad_data['Age'].hist(bins=30)
    # sns.jointplot(x='Age',y='Area Income', data=ad_data)
    # sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde')
    # sns.jointplot(x='Daily Internet Usage', y='Daily Time Spent on Site', data=ad_data)
    # sns.pairplot(data=ad_data,hue='Clicked on Ad')
    x_set = ad_data.drop(labels=['Clicked on Ad', 'City', 'Ad Topic Line', 'Country','Timestamp'], axis=1)
    print(x_set.head())
    y_set = ad_data['Clicked on Ad']
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.30, random_state=101)
    logmodel = LogisticRegression(max_iter=1000)
    logmodel.fit(x_train, y_train)
    prediction = logmodel.predict(x_test)
    classification = classification_report(y_true=y_test, y_pred=prediction)
    print(classification)
    confusion = confusion_matrix(y_true=y_test, y_pred=prediction)
    print(confusion)


    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    # samples_2()
    exercises()