# Confusion matrix

# from sklearn.[family] import [model]
from sklearn.linear_model import LinearRegression
import numpy as np
# from sklearn.cross_validation import train_test_split @ deprecated
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn import metrics

def samples_1():
    # model = LinearRegression(normalize=True)
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')
    cols = df.columns

    # sns.pairplot(df)
    # sns.distplot(df['Price'])

    # df_corr = df.corr()
    # sns.heatmap(df_corr, annot=True)

    x_cols = list(df.columns)
    X_df = df[x_cols].drop(axis=1, labels=['Price', 'Address'])
    y_df =df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.4, random_state=101)
    # X_test is test prices
    # y_test is the real prices
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    # print(lm.intercept_)
    # print(lm.coef_)
    cdf = pd.DataFrame(lm.coef_, X_df.columns, columns=['Coeff'])
    # print(cdf)
    # boston = load_boston() # example data that can be sued for above.

    # PART 2
    predictions = lm.predict(X_test)
    # print(predictions)
    # plt.scatter(y_test, predictions)
    # sns.distplot((y_test-predictions))

    mean_abs = metrics.mean_absolute_error(y_test, predictions)
    mean_sq = metrics.mean_squared_error(y_test, predictions)
    sqrt_mean = np.sqrt(mean_sq)
    print(sqrt_mean)


    plt.show()

    print()
    print('bye')


def exercises():
    customers = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers')
    # customers_desc = customers.describe()
    # print(customers_desc)
    # sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
    # sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, color='red', kind='hex')
    # sns.jointplot(x='Time on App', y='Time on App', data=customers, color='red', kind='hex')
    # sns.pairplot(data=customers)
    # sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

    X_data = customers.drop(axis=1, labels=['Address', 'Avatar', 'Yearly Amount Spent', 'Email'])
    y_data = customers['Yearly Amount Spent']

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=101)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    print(lm.coef_)
    prediction = lm.predict(X_test)
    # sns.scatterplot(x=prediction,y=y_test)
    mean_abs = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
    mean_sq = metrics.mean_squared_error(y_true=y_test, y_pred=prediction)
    mean_sqrt = np.sqrt(mean_sq)

    # sns.distplot(y_test-prediction)
    # plt.hist(y_test-prediction)
    pd.DataFrame()
    cdf = pd.DataFrame(data=lm.coef_, index=X_data.columns, columns=['Coeff'])
    print(cdf)


    plt.show()
    print('bye')


if __name__ == '__main__':
    # samples_1()
    exercises()