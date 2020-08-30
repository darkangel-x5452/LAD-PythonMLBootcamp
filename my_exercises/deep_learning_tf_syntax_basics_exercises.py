# import plotly.plotly as plt
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


def tf_syntax_basics_1():
    df = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/fake_reg.csv')
    print(df.head())
    print(df.info())
    # sns.pairplot(data=df)

    X = df.drop(labels='price', axis=1).values
    y = df['price'].values
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)

    # print(help(MinMaxScaler))
    scaler = MinMaxScaler()
    scaler.fit(xtrain)  # dont need it for y
    xtrain2 = scaler.transform(xtrain)
    xtest2 = scaler.transform(xtest)
    # print(xtrain2.max())
    # print(xtrain2.min())

    ### PART 2
    # print(help(Sequential))
    # print(help(Dense))
    # Model method 1
    # model = Sequential([Dense(units=4, activation='relu'),
    #                     Dense(units=2,activation='relu'),
    #                     Dense(units=1)])
    # Model method 2 (preferred method)
    model = Sequential()
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=4, activation='relu'))

    model.add(Dense(
        units=1))  # Final output. this us very import. For single result case, e.g. output a price result, the only need one unit (neuron)

    model.compile(optimizer='rmsprop',
                  loss='mse')

    # model.fit(x=xtrain2, y=ytrain, epochs=250, verbose=1)
    model.fit(x=xtrain2, y=ytrain, epochs=250, verbose=0)
    # print(model.history.history)
    model_hist = pd.DataFrame(model.history.history)
    # print(model_hist)
    # model_hist.plot()

    ### PART 3
    model_eval = model.evaluate(xtest2, ytest, verbose=0)
    # print(model_eval)
    model_eval = model.evaluate(xtrain2, ytrain, verbose=0)
    # print(model_eval)

    test_pred = model.predict(xtest2)
    test_pred = pd.Series(test_pred.reshape(300, ))  # related to number of test items.
    # print(test_pred)
    pred_df = pd.DataFrame(ytest, columns=['test true y'])
    # print(pred_df)
    pred_df = pd.concat([pred_df, test_pred], axis=1)
    pred_df.columns = ['test true y', 'model pred']
    # print(pred_df)

    sns.scatterplot(x='test true y', y='model pred', data=pred_df)
    meanabserr = mean_absolute_error(pred_df['test true y'], pred_df['model pred'])
    # print(meanabserr) # this is good depending on your data. Check your original data to see if price of $4 is big. e.g below
    # print(df.describe())  # check the original data to compare against error

    meansqerr = mean_squared_error(y_true=pred_df['test true y'], y_pred=pred_df['model pred'])
    rtmeansqerr = meansqerr ** 0.5  # this is to get the actually error by rooting the square.
    # print(rtmeansqerr)

    # example new item to price
    new_gem = [[998, 1000]]
    new_gem_scaled = scaler.transform(
        new_gem)  # remember the model is using scaled features. so new data also has to be scaled.
    new_gem_pred = model.predict(new_gem_scaled)
    # print(new_gem_pred)

    # IMPORTANT SAVE THE MODEL
    model.save('my_gem_model.h5')

    later_model = load_model('my_gem_model.h5')  # to load the model
    # print(later_model.predict(new_gem_scaled))

    plt.show()
    print('bye')


def tf_regression_1():
    df = pd.read_csv(
        '../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/kc_house_data.csv')
    # print(df.head())
    # print(df.info())

    # print(df.isnull().sum()) # Check if there is missing data.
    # print(df.describe().transpose())  # statistical analysis and transpose to make it easier to read.
    # sns.distplot(df['price'])
    # sns.countplot(df['bedrooms'])
    # print(df.corr())
    # print(df.corr()['price'])

    # sns.scatterplot(x='price', y='sqft_living', data=df)
    # sns.boxplot(x='bedrooms', y='price', data=df)
    # sns.scatterplot(x='price', y='long', data=df)
    # sns.scatterplot(x='price', y='lat', data=df)
    # sns.scatterplot(x='long', y='lat', data=df, hue='price')
    # print(df.sort_values(by='price', ascending=False).head(20))
    df_copy = df.copy()
    top_1_perc = round(len(df_copy) * 0.01)
    # print(len(df)*0.01) # 1% of houses count
    non_top_1_perc = df.sort_values(by='price', ascending=False).iloc[top_1_perc:]
    # sns.scatterplot(x='long', y='lat', data=non_top_1_perc, hue='price', edgecolor=None, alpha=0.2, palette='RdYlGn')
    # sns.boxplot(x='waterfront', y='price', data=df)

    df = df.drop(labels=['id'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    # print(df.info())
    df['year'] = df['date'].apply(lambda date: date.year)
    df['month'] = df['date'].apply(lambda date: date.month)
    # print(df.info())
    # sns.boxplot(x='month', y='price', data=df)
    month_grp = df.groupby(by='month').mean()['price']
    # print(month_grp)
    # month_grp.plot()
    df = df.drop(labels='date', axis=1)
    # print(df['zipcode'].value_counts())  # see if zipcodes is useful
    df = df.drop(labels='zipcode', axis=1)  # drop for now but might be worth doing later with more experience.
    # print(df['yr_renovated'].value_counts())  # see if renovated year is useful
    # print(df['sqft_basement'].value_counts())  # see if valuable.

    X = df.drop('price', axis=1).values  # returns numpy array.
    y = df['price'].values
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=101)
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)  # save time by doign fitting and tansforming in same step rather than 2.
    xtest = scaler.transform(xtest)  # dot do fit, cause we dont want data leak or assume higher form of our test set.
    model = Sequential()
    # print(xtrain.shape)  # look at the shape to assume amount of neurons. if there is 19 features, use 19 neurons.
    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))

    model.add(Dense(1, activation='relu'))  # one neuron for one output.
    model.compile(optimizer='adam', loss='mse')
    # model.fit(xtrain,ytrain, validation_data=(xtest, ytest), batch_size=128, epochs=400, verbose=0) # validation data validates the data and checks error rate, batch size breaks up the training data if it is large (do powers of 2s)
    model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=128, epochs=200, verbose=0)
    # print(model.history.history)
    losses = pd.DataFrame(model.history.history)
    # losses.plot()

    pred = model.predict(xtest)
    meansqerr = mean_squared_error(y_true=ytest, y_pred=pred)
    meanabserr = mean_absolute_error(y_true=ytest, y_pred=pred)
    sqrtmeansqerr = np.sqrt(meansqerr)
    # print(meansqerr, meanabserr, sqrtmeansqerr)
    # print(df['price'])
    # print(explained_variance_score(ytest,pred))
    # plt.scatter(ytest,pred)
    # plt.plot(ytest,ytest,'r')

    single_house = df.drop(labels='price', axis=1).iloc[0]
    print(single_house.shape)
    single_house = single_house.values.reshape(-1, 19)
    # print(single_house.values.shape)
    print(single_house.shape)
    single_house = scaler.transform(single_house)
    pred_single = model.predict(single_house)
    print(pred_single)

    plt.show()
    print('bye')


def tf_classification_1():
    df = pd.read_csv(
        '../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/cancer_classification.csv')
    print(df.info())
    # print(df.describe().transpose)

    # sns.countplot(x='benign_0__mal_1', data=df)
    initial_corr = df.corr()['benign_0__mal_1'].sort_values()
    # print(initial_corr)
    # initial_corr[:-1].plot(kind='bar')
    # sns.heatmap(df.corr())

    x = df.drop(labels='benign_0__mal_1', axis=1).values
    y = df['benign_0__mal_1'].values

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=101)
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    # Processing model stage (without error checking and stop loss)
    # model = Sequential()
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(15, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))  # because it is binary classification
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.fit(x=xtrain, y=ytrain, epochs=600, verbose=0, validation_data=(xtest,ytest))
    # err = pd.DataFrame(model.history.history)
    # err.plot()

    # Processing model stage (with overfitting checking and reduce loss)
    # model = Sequential()
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(15, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))  # because it is binary classification
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    # # print(help(EarlyStopping))
    # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=25)
    # model.fit(x=xtrain, y=ytrain, epochs=600, verbose=0, validation_data=(xtest,ytest), callbacks=[early_stop])
    # err = pd.DataFrame(model.history.history)
    # err.plot()

    # Processing model stage (with dropping neurons, drop out layer)
    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))  # Dropping
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))  # Dropping

    model.add(Dense(1, activation='sigmoid'))  # because it is binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # print(help(EarlyStopping))
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    model.fit(x=xtrain, y=ytrain, epochs=600, verbose=0, validation_data=(xtest, ytest), callbacks=[early_stop])
    err = pd.DataFrame(model.history.history)
    # err.plot()

    pred = model.predict_classes(xtest)
    print(pred)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    plt.show()
    print('bye')


def feat_info(col_name):
    data_info = pd.read_csv(
        '../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/lending_club_info.csv',
        index_col='LoanStatNew')
    return data_info.loc[col_name]['Description']


def mort_acc_mean(tot_acc, mort_acc, mort_acc_ass):
    if np.isnan(mort_acc):
        return mort_acc_ass[tot_acc]
    else:
        return mort_acc


def tf_project_1():
    df = pd.read_csv(
        '../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/lending_club_loan_two.csv')
    # print(df['loan_status'].head())
    # print(df.info())
    # sns.countplot(x='loan_status', data=df)
    # df['loan_amnt'].plot.hist(bins=60)
    df_corr = df.corr()
    # print(df_corr)
    # sns.heatmap(data=df_corr)
    # sns.scatterplot(x='installment', y='loan_amnt', data=df)
    # sns.boxplot(x='loan_status', y='loan_amnt', data=df)
    # loan_status_grp = df.groupby('loan_status')['loan_amnt'].describe()
    # print(loan_status_grp)
    grade_values = df['grade'].unique()
    subgrade_values = df['sub_grade'].unique()
    # print(grade_values)
    # print(subgrade_values)
    # sns.countplot(x='grade',hue='loan_status', data=df)
    subgrade_values.sort()
    # sns.countplot(x='sub_grade', data=df, order=subgrade_values, palette='coolwarm')
    # sns.countplot(x='sub_grade', data=df, order=subgrade_values, palette='coolwarm', hue='loan_status')

    f_g_grades = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
    subgrade_values = f_g_grades['sub_grade'].unique()
    subgrade_values.sort()
    # sns.countplot(x='sub_grade', data=f_g_grades, order=subgrade_values, palette='coolwarm', hue='loan_status')

    # df['loan_repaid'] = df['loan_status'].apply(lambda x: int(x == 'Fully Paid'))
    df['loan_repaid'] = df['loan_status'].map({"Fully Paid": 1, "Charged Off": 0})
    # print(df['loan_repaid'])
    df_corr = df.corr()['loan_repaid']
    df_corr = df_corr.sort_values().drop(labels='loan_repaid', axis=0)
    # print(df_corr)
    # df_corr.plot(kind='bar')

    ### SECTION 2 DATA PREPROCESSING
    # print(len(df))
    # df_len = len(df)
    # print(df.isnull().sum() / len(df) * 100)

    # emp_title = df['emp_title'].unique()
    # print(len(emp_title))
    df.drop(labels='emp_title', axis=1, inplace=True)
    emp_length = df['emp_length']
    # print(type(emp_length[0]))
    emp_length_sort = sorted(emp_length.dropna().unique())
    # print(emp_length_sort)
    # sns.countplot(x='emp_length', data=df, order=emp_length_sort)
    # sns.countplot(x='emp_length', data=df, order=emp_length_sort, hue='loan_status')
    emp_co = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
    emp_fp = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
    # print(emp_co)
    # print(emp_fp)
    emp_len_stat = emp_co / emp_fp
    # emp_len_stat.plot(kind='bar')
    df.drop(labels='emp_length', axis=1, inplace=True)
    df.drop(labels='title', axis=1, inplace=True)

    # print(df['mort_acc'].value_counts())
    # print(df.corr()['mort_acc'])
    tot_acc_grp = df.groupby('total_acc').mean()['mort_acc']
    df['mort_acc'] = df.apply(lambda x: mort_acc_mean(x['total_acc'], x['mort_acc'], tot_acc_grp), axis=1)
    # print(df['mort_acc'].isnull().sum())
    df.dropna(inplace=True)
    # print(df.isnull().sum())

    ### CATEGORY VARIABLES AND DUMMY VARIABLES
    df_str_type = list(df.select_dtypes(exclude=['float64', 'int64']).columns)
    # print(df['term'].unique())
    df['term'] = df['term'].map({' 36 months': 36, ' 60 months': 60})

    df.drop(labels='grade', axis=1, inplace=True)
    df_subgrade_dum = pd.get_dummies(data=df[['sub_grade','verification_status','application_type','initial_list_status','purpose']], drop_first=True)
    df = pd.concat([df.drop(labels=['sub_grade','verification_status','application_type','initial_list_status','purpose'], axis=1), df_subgrade_dum], axis=1)

    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
    df_subgrade_dum = pd.get_dummies(data=df[['home_ownership']], drop_first=True)
    df = pd.concat([df.drop(labels=['home_ownership'], axis=1), df_subgrade_dum], axis=1)

    df['zip_code'] = df['address'].apply(lambda x: x.split()[-1])
    df_subgrade_dum = pd.get_dummies(data=df[['zip_code']], drop_first=True)
    df = pd.concat([df.drop(labels=['zip_code', 'address'], axis=1), df_subgrade_dum], axis=1)

    df.drop(labels=['issue_d'], axis=1, inplace=True)

    df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x.split('-')[-1]))
    df.drop(labels=['earliest_cr_line'], axis=1, inplace=True)




    ### TRAIN TEST MODEL
    df.drop(labels=['loan_status'], axis=1, inplace=True)
    X = df.drop(labels=['loan_repaid'], axis=1).values
    y = df['loan_repaid'].values

    # df = df.sample(frac=0.1,random_state=101)
    print(len(df))

    xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size=0.2, random_state=101)
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    model = Sequential()
    model.add(Dense(78, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(39, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(19, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(x=xtrain, y=ytrain, batch_size=256, epochs=25, verbose=0, validation_data=(xtest, ytest))

    model.save('loan_exercise_model.h5')
    losses = pd.DataFrame(model.history.history)
    # losses[['loss','val_loss']].plot()
    losses.plot()

    pred_class = model.predict_classes(xtest)
    print(confusion_matrix(y_true=ytest,y_pred=pred_class))
    print(classification_report(y_true=ytest, y_pred=pred_class))

    random.seed(101)
    random_ind = random.randint(0, len(df))
    new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]

    cust_pred = model.predict_classes(new_customer.values.reshape(1,78))
    print(cust_pred)
    print(df.iloc[random_ind]['loan_repaid'])


    plt.show()
    print('bye')


if __name__ == '__main__':
    # tf_syntax_basics_1()
    # tf_regression_1()
    # tf_classification_1()
    tf_project_1()
