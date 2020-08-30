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
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def samples_1():
    # nltk.download_shell()
    # Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection
    # messages = [line.strip() for line in open('../../Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection')]
    # print(messages)
    # for mess_no, message in enumerate(messages[:10]):
        # print(mess_no, message)
    messages = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
    # print(messages)
    # print(messages.describe())
    # print(messages.groupby('label').describe())
    messages['length'] = messages['message'].apply(len)
    # messages['length'].plot.hist(bins=150)
    # print(messages['length'].describe())
    test = messages[messages['length'] == 910]['message'].iloc[0]
    # print(test)
    messages.hist(column='length', by='label', bins=60)





    # plt.legend()
    plt.show()
    print('bye')


def text_process(mess):
    """
    1. remove punc
    2. remove stopwords
    3. return list of clean words
    :param mess:
    :return:
    """
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_mess


def samples_2():
    # PART 2
    mess = 'sample !@# of punctuations! bye bye. Notice message it has punctuation~'
    # print(string.punctuation)
    # nopunc =[c for c in mess if c not in string.punctuation]
    # print(stopwords.words('english'))
    # nopunc = ''.join(nopunc)
    # clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    # print(clean_mess)
    # clean = text_process(mess)

    messages = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
    messages['length'] = messages['message'].apply(len)
    # test = messages['message'].head(5).apply(text_process)
    # print(test)
    messages = messages.iloc[:1000] # Make the message data lesser rows for faster processing.
    bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
    # print(len(bow_transformer.vocabulary_))
    mess4 = messages['message'][3]
    bow4 = bow_transformer.transform([mess4])
    print(bow4)
    print(bow4.shape)
    print(bow_transformer.get_feature_names()[1290])
    print('bye')


def samples_3():
    messages = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
    messages['length'] = messages['message'].apply(len)
    messages = messages.iloc[:1000] # Make the message data lesser rows for faster processing.
    bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
    mess4 = messages['message'][3]
    bow4 = bow_transformer.transform([mess4])

    messages_bow = bow_transformer.transform(messages['message'])
    # print(messages_bow.shape)
    # print(messages_bow.nnz)  #how many non-zero occurences
    sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
    # print('sparsity: {}'.format(sparsity))
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    tfidf4 = tfidf_transformer.transform(bow4)
    # print(tfidf4)
    # test = bow_transformer.vocabulary_
    # print(tfidf_transformer.idf_[bow_transformer.vocabulary_['Go']])  # 'university' works using complete messages dataframe.
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    # print(messages_tfidf)
    spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
    # print(spam_detect_model.predict(tfidf4))
    all_pred = spam_detect_model.predict(messages_tfidf)
    # print(all_pred)

    msgtrain, msgtest, labeltrain, labeltest = train_test_split(messages['message'], messages['label'], test_size=0.30)

    # THIS IS SKLEARN SHORTCUT THAT NULLIFIES FEATURES ABOVE FOR QUICKER IMPLEMENTATION FOR TRAIN AND TESTING
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
        # ('classifier', RandomForestClassifier())  # instead of MultinomialNB, you could run with RandomForest for different classifier.
    ])

    pipeline.fit(msgtrain, labeltrain)
    pred = pipeline.predict(msgtest)

    print(classification_report(y_true=labeltest, y_pred=pred))
    print(confusion_matrix(y_true=labeltest, y_pred=pred))

    print('bye')

def exercises():
    yelp = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv')
    print(yelp.head())
    # print(yelp.info())
    yelp['text length'] = yelp['text'].apply(len)
    # g = sns.FacetGrid(data=yelp, col='stars')
    # g.map(plt.hist, 'text length')
    # yelp.hist(column='text length', by='stars') # ugly
    # sns.boxplot(x='stars', y='text length', data=yelp)
    # sns.countplot(x='stars', data=yelp)

    yelp2 = yelp.groupby('stars').mean()
    # print(yelp2)

    yelp_corr = yelp2.corr()
    # print(yelp_corr)
    # sns.heatmap(data=yelp_corr,cmap='coolwarm',annot=True)

    yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
    print(yelp_class)

    x = yelp_class['text']
    y = yelp_class['stars']
    cv = CountVectorizer()
    x2 = cv.fit_transform( x)
    # print(x2.shape)
    print(len(y))
    # print(cv.get_feature_names()[26035])  # Need to create vocab first with transform.

    xtrain,xtest,ytrain,ytest = train_test_split(x2, y, test_size=0.30, random_state=101)
    nb = MultinomialNB()
    nb.fit(xtrain, ytrain)
    pred = nb.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))

    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    # REDO TRAIN TEST SPLIT
    x2 = yelp_class['text']
    y = yelp_class['stars']
    xtrain, xtest, ytrain, ytest = train_test_split(x2, y, test_size=0.30, random_state=101)
    pipeline.fit(xtrain,ytrain)
    pred = pipeline.predict(xtest)
    print(classification_report(y_true=ytest, y_pred=pred))
    print(confusion_matrix(y_true=ytest, y_pred=pred))




    plt.show()
    print('bye')

if __name__ == '__main__':
    # samples_1()
    # samples_2()
    # samples_3()
    exercises()

