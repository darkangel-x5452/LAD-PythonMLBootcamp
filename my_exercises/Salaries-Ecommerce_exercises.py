import numpy as np
import pandas as pd

def find_chief(job):
    if 'chief' in job.lower():
        return True
    else:
        return False

def salaries():
    sal = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/04-Pandas-Exercises/Salaries.csv')
    # new_df = sal['JobTitle'].value_counts().head(5)
    # new_df = sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
    # new_df = sum(sal['JobTitle'].apply(lambda x: find_chief(x)))
    sal['title_len'] = sal['JobTitle'].apply(len)
    new_df = sal[['title_len', 'JobTitle']].corr()
    print(new_df)
    print('bye')

def email_type(x):
    email = x.split('@')[1]
def ecommerce():
    ecom = pd.read_csv('../../Refactored_Py_DS_ML_Bootcamp-master/04-Pandas-Exercises/Ecommerce Purchases')
    # new_df = ecom.info()
    # new_df = ecom['Purchase Price'].mean()
    # new_df = ecom['Purchase Price'].max()
    # new_df = ecom['Purchase Price'].min()
    # new_df = ecom[ecom['Language'] == 'en'].info()
    # new_df = ecom[ecom['Job'] == 'Lawyer'].info()
    # new_df = ecom['AM or PM'].value_counts()
    # new_df = ecom['Job'].value_counts().head(5)
    # new_df = ecom[(ecom['CC Provider'] == 'American Express') & (ecom['Purchase Price'] > 95)].count()
    # new_df = sum(ecom['CC Exp Date'].apply(lambda x: x[3:] == '25'))
    new_df = ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5)
    print(new_df)
    print('bye')


if __name__ == '__main__':
    # salaries()
    ecommerce()