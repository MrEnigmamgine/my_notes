import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime
from pydataset import data

# SKLearn
import sklearn as sk
import sklearn.model_selection as skm


def get_db_url(database, hostname='', username='', password='', env=''):
    '''Creates a URL for a specific database and credential set to be used with pymysql.

    Can be used either with a set of credentials passed directly to the function or with an environment file containing the credentials.
    If both are provided, the environment file takes precedence.

    Parameters:
    database (str): The target database that pymysql will connect to, which will provide context for any SQL queries used in the connection.
    hostname (str): The DNS hostname or IP-Adress for the connection
    username (str), password (str): User credentials that will be used in a sql connection.
    env (str): Relative path to an environment file.  The file must include the hostname, username, and password variables.

    Returns:
    str: Full URL for use with a pymysql connection
    '''
    if env != '':
        d = {}
        file = open(env)
        for line in file:
            (key, value) = line.split('=')
            d[key] = value.replace('\n', '').replace("'",'').replace('"','')
        username = d['username']
        hostname = d['hostname']
        password = d['password']
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

# Easily load a google sheet (first tab only)
def read_google(url):
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)


# Return a float from a string representing a usd amount.
def unstring_usd(str):
    return float(str.replace('$','').replace(',',''))

# Return a USD formatted string from a float.
def as_currency(amount):
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)


# Prettify chi^2 test

def chi2_test(df, alpha=0.05):
    chi2, p, degf, expected = stats.chi2_contingency(df)
    print('Observed\n')
    print(df.values)
    print('---\nExpected\n')
    print(expected.astype(int))
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'degf = {degf}')
    print(f'p     = {p:.4f}')
    print('---\n')
    if p < alpha:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")


## Generic split data function
def train_validate_test_split(df, seed=123, stratify=None):
    # First split off our testing data.
    train_and_validate, test = skm.train_test_split(
        df, 
        test_size=0.2, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    train, validate = skm.train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify= (train_and_validate[stratify] if stratify else None)
    )
    return train, validate, test