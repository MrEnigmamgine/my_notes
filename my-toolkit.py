# Core modules
import os
import math

# DS Modules
import numpy as np
import pandas as pd
from pydataset import data

# Visualization modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import datetime

# SKLearn
import sklearn as sk
from sklearn.model_selection import train_test_split

# Pandas Options
pd.options.display.max_columns = None
pd.options.display.max_rows = 70
pd.options.display.float_format = '{:20,.2f}'.format

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

SEED = 8
FEATURES = []
TARGETS = []

#####################################################
#               END CONFIG VARIABLES                #
#####################################################

def year_to_decade_name(year: int) -> str:
    """Convert a year int into a categorical str"""
    decade = math.floor(int(year) / 10) * 10
    return f"{decade}s"

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
def train_validate_test_split(df, seed=SEED, stratify=None):
    """Splits data 60%/20%/20%"""
    # First split off our testing data.
    train, test_validate = train_test_split(
        df, 
        test_size=3/5, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    test, validate = train_test_split(
        test_validate,
        test_size=1/2,
        random_state=seed,
        stratify= (test_validate[stratify] if stratify else None)
    )
    return train, test, validate

def dropna_df(df):
    """Returns a dataframe free of null values where the columns have the proper dtypes"""
    df = df.dropna()
    df = df.convert_dtypes()
    # convert_dtypes() chooses some slightly wonky data types that cause problems later.
    # Fix the wonk by creating a new dataframe from the dataframe.
    fix = pd.DataFrame(df.to_dict()) 
    return fix

def get_highcounts(df):
    """Returns a dataframe containing the 4 highest value counts for each column.
    Or in the case of continuous variables, the counts of 4 bins."""
    categorical_types =['object','string','bool','category'] # The dtypes we will treat as categorical. Might not be a complete list.
    d = {} # The dictionary we will build
    # Loop through each column
    for col in df:
        # and get the highest 4 (value, count) tuples using .head() and .iteritems()
        if df[col].dtype in categorical_types:
            d[col] = (  list(df[col].value_counts(dropna = False).head(4).iteritems()) )
        # Make sure there are more than 4 values before we try binning
        elif df[col].nunique() > 4:
            d[col] = (  list(df[col].value_counts(bins = 4, dropna=False).iteritems()) )
        # And then get the rest.
        else:
            d[col] = (  list(df[col].value_counts(dropna = False).head(4).iteritems()) )

    # Build the dataframe using from_dict and orient="index"
    outdf = pd.DataFrame.from_dict(d, orient='index')
    # Rename the columns for ease of access
    outdf.columns = ['highcount_'+str(col) for col in outdf]
    return outdf

def col_summary(df):
    """Returns a dataframe full of statistics about each column in a dataframe.
    Useful for scrubbing datatypes and handling nulls."""
    # Build the datatype column
    dt = pd.DataFrame(df.dtypes)
    dt.columns = ['dtype']
    # Count of nulls column
    ns = pd.DataFrame(df.isna().sum())
    ns.columns = ['null_sum']
    # Percentage of nulls column
    nm = pd.DataFrame(df.isna().mean())
    nm.columns = ['null_mean']
    # Count of unique values
    nu = pd.DataFrame(df.nunique())
    nu.columns = ['n_unique']
    # Count of possible hidden nulls
    d = {}
    for col in df:
        d[col] = df[col].apply(lambda x: str(x).strip() == '' or str(x).strip().lower() in ['na','n/a']).sum()
    hnulls = pd.DataFrame.from_dict(d, orient='index')
    hnulls.columns = ['hidden_nulls']


    out = pd.concat([dt, ns, nm, hnulls, nu], axis=1)
    out['duplicates'] = len(df) - out['n_unique']

    return out

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})
    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True)[['parcelid', 'num_cols_missing', 'percent_cols_missing']]
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def get_gotchas(df):
    out = {
        'possible_ids': [],
        'possible_bools': []
        }
    summary = col_summary(df)
    for name, row in summary.iterrows():
        if len(df) - row.loc['n_unique'] <= 1:
            out['possible_ids'].append(name)
        if row.loc['n_unique'] in [1,2]:
            out['possible_bools'].append(name)

    return out

def all_the_hist(df):
    import math
    vizcols = 5
    vizrows = math.ceil(len(df.columns)/vizcols)
    out = []
    fig, ax = plt.subplots(vizrows, vizcols, figsize=(15,vizrows*4))
    for i, col in enumerate(df):
        r, c = i % vizrows, i % vizcols
        a = ax[r][c]
        a.set_title(col)
        df[col].hist(ax=a)
    fig
