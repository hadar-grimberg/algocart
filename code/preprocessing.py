# -*- coding: utf-8 -*-
"""
Hadar Grimberg
9/27/2021

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# change pandas  display setting: see all columns when printing and only 2 decimal
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = '{:,.2f}'.format


def load_dataset(path,name):
    # load data and make the PassengerId as the index of the DataFrame
    data=pd.read_csv(path, index_col=0)
    # convert the label and the sex to categorical and Pclass to ordered categorical
    if name == "train":
        data.Survived = data.Survived.astype('category')
    data.Sex = data.Sex.astype('category')
    data.Pclass = pd.Categorical(data.Pclass, categories=[3,2,1], ordered=True)
    # explore the columns and the head of the data
    print(f"The head 5 rows of {name} data:\n", data.head(5))
    print (data.info())
    print(data.describe().round(decimals=2))
    return data


def sum_missing_values(train,test):
    # Calculate the total missing values and the percentage of each feature for a table
    def calc_missing_table(df,set):
        total = df.isnull().sum()
        perc = (df.isnull().sum()/len(df)*100).map('{:,.2f}%'.format)
        return pd.concat([total, perc], axis=1, keys=[f'{set} Total',f'{set} percent'])
    # build missing data tables for each dataset's features
    train_missing = calc_missing_table(train.loc[:, train.columns != 'Survived'],"Train")
    test_missing = calc_missing_table(test,"Test")
    # Validate that we have no missing labels
    print(f"There are {train.Survived.isnull().sum()} missing labels")
    return pd.concat([train_missing, test_missing], axis=1).sort_values(by=['Train Total'])


def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2))
    ## creating a df with th
    total = pd.DataFrame(df.loc[:, feature].value_counts(dropna=False))
    ## concating percent and total dataframe

    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis=1)


# Outlier detection - visualization
def Box_plots(df,col):
    plt.figure(figsize=(10, 4))
    plt.title(f"Box Plot of {col}")
    sns.boxplot(data=df)
    plt.show()

# Outlier detection
def detect_outliers(df):
    """
    Takes a dataframe df of features and returns a list of the indices
    of observations that containing outliers according to the Tukey method.
    """
    outlier_indices = {}
    # iterate over features(columns)
    for col in df.columns:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col].dropna(), 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col].dropna(), 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index.to_list()

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices[col]=outlier_list_col
        if outlier_list_col:
            Box_plots(df[col],col)
    return outlier_indices

# print(train.isnull().sum())
# print(test.info())


if __name__ == '__main__':
    # Load train data and preliminary examination
    train = load_dataset(r"..\data\train.csv","train")
    """from first examination of the train set, one may see that there are many nulls within Age and Cabin.
    The mean age is 28 and 50% of the passengers are between 20 to 38 years old. At least 75% of the passengers
    hadn't parents or children on board and at least 50% of the passengers hadn't siblings/spouse on board.
    The mean fare was 32.2 which is ~6% of the maximum fare, less than 25% paid fare of above the average."""

    # Load test data and preliminary examination
    test = load_dataset(r"..\data\test.csv","test")
    """from first examination of the test set, one may see that there are many nulls within Age and Cabin as seen
     in train set. Mean age is 30.27, a little bit higher than in train set. At least 75% of the passengers
    hadn't parents or children on board and at least 50% of the passengers hadn't siblings/spouse on board, like 
    the train set. The mean fare was 35.63 which is ~7% of the maximum fare, less than 25% paid fare of above the average."""

    # handling the missing data
    missing_vals = sum_missing_values(train,test)
    """Age feature has about 180 nulls
    and Cabin feature has about 690 nulls, there ar no other nulls in the train set.
    The Name contains titles that indicate a certain age group. It might be use to fill the missing data"""


    # Join train and test datasets in order to clean both datasets at once and to obtain
    # the same number of features during categorical conversion
    dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


    # detect outliers for numerical features
    numeric_features=[col for col in dataset.select_dtypes(include=np.number).columns.tolist() if col not in ['PassengerId', 'Survived',]]
    # Outliers_to_drop = detect_outliers(train[numeric_features])
