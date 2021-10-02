# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

import pandas as pd
import autosklearn.classification
from autosklearn.metrics import balanced_accuracy, precision, recall, f1

def import_data():
    x_train = pd.read_excel(r"..\data\processed\x_train.xlsx", index_col=0)
    x_val = pd.read_excel(r"..\data\processed\x_val.xlsx", index_col=0)
    y_train = pd.read_excel(r"..\data\processed\y_train.xlsx", index_col=0)
    y_val = pd.read_excel(r"..\data\processed\y_val.xlsx", index_col=0)
    test = pd.read_excel(r"..\data\processed\test.xlsx", index_col=0)
    # drop similar coloumns ('SibSp', 'Parch', 'FamilyS' engineered to 'FamilyS_g_Single',
    # 'FamilyS_g_OneFM', 'FamilyS_g_SmallF', 'FamilyS_g_MedF')
    x_train.drop(['SibSp', 'Parch', 'FamilyS'], axis=1, inplace=True)
    x_val.drop(['SibSp', 'Parch', 'FamilyS'], axis=1, inplace=True)
    return x_train, x_val, y_train, y_val, test

def model_selection(x_train, x_val, y_train, y_val):
    models = autosklearn.classification.AutoSklearnClassifier(
        include={'feature_preprocessors': ["no_preprocessing"]},
        exclude=None,
        resampling_strategy="cv",
        resampling_strategy_arguments = {"folds": 10},
        scoring_functions=[balanced_accuracy, precision, recall, f1],
        tmp_folder = r"..\data\autosklearn_classification_example_tmp", ensemble_size=0
    )
    models.fit(x_train, y_train, x_val, y_val)
    return models

if __name__ == '__main__':
    # Load the preprocessed data
    x_train, x_val, y_train, y_val, test = import_data()
    # Select the best model and hyper-parameters
    models = model_selection(x_train, x_val, y_train, y_val)