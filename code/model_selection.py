# -*- coding: utf-8 -*-
"""
Hadar Grimberg
10/2/2021

"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


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
    # finding the best estimator from 14 classifiers
    random_state = 17

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "Extra Trees", "Gradient Boosting", "Logistic Regression", "LDA", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=random_state),
        SVC(gamma=2, C=1, random_state=random_state),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
        DecisionTreeClassifier(max_depth=5, random_state=random_state),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,random_state=random_state),
        MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        GaussianNB(),
        ExtraTreesClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]
    # Same classifiers but without params for future usage
    classifiersNP = [
        KNeighborsClassifier(),
        SVC(kernel="linear", random_state=random_state),
        SVC(random_state=random_state),
        GaussianProcessClassifier(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        RandomForestClassifier(random_state=random_state),
        MLPClassifier(random_state=random_state),
        AdaBoostClassifier(random_state=random_state),
        GaussianNB(),
        ExtraTreesClassifier(random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        LogisticRegression(random_state=random_state),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=10)

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_val_score(classifier, x_train, y=y_train, scoring="accuracy", cv=kfold))

    # Mean the results of each model 10-k folds
    CVmeans = []
    CVstd = []
    for cv_result in cv_results:
        CVmeans.append(cv_result.mean())
        CVstd.append(cv_result.std())
    #sort and show results
    CVtable = pd.DataFrame({ "Algorithm": names, "CrossValMeans": CVmeans, "CrossValerrors": CVstd}).sort_values(by=['CrossValMeans'],ascending=False)
    print(CVtable)
    # 6 models had accurarcy of 0.8 or above. The models will take forward for hyper-parameters tuning
    classifiersTuning = [classifiersNP[cls] for cls in CVtable.index[CVtable.CrossValMeans > 0.8].to_list()]

    return classifiersTuning

if __name__ == '__main__':
    # Load the preprocessed data
    x_train, x_val, y_train, y_val, test = import_data()
    # Select the best model and hyper-parameters
    classifiersTuning = model_selection(x_train, x_val, y_train, y_val)