import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.feature_selection import RFE


#################################### Data Preprocessing ####################################

# display setting
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 6)

# import data from income.csv via pandas
data = pd.read_csv("dataset/income.csv", header=0)
data = data.dropna()
# print(data)

#drop data if contains ' ?'
data = data[~data.eq(' ?').any(1)]
# print(data)

# handling imbalanced data via downsampling
# print(data['y'].value_counts())
data_majority = data[data.y==0]
data_minority = data[data.y==1]
maj_len = len(data[data['y']==0])


df_minority_upsampled = resample(data_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=maj_len)     # to match minority class
 
# Combine minority class with downsampled majority class and shuffle
data_upsampled = pd.concat([data_majority, df_minority_upsampled])
data_upsampled = shuffle(data_upsampled)
# print('Data after downsampling:')
# print(data_upsampled)
# print(data_upsampled['y'].value_counts())

# One-hot encoding for categorical features
# obj_data = data_upsampled.select_dtypes(include=['object']).copy()
data_final = pd.get_dummies(data_upsampled, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
       'race', 'sex', 'nativecountry'], prefix=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
       'race', 'sex', 'nativecountry'])
# print(data_final)

# Split dataset as target variable and feature variabes as arrays
# X = data_final.iloc[:, :-1].values
# y = data_final.iloc[:, -1].values
X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
# print(X)
# print(y)

# Applying standard scaling to get optimized result, avoiding bias
sc = StandardScaler()
X = sc.fit_transform(X)

# Spliting X and y to test and train set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Just for Validation
# X_train.to_csv('X_train.csv')
# X_test.to_csv('X_test.csv')
# y_train.to_csv('y_train.csv')
# y_test.to_csv('y_test.csv')

# # Applying standard scaling to get optimized result, avoiding bias
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

print('\n' * 4)
print('-' * 80)

""" ######################################## Algorithms ########################################

# SVM Classifier
svm = svm.SVC()
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

# Review how model performed
print(classification_report(y_test, pred_svm))
print(svm.score(X_test, y_test))
print(confusion_matrix(y_test, pred_svm))
print('-' * 80)

##############################################################################################

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Review how model performed
print(classification_report(y_test, pred_lr))
print(lr.score(X_test, y_test))
print(confusion_matrix(y_test, pred_lr))
print('-' * 80)

##############################################################################################

# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)

# Review how model performed
print(classification_report(y_test, pred_nb))
print(nb.score(X_test, y_test))
print(confusion_matrix(y_test, pred_nb))
print('-' * 80)

############################################################################################## """

#################################### K-fold Cross Validation ####################################

def get_results(model, X_train, X_test, y_train, y_test):
       model.fit(X_train, y_train)
       pred = model.predict(X_test)

       cr = classification_report(y_test, pred)
       cm = confusion_matrix(y_test, pred)
       sc = model.score(X_test, y_test)
       return cr, cm, sc

# print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[0])
# print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[1])
# print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[2])

# # folds = StratifiedKFold(n_splits=5)
kf = KFold(n_splits=10) # Define the split - into 10 folds 
kf.get_n_splits(X)

# scores_lr = []
# scores_nb = []
# scores_svm = []

print(X.shape)
print(y.shape)
for train_index, test_index in kf.split(X):
       # print("TRAIN:", train_index, "TEST:", test_index)
       # X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
       Train_set = data_final.iloc[train_index]
       X_train = Train_set.loc[:, data_final.columns != 'y']
       y_train = Train_set.loc[:, data_final.columns == 'y']

       Test_set = data_final.iloc[test_index]
       X_test = Test_set.loc[:, data_final.columns != 'y']
       y_test = Test_set.loc[:, data_final.columns == 'y']

       # Applying standard scaling to get optimized result, avoiding bias
       sc = StandardScaler()
       X_train = sc.fit_transform(X_train)
       X_test = sc.fit_transform(X_test)

       # print(Train_set.shape)
       # print(Test_set.shape)
       # print(X_train.shape)
       # print(X_test.shape)
       # print(y_train)       
       # print(y_test)     

       print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[0])
       print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[1])
       print(get_results(LogisticRegression(), X_train, X_test, y_train, y_test)[2])  


# Parameters Tuning via Grid Search
# svms = []
# lrs = []
# nbs =[]

# lr = LogisticRegression()
# svm = svm.SVC()
# nb = GaussianNB()

# scores_lr = cross_val_score(lr, X, y, cv=cv)
# scores_svm = cross_val_score(svm, X, y, cv=cv)
# scores_nb = cross_val_score(nb, X, y, cv=cv)

# print("Accuracy for LogisticRegression: %0.2f (+/- %0.2f)" % (scores_lr.mean(), scores_lr.std() * 2))
# print("Accuracy for SVM: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))
# print("Accuracy for GaussianNB: %0.2f (+/- %0.2f)" % (scores_nb.mean(), scores_nb.std() * 2))