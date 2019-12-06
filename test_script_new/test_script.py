import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statistics 

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

# drop data if contains ' ?'
data = data[~data.eq(' ?').any(1)]

# handling imbalanced data via downsampling
data_majority = data[data.y==0]
data_minority = data[data.y==1]
min_len = len(data[data['y']==1])

data_majority_downsampled = resample(data_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=min_len)     # to match majority class
 
# Combine minority class with downsampled majority class and shuffle
data_downsampled = pd.concat([data_majority_downsampled, data_minority])
data_downsampled = shuffle(data_downsampled)
# print(data_downsampled['y'].value_counts())

# One-hot encoding for categorical features
data_final = pd.get_dummies(data_downsampled, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
       'race', 'sex', 'nativecountry'], prefix=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship',
       'race', 'sex', 'nativecountry'])
print('\n' * 4)

#################################### K-fold Cross Validation ####################################

# A simple function which returns the classification reports, confusion matrix, the accuracy score by indices 0, 1, 2
def get_results(model, X_train, X_test, y_train, y_test):
       model.fit(X_train, y_train)
       pred = model.predict(X_test)

       cr = classification_report(y_test, pred)
       cm = confusion_matrix(y_test, pred)
       sc = model.score(X_test, y_test)
       return cr, cm, sc

run_count = 0

# Define the split - into 10 folds
folds = KFold(n_splits=2)  

# Accuracy list of different algorithms
scores_lr = []
scores_nb = []
scores_svm = []

# Parameters Tuning via Grid Search(for loop)
# lrs = []
# svms = []
# nbs =[]

for train_index, test_index in folds.split(data_final):
       run_count += 1
       print('=' * 80) 
       print('Running on iteration k =  ' + str(run_count) )
       print('=' * 80) 
       # Split dataset to target labels and feature variabes as arrays
       Train_set = data_final.iloc[train_index]
       Test_set = data_final.iloc[test_index]

       X_train = Train_set.loc[:, data_final.columns != 'y']
       X_test = Test_set.loc[:, data_final.columns != 'y']
       y_train = Train_set.loc[:, data_final.columns == 'y']
       y_test = Test_set.loc[:, data_final.columns == 'y']

       # Applying standard scaling to get optimized results, avoiding bias
       sc = StandardScaler()
       X_train = sc.fit_transform(X_train)
       X_test = sc.fit_transform(X_test)

       lr = LogisticRegression()
       nb = GaussianNB()
       svmc = svm.SVC()

       print(get_results(lr, X_train, X_test, y_train, y_test)[0])
       print(get_results(nb, X_train, X_test, y_train, y_test)[0])
       print(get_results(svmc, X_train, X_test, y_train, y_test)[0])
       
       scores_lr.append(get_results(lr, X_train, X_test, y_train, y_test)[2])
       scores_nb.append(get_results(nb, X_train, X_test, y_train, y_test)[2])
       scores_svm.append(get_results(svmc, X_train, X_test, y_train, y_test)[2])

print('\n' * 2)
print('*' * 80)
print("Accuracy for LogisticRegression: %0.2f (+/- %0.2f)" % (statistics.mean(scores_lr), statistics.stdev(scores_lr)))
print("Accuracy for GaussianNB: %0.2f (+/- %0.2f)" % (statistics.mean(scores_nb), statistics.stdev(scores_nb)))
print("Accuracy for SVM: %0.2f (+/- %0.2f)" % (statistics.mean(scores_svm), statistics.stdev(scores_svm)))