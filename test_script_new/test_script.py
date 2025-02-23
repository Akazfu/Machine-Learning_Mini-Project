import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scikitplot as skplt
from scipy import stats

from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


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
       probs = model.predict_proba(X_test)

       cr = classification_report(y_test, pred)
       accuracy = accuracy_score(y_test, pred)
       precision = precision_score(y_test, pred)
       recall = recall_score(y_test, pred)
       f1 = f1_score(y_test, pred)
       auc = roc_auc_score(y_test, pred)
       skplt.metrics.plot_roc_curve(y_test, probs)
       plt.savefig('roc.png')
       return cr, accuracy, precision, recall, f1, auc

run_count = 0

# Define the split - into 10 folds
folds = KFold(n_splits=10)

# Metric Lists for the test results
scores = []
precisions = []
recalls = []
f1s = []
aucs = []

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

       # Parameters Tuning via Grid Search(for loop)
       lr_params = {'C': [0.01, 0.1, 1], 'penalty': ['l1', 'l2']}
       svm_params = {'C': [0.01, 0.1, 1], 'kernel': ['linear', 'poly', 'rbf']}
       nb_params = {'priors':[[0.1, 0.9]]}

       # Algorithms to test
       lr = LogisticRegression(penalty = 'none', C = 0.0001)
       nb = GaussianNB()
       svmc = svm.SVC(kernel = 'linear', C = 0.0001, probability = True)

       cr, accuracy, precision, recall, f1, auc = get_results(nb, X_train, X_test, y_train, y_test)
       # cr, accuracy, precision, recall, f1 = get_results(nb, X_train, X_test, y_train, y_test)
       # cr, accuracy, precision, recall, f1 = get_results(svmc, X_train, X_test, y_train, y_test)
       
       print(cr)
       scores.append(accuracy)
       precisions.append(precision)
       recalls.append(recall)
       f1s.append(f1)
       aucs.append(auc)
       
print('\n' * 2)
print('*' * 80)
# print('LogisticRegression:  Param:(penalty = none):')
# print('svm.SVC:  Param:(kernel = linear, C = 0.0001):')
print('GaussianNB:  Param:(no priors):')

print("Accuracy: %0.6f (Std: +/- %0.6f), Variance: %0.6f" % (np.mean(scores), np.std(scores), np.var(scores)))
print("Precision: %0.6f (Std: +/- %0.6f), Variance: %0.6f" % (np.mean(precisions), np.std(precisions), np.var(scores)))
print("Recall: %0.6f (Std: +/- %0.6f), Variance: %0.6f" % (np.mean(recalls), np.std(recalls), np.var(recalls)))
print("F1: %0.6f (Std: +/- %0.6f), Variance: %0.6f" % (np.mean(f1s), np.std(f1s), np.var(f1s)))
print("Auc: %0.6f (Std: +/- %0.6f), Variance: %0.6f" % (np.mean(aucs), np.std(aucs), np.var(f1s)))

#################################### Wilcoxon signed rank test ####################################
# X = data_final.loc[:, data_final.columns != 'y']
# y = data_final.loc[:, data_final.columns == 'y']

# # y1_pred = cross_val_predict(svm.SVC(kernel='linear'), X, np.ravel(y,order = 'C'), cv=2)
# y2_pred = cross_val_predict(GaussianNB(), X, np.ravel(y,order = 'C'), cv=2)
# y3_pred = cross_val_predict(LogisticRegression(), X, np.ravel(y,order = 'C'), cv=2)

# difference =[]
# # # SVM & Naive Bayes
# # for i in range(len(y1_pred)):
# #    difference[i] = y1_pred[i]-y2_pred[i]

# # w, p = stats.wilcoxon(difference)
# # print("SVM & Naive:w:",w,"p:",p)

# # # SVM & logistic regression
# # for i in range(len(y1_pred)):
# #    difference[i] = y1_pred[i]-y3_pred[i]

# # w, p = stats.wilcoxon(difference)
# # print("SVM & Logistic:w:",w,"p:",p)

# # Naive Bayes & logistic regression
# for i in range(len(y2_pred)):
#    difference[i] = y2_pred[i]-y3_pred[i]

# w, p = stats.wilcoxon(difference)
# print("Naive Bayes & Logistic:w:",w,"p:",p)

# print('GaussianNB:  Param:(no priors):')
# print('Accuracy: 0.500133 (Std: +/- 0.016439), Variance: 0.000270')
# print('Recall: 0.540100 (Std: +/- 0.000739), Variance: 0.000000')
# print('Precision: 0.472000 (Std: +/- 0.400000), Variance: 0.000270')
# print('F1: 0.511200 (Std: +/- 0.001076), Variance: 0.000001')
# print('Auc: 0.500135 (Std: +/- 0.000269), Variance: 0.000001')

# print('GaussianNB:  Param:(priors = [0.1, 0.9]):')
# print('Accuracy: 0.499667 (Std: +/- 0.012825), Variance: 0.000164')
# print('Precision: 0.522000 (Std: +/- 0.000830), Variance: 0.000164')
# print('Recall: 0.451100 (Std: +/- 0.607800), Variance: 0.000000')
# print('F1: 0.514320 (Std: +/- 0.000116), Variance: 0.000011')
# print('Auc: 0.499676 (Std: +/- 0.000523), Variance: 0.000003')
