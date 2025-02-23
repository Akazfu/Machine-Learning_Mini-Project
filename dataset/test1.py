import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from scipy import stats
from sklearn.utils import resample
from matplotlib import pyplot

data = pd.read_csv("test.csv",header =0)

def replace_most_common(x):
    if pd.isnull(x):
        return most_common
    else:
        return x
    
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)

not_fraud = data[data.y==0]
fraud = data[data.y==1]

# downsample minority
not_fraud_downsampled = resample(not_fraud,
                          replace=False, # sample with replacement
                          n_samples=len(fraud), # match number in majority class
                          random_state=27) # reproducible results
data = pd.concat([not_fraud_downsampled, fraud])

# checking counts
# print(data.y.value_counts())
#
#print(data['maritalstatus'])
#data = data['maritalstatus'].map(replace_most_common)
#data = data['nativecountry'].map(replace_most_common)
#data = data['sex'].map(replace_most_common)
#data = data['race'].map(replace_most_common)
#data = data['relationship'].map(replace_most_common)
#data = data['occupation'].map(replace_most_common)
#data = data['education'].map(replace_most_common)
#data = data['workclass'].map(replace_most_common)
#
#
#
#data[['fnlwgt']] = imputer.fit_transform(data[['fnlwgt']])
#data[['age']] = imputer.fit_transform(data[['age']])
#data[['capitalgain']] = imputer.fit_transform(data[['capitalgain']])
#data[['capitalloss']] = imputer.fit_transform(data[['capitalloss']])
#data[['educationnum']] = imputer.fit_transform(data[['educationnum']])
#data[['hourperweek']] = imputer.fit_transform(data[['hourperweek']])

#data = data.dropna()
#for i in data.columns.values.tolist():
#    indexNames = data[data[i].values.tolist() == "?" ].index

#print(data[data['nativecountry']== "India"])
#print(data.columns.dtype)
########Optimize feature#############

# print(data["y"].value_counts())
#print(data.head(16))
#print(data.groupby("y").mean())
#print(data.isnull().sum())
#pd.crosstab(data.age,data.y).plot(kind='bar')
#plt.title('Rate with age')
#plt.xlabel('Age')
#plt.ylabel('Rate')
#plt.savefig('Agefig')
#plt.show()
##
##
######classify the category feature#################
tem = list(data.columns)
tem.remove('y')
cat_vars = tem
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]



####################################################################

#over sampling data using smote

####################################################################

X = data_final.iloc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
#print(X.columns)
#from imblearn.over_sampling import SMOTE
#os = SMOTE(random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#columns = X_train.columns
#os_data_X,os_data_y=os.fit_sample(X_train, np.ravel(y_train,order = 'C'))
#os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
#os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
## we can Check the numbers of our data
#print("length of oversampled data is ",len(os_data_X))
#print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
#print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
#print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
#print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))


####################################################################

# Use Recursive Feature Elimination to choose best feature

####################################################################


# data_final_vars=data_final.columns.values.tolist()
# # y=['y']
# # X=[i for i in data_final_vars if i not in y]
# logreg = LogisticRegression(max_iter=120,solver='lbfgs')
# rfe = RFE(logreg, 20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# rfe = rfe.fit(X_train, y_train.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)


####################################################################

# Cross validation with ROC curve and stats data

####################################################################
#

#n_samples, n_features = X.shape
#random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, n_features)]

# #if "SVM":
# cv = StratifiedKFold(n_splits=2)


# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)

# fig, ax = plt.subplots()
# for i, (train, test) in enumerate(cv.split(X, np.ravel(y,order = 'C'))):
#    classifier.fit(X[train], y[train])
#    viz = plot_roc_curve(classifier, X[test], y[test],
#                         name='ROC fold {}'.format(i),
#                         alpha=0.3, lw=1, ax=ax)
#    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
#    interp_tpr[0] = 0.0
#    tprs.append(interp_tpr)
#    aucs.append(viz.roc_auc)

# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#        label='Chance', alpha=.8)

# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(mean_fpr, mean_tpr, color='b',
#        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#        lw=2, alpha=.8)

# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                label=r'$\pm$ 1 std. dev.')

# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#       title="Receiver operating characteristic example")
# ax.legend(loc="lower right")
# plt.show()
# print("!!!!!!!!!!!!!!")
classifier = svm.SVC(kernel='linear')
y1_pred = cross_val_predict(classifier, X, np.ravel(y,order = 'C'), cv=2)
print("done")
conf_mat = confusion_matrix(y, y1_pred)
print(conf_mat)
print(classification_report(y, y1_pred))
print("roc score for svm",roc_auc_score(y,y1_pred))


ns_probs = [0 for _ in range(len(y))]


# calculate scores
ns_auc = roc_auc_score(y, ns_probs)
lr_auc = roc_auc_score(y, y1_pred)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Naive Bayes: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y, y1_pred)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Naive')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# #if "Naive bayes"
# #cv = StratifiedKFold(n_splits=10)

# #
# #tprs = []
# #aucs = []
# #mean_fpr = np.linspace(0, 1, 100)
# #
# #fig, ax = plt.subplots()
# #for i, (train, test) in enumerate(cv.split(X, y)):
# #    classifier.fit(X[train], y[train])
# #    viz = plot_roc_curve(classifier, X[test], y[test],
# #                         name='ROC fold {}'.format(i),
# #                         alpha=0.3, lw=1, ax=ax)
# #    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
# #    interp_tpr[0] = 0.0
# #    tprs.append(interp_tpr)
# #    aucs.append(viz.roc_auc)
# #
# #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# #        label='Chance', alpha=.8)
# #
# #mean_tpr = np.mean(tprs, axis=0)
# #mean_tpr[-1] = 1.0
# #mean_auc = auc(mean_fpr, mean_tpr)
# #std_auc = np.std(aucs)
# #ax.plot(mean_fpr, mean_tpr, color='b',
# #        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
# #        lw=2, alpha=.8)
# #
# #std_tpr = np.std(tprs, axis=0)
# #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# #ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
# #                label=r'$\pm$ 1 std. dev.')
# #
# #ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
# #       title="Receiver operating characteristic example")
# #ax.legend(loc="lower right")
# #plt.show()
# #
nai = GaussianNB()
y2_pred = cross_val_predict(nai, X, np.ravel(y,order = 'C'), cv=2)
# conf_mat = confusion_matrix(y, y2_pred)
# print(classification_report(y, y2_pred))
# print(roc_auc_score(y,y2_pred))

# ns_probs = [0 for _ in range(len(y))]


# # calculate scores
# ns_auc = roc_auc_score(y, ns_probs)
# lr_auc = roc_auc_score(y, y2_pred)
# # summarize scores
# print('No Skill: ROC AUC=%.3f' % (ns_auc))
# print('Naive Bayes: ROC AUC=%.3f' % (lr_auc))
# # calculate roc curves
# ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
# lr_fpr, lr_tpr, _ = roc_curve(y, y2_pred)
# # plot the roc curve for the model
# pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
# pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Naive')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# # show the legend
# pyplot.legend()
# # show the plot
# pyplot.show()

# #
##if "logistic regression"
#cv = StratifiedKFold(n_splits=10)
# #
# #tprs = []
# #aucs = []
# #mean_fpr = np.linspace(0, 1, 100)
# #
# #fig, ax = plt.subplots()
# #for i, (train, test) in enumerate(cv.split(X, y)):
# #    classifier.fit(X[train], y[train])
# #    viz = plot_roc_curve(classifier, X[test], y[test],
# #                         name='ROC fold {}'.format(i),
# #                         alpha=0.3, lw=1, ax=ax)
# #    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
# #    interp_tpr[0] = 0.0
# #    tprs.append(interp_tpr)
# #    aucs.append(viz.roc_auc)
# #
# #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
# #        label='Chance', alpha=.8)
# #
# #mean_tpr = np.mean(tprs, axis=0)
# #mean_tpr[-1] = 1.0
# #mean_auc = auc(mean_fpr, mean_tpr)
# #std_auc = np.std(aucs)
# #ax.plot(mean_fpr, mean_tpr, color='b',
# #        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
# #        lw=2, alpha=.8)
# #
# #std_tpr = np.std(tprs, axis=0)
# #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# #ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
# #                label=r'$\pm$ 1 std. dev.')
# #
# #ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
# #       title="Receiver operating characteristic example")
# #ax.legend(loc="lower right")
# #plt.show()
# #
# #
print("log start")
logi = LogisticRegression(solver='liblinear')
y3_pred = cross_val_predict(logi, X, np.ravel(y,order = 'C'), cv=2)

# conf_mat = confusion_matrix(y, y3_pred)
# print(conf_mat)
# print(classification_report(y, y3_pred))
# print(roc_auc_score(y,y3_pred))

# ns_probs = [0 for _ in range(len(y))]


# # calculate scores
# ns_auc = roc_auc_score(y, ns_probs)
# lr_auc = roc_auc_score(y, y3_pred)
# # summarize scores
# print('No Skill: ROC AUC=%.3f' % (ns_auc))
# print('Logistic: ROC AUC=%.3f' % (lr_auc))
# # calculate roc curves
# ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
# lr_fpr, lr_tpr, _ = roc_curve(y, y3_pred)
# # plot the roc curve for the model
# pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
# pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# # show the legend
# pyplot.legend()
# # show the plot
# pyplot.show()

# #
#
#####################################################
#
## Wilcoxon signed rank test
#
#####################################################

difference =[]
# SVM & Naive Bayes
for i in range(len(y1_pred)):
   difference[i] = y1_pred[i]-y2_pred[i]

w, p = stats.wilcoxon(difference)
print("SVM & Naive:w:",w,"p:",p)

# SVM & logistic regression
for i in range(len(y1_pred)):
   difference[i] = y1_pred[i]-y3_pred[i]

w, p = stats.wilcoxon(difference)
print("SVM & Logistic:w:",w,"p:",p)

# Naive Bayes & logistic regression
for i in range(len(y1_pred)):
   difference[i] = y1_pred[i]-y3_pred[i]

w, p = stats.wilcoxon(difference)
print("Naive Bayes & Logistic:w:",w,"p:",p)
