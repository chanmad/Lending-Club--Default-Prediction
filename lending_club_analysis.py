# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
# figure size
rcParams['figure.figsize'] = 15,10
pd.set_option('display.max_columns', None)

#file input
loan_all = pd.read_csv("C:\\Users\\Madhava\\OneDrive\\Documents\\UC-BANA\\Summer 2018\\Capstone\\Data LC\\combine.csv",low_memory=False)
#formatted numbers and dates in excel
loan_all.info()
loan_all.describe()

#PreProcessing

#It appears there are a lot of NaNs. Few cols with 100% NAs- id, member_id, url; zip_code info is masked
loan_all.drop(['id', 'member_id', 'url', 'zip_code' ,'title' ], axis=1, inplace=True)

#also dropping cols with < 1/3 data
missingdata = [x for x in loan_all.count() < len(loan_all)*0.33]
loan_all.drop(loan_all.columns[missingdata], axis=1, inplace=True)
loan_all.shape
loan_all.columns

unique = loan_all.nunique()
unique = unique[unique.values == 1]
loan_all.drop(labels = list(unique.index), axis =1, inplace=True)
missingdata = [x for x in loan_all.count() < len(loan_all)*0.33]
loan_all.drop(loan_all.columns[missingdata], axis=1, inplace=True)
loan_all.shape
loan_all.columns

loan_all['annual_inc']= loan_all['annual_inc'].astype(float)
loan_all['annual_inc'].describe()
#len(loan_all[loan_all['annual_inc']>1e+06]) #251 appear to be outliers
loan=loan_all.drop(loan_all[loan_all.annual_inc>1e+06].index)

#data cleaning- bucketing, drop NAs
loan['revol_util'].isnull().sum() #856 NAs
loan[loan['revol_util'].isnull()]['loan_status'].value_counts() #mostly current loans
loan.dropna(subset=['revol_util'],inplace=True)
loan['revol_util'].isna().sum()

#buckets
loan['int_rate']=loan['int_rate'].astype(str)
loan['int_rate']= loan['int_rate'].map(lambda x: x.rstrip('%'))
loan['int_rate']= loan['int_rate'].astype(float)
loan['int_rate'].describe() # 5-30.9
buck = [0, 5, 10, 15, 20,25, 35]
lab = ['0-5', '5-10', '10-15', '15-20', '20-25','>25']
loan['int_rate_range'] = pd.cut(loan['int_rate'], buck, labels=lab)

loan['loan_amnt'].describe() #0-40k
buck = [0, 5000, 10000, 15000, 20000, 25000,40000]
lab = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000 and above']
loan['loan_amnt_range'] = pd.cut(loan['loan_amnt'], buck, labels=lab)

loan['annual_inc'].describe() #range 1 to 1 mill
buck = [0, 25000, 50000, 75000, 100000,1000000]
lab = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000 and above']
loan['annual_inc_range'] = pd.cut(loan['annual_inc'], buck, labels=lab)

#plotting
sns.distplot(loan['loan_amnt'])
sns.distplot(loan['int_rate'])
sns.distplot(loan['annual_inc'])

sns.countplot(loan['loan_status'])
loan['loan_status'].value_counts()
#good loans- Fully paid, current and bad loans- charged off,default, late, in grace period 
#deleting issued loans from data
loan['loan_status'].value_counts()/len(loan)
loan.drop(loan[loan.loan_status== 'Issued'].index, inplace=True)
loan['good_loan'] = np.where((loan.loan_status == 'Fully Paid') |
                        (loan.loan_status == 'Current'), 1, 0)
loan['good_loan'].value_counts()/len(loan)

sns.countplot(loan['good_loan'])
sns.countplot(loan['purpose'],hue=loan['good_loan'])
sns.countplot(loan['purpose'],hue=loan['loan_amnt_range'])

#home ownership- any and none dont signify anything
loan.drop(loan[loan['home_ownership']== 'ANY'].index, inplace=True)
loan.drop(loan[loan['home_ownership']== 'NONE'].index, inplace=True)
sns.countplot(loan['home_ownership'],hue=loan['good_loan']) 

#employment length
sns.countplot(loan['emp_length'],hue=loan['good_loan'])
sns.countplot(loan['emp_length'],hue=loan['loan_amnt_range'])

#geography
sns.countplot(loan['addr_state'],hue=loan['good_loan']) 
sns.countplot(loan['addr_state'],hue=loan['loan_amnt_range']) 

#monthly trend
loan['issue_yr']=pd.DatetimeIndex(loan['issue_d']).year
loan['issue_mon']=pd.DatetimeIndex(loan['issue_d']).month
sns.countplot(loan['issue_mon'],hue=loan['good_loan'])

#understanding correlation between some key business variables
cor_loan=loan[['loan_amnt','annual_inc', 'good_loan', 'int_rate', 'dti', 
               'tot_cur_bal', 'funded_amnt']]
f, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(cor_loan.corr(), 
            xticklabels=cor_loan.columns.values,
            yticklabels=cor_loan.columns.values,annot= True)
plt.show()

#VARIABLE SELECTION ################################
#for classification purpose, we need to drop retrospective variables
loan.drop(['funded_amnt', 'funded_amnt_inv', 'total_pymnt', 
'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 
'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d', 
'recoveries', 'collection_recovery_fee'], axis=1, inplace=True)

#change variable type- encoding - 'earliest_cr_line' and emp_length

loan.head()

loan['emp_length'].value_counts()
emp_range= {'< 1 year':0.5, '1 year':1, '2 years': 2, '3 years':3,
            '4 years':4, '5 years':5,'6 years':6,'7 years':7,
            '8 years':8,'9 years':9, '10+ years':10}
loan['emplen'] = loan["emp_length"].map(emp_range)
loan['emplen'].isnull().sum() 
loan['emplen'].value_counts() 

############Missing Value- Imputation############
nullseries=pd.isnull(loan).sum()
nullseries[nullseries>0]

loan['emplen'] = loan['emplen'].replace(np.nan, 10)
loan.drop(['emp_length'],axis=1,inplace=True)

loan['mths_since_last_delinq'] = loan['mths_since_last_delinq'].fillna(loan['mths_since_last_delinq'].median()) #mean and median v similar
loan['mths_since_recent_revol_delinq']=loan['mths_since_recent_revol_delinq'].fillna(loan['mths_since_recent_revol_delinq'].median())

#very few missing values for all of them
loan['dti'] = loan['dti'].fillna(loan['dti'].mean())
loan['inq_last_6mths'] = loan['inq_last_6mths'].fillna(loan['inq_last_6mths'].mean())
loan['open_acc_6m'] = loan['open_acc_6m'].fillna(loan['open_acc_6m'].median())
loan['open_act_il'] = loan['open_act_il'].fillna(loan['open_act_il'].median())
loan['open_il_12m'] = loan['open_il_12m'].fillna(loan['open_il_12m'].median())
loan['open_il_24m'] = loan['open_il_24m'].fillna(loan['open_il_24m'].median())
loan['open_rv_12m'] = loan['open_rv_12m'].fillna(loan['open_rv_12m'].median())
loan['open_rv_24m'] = loan['open_rv_24m'].fillna(loan['open_rv_24m'].median())
loan['max_bal_bc'] = loan['max_bal_bc'].fillna(loan['max_bal_bc'].median())
loan['all_util'] = loan['all_util'].fillna(loan['all_util'].median())
loan['inq_fi'] = loan['inq_fi'].fillna(loan['inq_fi'].median())
loan['total_cu_tl'] = loan['total_cu_tl'].fillna(loan['total_cu_tl'].median())
loan['total_bal_il']=loan['total_bal_il'].fillna(loan['total_bal_il'].mean())
loan['inq_last_12m'] = loan['inq_last_12m'].fillna(loan['inq_last_12m'].median())

#A lot of NAs- with integer values so median replacement
loan['mths_since_rcnt_il']=loan['mths_since_rcnt_il'].fillna(loan['mths_since_rcnt_il'].median())
loan['il_util']=loan['il_util'].fillna(loan['il_util'].median())

loan['bc_open_to_buy'].value_counts() # 0s occur the most
loan['bc_open_to_buy']=loan['bc_open_to_buy'].fillna(0)

# bc_util= total current balance/credit limit
loan[loan['bc_util'].isnull()]['bc_open_to_buy'].value_counts() #mostly 0s
loan[loan['bc_open_to_buy']==0]['bc_util'].value_counts() #bcutil is~100
loan['bc_util']=loan['bc_util'].fillna(100)
loan['mo_sin_old_il_acct']=loan['il_util'].fillna(loan['il_util'].median()) #mostly around 120-130
loan['mths_since_recent_bc']=loan['mths_since_recent_bc'].fillna(loan['mths_since_recent_bc'].median()) #no corr found
loan['mths_since_recent_inq']=loan['mths_since_recent_inq'].fillna(loan['mths_since_recent_inq'].median()) #no corr found
loan['num_tl_120dpd_2m']=loan['num_tl_120dpd_2m'].fillna(0) #0 common

#Feature Engineering- credit length
loan['earliest_crline_yr']=pd.DatetimeIndex(loan['earliest_cr_line']).year
#loan['earliest_crline_mon'],loan['earliest_crline_yr'] = loan['earliest_cr_line'].str.split('-', 1).str

#changing similarly for issue year as well- to capture the difference in years
loan['issue_yr'].apply(int)
loan.loc[loan['issue_yr'].apply(pd.to_numeric, args=('coerce',))> 18, 'issue_yr'] = '19' + loan['issue_yr'].astype(str)
loan.loc[loan['issue_yr'].apply(pd.to_numeric, args=('coerce',))< 19, 'issue_yr'] = '20' + loan['issue_yr'].astype(str)
loan['issue_yr'].apply(int)
loan.drop(['issue_d', 'issue_mon'], axis=1, inplace=True)

#now to introduce new variable for checking the difference of these 2 and drop them
loan['credit_len']=loan['issue_yr'].apply(int)-loan['earliest_crline_yr'].apply(int)
loan.drop(['issue_yr', 'earliest_crline_yr'],axis=1, inplace=True)

#Categorical variables- introducing levels/encoding where not ordinal 
#verification status,subgrade, purpose,addr_state

verification_map={'Source Verified':3, 'Verified':2, 'Not Verified':1}
loan['verification_status']=loan['verification_status'].map(verification_map)

ownership_map={'MORTGAGE':1, 'RENT':2, 'OWN':3}
loan['home_ownership']=loan['home_ownership'].map(ownership_map)

subgrade_map={'A1':1,'A2':2, 'A3':3, 'A4':4, 'A5':5, 'B1':6, 'B2':7, 'B3':8, 'B4':9, 'B5':10, 
              'C1':11, 'C2':12, 'C3':13, 'C4':14, 'C5':15, 'D1':16, 'D2':17, 'D3':18, 
              'D4':19, 'D5':20, 'E1':21, 'E2':22, 'E3':23, 'E4':24, 'E5':25, 'F1':26, 
              'F2':27, 'F3':28, 'F4':29, 'F5':30, 'G1':31, 'G2':32, 'G3':33, 'G4':34, 'G5':35}
loan['sub_grade']=loan['sub_grade'].map(subgrade_map)
loan.drop(['grade'],axis=1, inplace=True)

hardship_map={'N':0, 'Y':1}
loan['hardship_flag']=loan['hardship_flag'].map(hardship_map)

debtsett_map={'N':0, 'Y':1}
loan['debt_settlement_flag']=loan['debt_settlement_flag'].map(debtsett_map)

#purpose needs to be hot encoded- so better to drop unnecessary levels
loan= loan[loan['purpose'] != 'educational']
loan= loan[loan['purpose'] !='wedding']
loan= loan[loan['purpose'] !='other']

loan['loan_amnt'].isna().sum()
loan.shape

enc1= pd.get_dummies(loan['purpose'])
loan=pd.concat((loan,enc1), axis=1)
loan.drop(['purpose'],axis=1,inplace=True)

enc2= pd.get_dummies(loan['addr_state'])
loan=pd.concat((loan,enc2), axis=1)
loan.drop(['addr_state'],axis=1,inplace=True)

enc3= pd.get_dummies(loan['disbursement_method'])
loan=pd.concat((loan,enc3), axis=1)
loan.drop(['disbursement_method'],axis=1,inplace=True)

enc4= pd.get_dummies(loan['application_type'])
loan=pd.concat((loan,enc4), axis=1)
loan.drop(['application_type'],axis=1,inplace=True)

#to delete
loan.drop(['earliest_cr_line','loan_amnt_range', 'annual_inc_range',
           'int_rate_range','loan_status', 'next_pymnt_d','emp_title'
           ,'pymnt_plan','initial_list_status', 'total_rec_late_fee'],axis=1, inplace=True)
loan.isna()
loan.isnull().sum()

###################SAMPLING##############

from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

#SMOTE: can increase recall at the cost of precision
#undersampling: if less data overall, minority class gets you less data
#ADASYN will focus on samples which are difficult to classify with NN
X= loan[loan.columns.difference(['good_loan'])] #except label
y= loan['good_loan']

#Split original data-oversample training set-test on original test data
#http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/under-sampling/plot_random_under_sampler.html#sphx-glr-auto-examples-under-sampling-plot-random-under-sampler-py
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE 
from imblearn.under_sampling import RandomUnderSampler

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0, stratify=y)

#OVERSAMPLING-ADASYN
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_sample(X_train, y_train)
Counter(y_res) #data exploded to 1mn records with 145 variables

#OVERSAMPLING-SMOTE
sm= SMOTE(random_state=42)
X_sm, y_sm = sm.fit_sample(X_train, y_train)
#Counter(y_res) #data exploded to 1mn records with 145 variables

#OVERSAMPLING-RANDOM
ros= RandomOverSampler(random_state=555)
X_over, y_over= ros.fit_sample(X_train, y_train)

#Undersampling
rus = RandomUnderSampler(return_indices=True, random_state=555)
X_resampled, y_resampled, idx_resampled= rus.fit_sample(X_train, y_train)

#stadardization
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std= scaler.fit_transform(X_sm)
X_std_test= scaler.fit_transform(X_test)

###################CLASSIFICATION##############
#1 Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,classification_report

#SMOTE
lr_sm = LogisticRegression() 
lr_sm.fit(X_sm, y_sm)
lr_sm.score(X_sm, y_sm)
y_pred_sm= lr_sm.predict(X_test)
accuracy_score(y_test, y_pred_sm)
roc_auc_score(y_test, y_pred_sm)
classification_report(y_test, y_pred_sm)
f1_score(y_test, y_pred_sm)

lr_sm.coef_.shape

X.columns[128]

#ADASYN
model = LogisticRegression() 
model.fit(X_res, y_res)
model.score(X_res, y_res)
y_pred= model.predict(X_test)
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
classification_report(y_test, y_pred)

#oversample- random
model_over = LogisticRegression()
model_over.fit(X_over, y_over) 
model_over.score(X_over, y_over)
y_pred_over= model_over.predict(X_test) 
accuracy_score(y_test, y_pred_over)
roc_auc_score(y_test, y_pred_over)
classification_report(y_test, y_pred_over)

#undersample
model_under = LogisticRegression()
model_under.fit(X_resampled, y_resampled) 
model_under.score(X_resampled, y_resampled)
y_pred_under= model_under.predict(X_test) 
accuracy_score(y_test, y_pred_under)
roc_auc_score(y_test, y_pred_under)
classification_report(y_test, y_pred_under)

#Of all SMOTE has the highest F1 score- which is the class we are concerned about

#2 Decision Tree
from sklearn.tree import tree, DecisionTreeClassifier
dt = tree.DecisionTreeClassifier(criterion='gini')
dt.fit(X_sm, y_sm)
dt.score(X_sm, y_sm)
y_pred_sm= dt.predict(X_test)
accuracy_score(y_test, y_pred_sm)
roc_auc_score(y_test, y_pred_sm)
classification_report(y_test, y_pred_sm)
f1_score(y_test, y_pred_sm)    


#visualization
from os import system
dotfile = open("C:/Users/Madhava/OneDrive/Documents/UC-BANA/Summer 2018/Capstone:/dtree2.dot", 'w')
tree.export_graphviz(dt, out_file = dotfile, feature_names = X_sm.columns)
dotfile.close()
system("dot -Tpng D:.dot -o D:/dtree2.png")

with open("dt.txt", "w") as f:
    f = tree.export_graphviz(dt, out_file=f)

#3 Random Forests
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(n_estimators=80, max_features= 'log2')
rf.fit(X_sm, y_sm)
#rf.score(X_sm, y_sm)
y_rf= rf.predict(X_test)
accuracy_score(y_test, y_rf)
roc_auc_score(y_test, y_rf)
classification_report(y_test, y_rf)
f1_score(y_test, y_rf)    

#tuning RF
from sklearn.grid_search import GridSearchCV
param_grid = { 
    'n_estimators': [20, 80],
    'max_features': [None, 'log2', 'sqrt']
}
CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_sm, y_sm)
print (CV_rf.best_params)


#variable importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

#PCA
from sklearn.preprocessing import scale
from sklearn import decomposition
X_scaled=scale(X_sm)
pca=decomposition.PCA()
X_pca=pca.fit(X_scaled)

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
pca = decomposition.PCA(n_components=120)
pca.fit(X)
X1=pca.fit_transform(X)

#GBM
from sklearn.ensemble import GradientBoostingClassifier  
gb = GradientBoostingClassifier(n_estimators=80, learning_rate=1, 
                                random_state=42)
gb.fit(X_sm, y_sm)
#rf.score(X_sm, y_sm)
y_gb= gb.predict(X_test)
#accuracy_score(y_test, y_rf)
roc_auc_score(y_test, y_gb)
classification_report(y_test, y_gb)
f1_score(y_test, y_gb) 


#SVM
from sklearn import svm
from sklearn.svm import SVC
model_svm = svm.SVC(random_state=42, tol=100,class_weight='balanced')
model_svm.fit(X_std, y_sm)
y_svm= model_svm.predict(X_std_test)
accuracy_score(y_test, y_svm)
roc_auc_score(y_test, y_svm)
classification_report(y_test, y_svm)
f1_score(y_test, y_svm)

#keras- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(777)
# create model
model = Sequential()
model.add(Dense(12, input_dim=143, activation='relu'))
model.add(Dense(143, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_std, y_sm, epochs=50, batch_size=10)
# evaluate the model
scores = model.evaluate(X_std, y_sm)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_std_test)
y_rounded = [round(x[0]) for x in predictions]
scores_test = model.evaluate(X_std_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

accuracy_score(y_test, y_rounded)
f1_score(y_test, y_rounded)
roc_auc_score(y_test, y_rounded)




