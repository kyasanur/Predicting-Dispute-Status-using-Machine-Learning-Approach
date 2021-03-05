# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas
import scipy.stats
#from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %%
# importing and reading the dataset
dataset = pandas.read_csv('C:\\Reva\\BA06\\Capstone Projects\\data\\Dispute Mask Data_Final.csv',encoding= 'unicode_escape')
dataset.columns = dataset.columns.str.replace(' ', '_')
dataset = dataset[ ['Dispute_Status'] + [ col for col in dataset.columns if col != 'Dispute_Status' ] ]
dataset['Creation_Date'] = pd.to_datetime(dataset['Creation_Date'])
dataset["Invoice_Amount"] = dataset["Invoice_Amount"].astype(float, errors = 'raise')
dataset["Dispute_Amount"] = dataset["Dispute_Amount"].astype(float, errors = 'raise')
dataset['Creation_month'] = pd.DatetimeIndex(dataset['Creation_Date']).month
dataset['Creation_day'] = pd.DatetimeIndex(dataset['Creation_Date']).day 
dataset['Creation_week'] = pd.DatetimeIndex(dataset['Creation_Date']).week 
dataset['Dispute_Status'] = np.where(dataset.Dispute_Status == 'COMPLETE', 0, dataset.Dispute_Status)
dataset['Dispute_Status'] = np.where(dataset.Dispute_Status == 'CANCELLED', 1, dataset.Dispute_Status)
dataset['Dispute_Status'] = np.where(dataset.Dispute_Status == 'NOT_APPROVED', 1, dataset.Dispute_Status)
dataset['Dispute_Status'] = np.where(dataset.Dispute_Status == 'PENDING_APPROVAL', 2, dataset.Dispute_Status)
dataset['Dispute_Status'] = np.where(dataset.Dispute_Status == 'APPROVED_PEND_COMP', 2, dataset.Dispute_Status)
dataset["Dispute_Status"] = dataset["Dispute_Status"].astype(int, errors = 'raise')


# %%
complete = dataset['Dispute_Status'] != 2
reject = dataset['Dispute_Status'] == 2
complete_dataset = dataset[complete]
reject_dataset = dataset[reject]
reject_balanced_dataset = dataset[reject]
complete_dataset.head()
reject_dataset.head()


# %%
complete_dataset.info()
reject_dataset.info()
reject_balanced_dataset.info()


# %%
complete_dataset['Dispute_Status'].unique()
reject_dataset['Dispute_Status'].unique()
reject_balanced_dataset['Dispute_Status'].unique()


# %%
complete_dataset.drop(['Dispute_no','Customer_Number','Updated_Activity_Result','New_Invoice_Amount','Activity_Result','Activity_Status','Days_Pending','Approval_Date','Credit_Memo_Creation_Date','Credit_Memo_Amount','Creation_Date','Inv_Creation_Date','Notified_Date'], axis=1, inplace=True)
reject_dataset.drop(['Dispute_no','Customer_Number','Updated_Activity_Result','New_Invoice_Amount','Activity_Result','Activity_Status','Days_Pending','Approval_Date','Credit_Memo_Creation_Date','Credit_Memo_Amount','Creation_Date','Inv_Creation_Date','Notified_Date'], axis=1, inplace=True)
reject_balanced_dataset.drop(['Dispute_no','Customer_Number','Updated_Activity_Result','New_Invoice_Amount','Activity_Result','Activity_Status','Days_Pending','Approval_Date','Credit_Memo_Creation_Date','Credit_Memo_Amount','Creation_Date','Inv_Creation_Date','Notified_Date'], axis=1, inplace=True)


# %%
for col in complete_dataset.columns:
    if complete_dataset[col].dtype == 'object':
        complete_dataset[col] = pd.Categorical(complete_dataset[col]).codes


# %%
complete_dataset


# %%
import statsmodels.formula.api as snf


# %%
complete_dataset.rename(columns={ complete_dataset.columns[8]: "RecipientTeam_BroadLevel" },inplace=True)


# %%
complete_dataset.head()


# %%
scipy.stats.chisquare(complete_dataset["Dispute_Status"].value_counts())
scipy.stats.chisquare(complete_dataset["Assigned_User"].value_counts())
scipy.stats.chisquare(complete_dataset["Trx_Type"].value_counts())
scipy.stats.chisquare(complete_dataset["Reason_Code"].value_counts())
scipy.stats.chisquare(complete_dataset["Requester"].value_counts())
scipy.stats.chisquare(complete_dataset["Country"].value_counts())
scipy.stats.chisquare(complete_dataset["RecipientTeam_BroadLevel"].value_counts())


# %%
RecipientTeam_BroadLevel = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["RecipientTeam_BroadLevel"])
scipy.stats.chi2_contingency(RecipientTeam_BroadLevel)


# %%
Country = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["Country"])
scipy.stats.chi2_contingency(Country)


# %%
Requester = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["Requester"])
scipy.stats.chi2_contingency(Requester)


# %%
ReasonCode = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["Reason_Code"])
scipy.stats.chi2_contingency(ReasonCode)


# %%
TrxType = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["Trx_Type"])
scipy.stats.chi2_contingency(TrxType)


# %%
AssignedUser = pd.crosstab(complete_dataset["Dispute_Status"],complete_dataset["Assigned_User"])
scipy.stats.chi2_contingency(AssignedUser)


# %%
complete_dataset.head()


# %%
# Split-out validation dataset
array = complete_dataset.values
Xc = array[:,1:]
Yc = array[:,0]
Xc_train, Xc_test, Yc_train, Yc_test = model_selection.train_test_split(Xc, Yc, test_size=0.2, random_state=7)


# %%
pd.DataFrame(Xc)


# %%
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xc_train = sc.fit_transform(Xc_train)
Xc_test = sc.transform(Xc_test)
print(Xc_train)
print(Xc_test)


# %%
from collections import Counter
counter = Counter(Yc)
for k, v in counter.items():
    dist = v / len(Yc) * 100
    print(f'Class={k}, n={v} ({dist}%)')


# %%
Unbalanced = pd.DataFrame(Yc)
Unbalanced['Disputed_Status for Imbalanced Dataset'] = pd.DataFrame(Yc)
Unbalanced


# %%
plt.figure(figsize=(5,5))
sns.countplot(x="Disputed_Status for Imbalanced Dataset", data=Unbalanced)


# %%
# Test options and evaluation metric c
seed = 10
scoring = 'accuracy'


# %%
# Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# %%
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=seed)
 cv_results = model_selection.cross_val_score(model, Xc_train, Yc_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)


# %%
# Make predictions for Decision Tree
cart = DecisionTreeClassifier()
cart.fit(Xc_train, Yc_train)
predictions = cart.predict(Xc_test)
predictions_train = cart.predict(Xc_train)
print(accuracy_score(Yc_test, predictions))
print(confusion_matrix(Yc_test, predictions))
print(classification_report(Yc_test, predictions))


# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Yc_test, predictions)
accuracy_score(Yc_test, predictions)
label1 = ["Predicted 0", "Predicted 1"]
label2 = ["True 0", "True 1"]
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", xticklabels=label1, yticklabels=label2)
plt.show()


# %%
print(accuracy_score(Yc_train, predictions_train))
print(confusion_matrix(Yc_train, predictions_train))
print(classification_report(Yc_train, predictions_train))


# %%
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes=10, criterion = 'entropy', random_state = 0)
classifier.fit(Xc_train, Yc_train)
import pydotplus as pydot
from sklearn import tree
tree.plot_tree(classifier)


# %%
reject_dataset.head()
reject_dataset.info()
for col in reject_dataset.columns:
    if reject_dataset[col].dtype == 'object':
        reject_dataset[col] = pd.Categorical(reject_dataset[col]).codes
# Split-out validation dataset
array = reject_dataset.values
Xr = array[:,1:]
Yr = array[:,0]
# Predicting the Test set results
y_r_pred = cart.predict(Xr)
result_r = pd.DataFrame((np.concatenate((y_r_pred.reshape(len(y_r_pred),1), Yr.reshape(len(Yr),1)),1)))
result_r
rejected_dataset = dataset[reject]
rejected_dataset
rejected_dataset['Predicted_Dispute_status'] = y_r_pred
rejected_dataset.to_excel('C:\\Reva\\BA06\\Capstone Projects\\output\\rejected_unbalanced.xlsx')


# %%
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes=10, criterion = 'entropy', random_state = 0)
classifier.fit(Xc_train, Yc_train)
import pydotplus as pydot
from sklearn import tree
tree.plot_tree(classifier)


# %%
import pydotplus as pydot 
with open("C:/Reva/BA06/Capstone Projects/output/classifier_8.txt", "w") as z:
    z = tree.export_graphviz(classifier, out_file=z)


# %%
# Predicting the Test set results
y_pred = classifier.predict(Xc_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Yc_test.reshape(len(Yc_test),1)),1))


# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Yc_test, y_pred)
print(cm)
accuracy_score(Yc_test, y_pred)


# %%
import pydotplus as pydot
from sklearn import tree
tree.plot_tree(classifier)


# %%
from sklearn.metrics import roc_curve, auc
# ROC Chart
fpr, tpr, th = roc_curve(Yc_test, predictions)
roc_auc = auc(fpr, tpr)
import matplotlib.pyplot as plt 
plt.title('ROCR CHART for Unbalanced Data')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# %%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# %%
oversample = SMOTE()
undersample = RandomUnderSampler()
steps = [('O', oversample), ('u', undersample)]
Pipeline = Pipeline(steps=steps)
Xc, Yc = oversample.fit_resample(Xc, Yc)


# %%
Xc_train, Xc_test, Yc_train, Yc_test = model_selection.train_test_split(Xc, Yc, test_size=0.15, random_state=7)


# %%
from collections import Counter
counter = Counter(Yc)
for k, v in counter.items():
    dist = v / len(Yc) * 100
    print(f'Class={k}, n={v} ({dist}%)')


# %%
balanced = pd.DataFrame(Yc)
balanced['Disputed_Status for Balanced Dataset'] = pd.DataFrame(Yc)
balanced


# %%
plt.figure(figsize=(5,5))
sns.countplot(x="Disputed_Status for Balanced Dataset", data=balanced)


# %%
# Test options and evaluation metric : Model 02
seed = 10
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=seed)
 cv_results = model_selection.cross_val_score(model, Xc_train, Yc_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)


# %%
# Make predictions for Decision Tree
cart = DecisionTreeClassifier()
cart.fit(Xc_train, Yc_train)
predictions = cart.predict(Xc_test)
print(accuracy_score(Yc_test, predictions))
print(confusion_matrix(Yc_test, predictions))
print(classification_report(Yc_test, predictions))


# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Yc_test, predictions)
accuracy_score(Yc_test, predictions)
label1 = ["Predicted 0", "Predicted 1"]
label2 = ["True 0", "True 1"]
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", xticklabels=label1, yticklabels=label2)
plt.show()


# %%
reject_balanced_dataset.head()
reject_balanced_dataset.info()
for col in reject_balanced_dataset.columns:
    if reject_balanced_dataset[col].dtype == 'object':
        reject_balanced_dataset[col] = pd.Categorical(reject_balanced_dataset[col]).codes
# Split-out validation dataset
array = reject_balanced_dataset.values
Xr = array[:,1:]
Yr = array[:,0]
# Predicting the Test set results
y_r_pred = cart.predict(Xr)
result_r = pd.DataFrame((np.concatenate((y_r_pred.reshape(len(y_r_pred),1), Yr.reshape(len(Yr),1)),1)))
result_r
rejected_dataset = dataset[reject]
rejected_dataset
rejected_dataset['Predicted_Dispute_status'] = y_r_pred
rejected_dataset.to_excel('C:\\Reva\\BA06\\Capstone Projects\\output\\rejected_balanced.xlsx')


# %%
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes=8, criterion = 'entropy', random_state = 0)
classifier.fit(Xc_train, Yc_train)


# %%
# Predicting the Test set results
y_pred = classifier.predict(Xc_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Yc_test.reshape(len(Yc_test),1)),1))


# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Yc_test, y_pred)
print(cm)
accuracy_score(Yc_test, y_pred)


# %%
# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_leaf_nodes=8, criterion = 'entropy', random_state = 0)
classifier.fit(Xc_train, Yc_train)
tree.plot_tree(classifier)


# %%
import pydotplus as pydot
from sklearn import tree
tree.plot_tree(classifier)


# %%
from sklearn.metrics import roc_curve, auc
# ROC Chart
fpr, tpr, th = roc_curve(Yc_test, predictions)
roc_auc = auc(fpr, tpr)
import matplotlib.pyplot as plt 
plt.title('ROCR CHART for Balanced Data')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f'% roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# %%
reject_dataset.head()
reject_dataset.info()


# %%
for col in reject_dataset.columns:
    if reject_dataset[col].dtype == 'object':
        reject_dataset[col] = pd.Categorical(reject_dataset[col]).codes


# %%
# Split-out validation dataset
array = reject_dataset.values
Xr = array[:,1:]
Yr = array[:,0]


# %%
# Predicting the Test set results
y_r_pred = cart.predict(Xr)
result_r = pd.DataFrame((np.concatenate((y_r_pred.reshape(len(y_r_pred),1), Yr.reshape(len(Yr),1)),1)))
result_r


