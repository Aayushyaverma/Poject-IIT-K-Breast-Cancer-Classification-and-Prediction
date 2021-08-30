# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <h1>BREAST CANCER CLASIFICATION</h1>

# <h2>Importing All Liberaries</h2>

# Source of Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/download

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
plt.style.use('ggplot')

from scipy import stats
from warnings import filterwarnings

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel, RFE

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_curve

from sklearn.preprocessing import StandardScaler, RobustScaler

# Many libarary are imported below for ease  

# <h2>LOADING THE DATASET</h2>

datax = load_breast_cancer()
print (datax.feature_names)
print (datax.target_names)

df = pd.read_csv('C:/Users/Aayus/Python/AIML IIT/data.csv')

# <h2>PRE-PROCESSING AND ANALYSING

df.info()

# The dataset has 569 rows and 33 columns. The diagnosis column classifies tumor as 'M' for malignant and 'B' for benign. The last column 'Unnamed:32' has all Nans and will be removed

df.drop(df.columns[[-1, 0]], axis=1, inplace=True)
df.info()

df.describe()

df.columns

#Get a count of the number of 'M' & 'B' cells
df['diagnosis'].value_counts()

# About 59% values in the diagnosis column have been classified as 'M' ie Malignant.

#Visualize this count
sns.countplot(df['diagnosis'],label="Count")

#Look at the data types 
df.dtypes

# There are now 30 features we can visualize. We plot 10 features at a time. This will lead to 3 plots containing 10 features each. The means of all the features are plotted together, so are the standard errors and worst dimensionsAll the columns are numeric except the diagnosis column which has categorical data

# There are now 30 features we can visualize.
# We plot 10 features at a time. 
# This will lead to 3 plots containing 10 features each.
# The means of all the features are plotted together, so are the standard errors and worst dimensions

# y includes our labels and x includes our features
y = df.diagnosis                          # M or B 
list = ['diagnosis']
X = df.drop(list,axis = 1 )
X.head()

X.describe()

# first ten features
data_dia = y
data = X
data_std = (data - data.mean()) / (data.std())              # standardization
data = pd.concat([y,data_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

# For the texture_mean feature, median of the Malignant and Benign looks separated and away from each other, so it can be good for classification. However, in fractal_dimension_mean feature, median of the Malignant and Benign looks almost the same which might not be good for classification. smoothness_mean seems to have the highest range of values.

# Second ten features
data = pd.concat([y,data_std.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

# The medians for almost all Malignant or Benign don't vary much for all the features above except for maybe concave points_se and concavity_se. smoothness_se or symmetry_se have almost same distribution ie Malignant and Benign sections might not be well separated, making classification difficult! The shape of violin plot for area_se looks wraped. The distribution of data points for benign and laignant in area_se looks very different and varys the most.

# Last ten features
data = pd.concat([y,data_std.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

# area_worst look well separated, so it might be easier to use this feature for classification! Variance seems highest for fractal_dimension_worst. concavity_worst and concave_points_worst seem to have similar data distribution. 

# <h3>Checking the corelation of features

filterwarnings('ignore')
sns.jointplot(X.loc[:,'concavity_worst'], X.loc[:,'concave points_worst'], kind="reg", color="#ce1414").annotate(stats.pearsonr)

# concavity_worst and concave points_worst show high correlation of 0.86 and a significant p-value

#correlation map
f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)

# Compactness_mean, concavity_mean and concave points_mean are correlated with each other. Apart from these, radius_se, perimeter_se and area_se are correlated. radius_worst, perimeter_worst and area_worst are correlated. Compactness_worst, concavity_worst and concave points_worst. Compactness_se, concavity_se and concave points_se. texture_mean and texture_worst are correlated. area_worst and area_mean are correlated. radius_worst, perimeter_worst and area_worst with radius_mean, perimeter_mean and area_mean have a correlation of 1

# VISUALIZE some features via BOX plots and performed a t test to detect statistical significance

X.columns

plot_5 = sns.boxplot(x='diagnosis', y='texture_mean', data=df, showfliers=False)
plot_5.set_title("Graph of texture mean vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['texture_mean', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# The p value is significant (<0.01) so we can reject null hypothesis. The difference in means for texture_mean is statistically significant.

plot_5 = sns.boxplot(x='diagnosis', y='fractal_dimension_mean', data=df, showfliers=False)
plot_5.set_title("Graph of fractal dimension mean vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['fractal_dimension_mean', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# t statistic is negative so if there is a difference between the M and B samples, it will be in the negative direction, meaning M samples might have lesser means than B samples. However the value of t statistic is very small and p value > 0.01, this means we cannot reject null hypothesis. The difference in means for fractal dimension_mean samples of M and B tumors might not be statitiscally significant.

plot_5 = sns.boxplot(x='diagnosis', y='area_se', data=df, showfliers=False)
plot_5.set_title("Graph of area se vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['area_se', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# As expected from the boxplot, p-value is very small which indicates the difference in means for M and B sample is statistically significant.

plot_5 = sns.boxplot(x='diagnosis', y='concavity_se', data=df, showfliers=False)
plot_5.set_title("Graph of concave points se vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['concavity_se', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# p-value is small indicating statistical significance between the 2 samples.

plot_5 = sns.boxplot(x='diagnosis', y='radius_worst', data=df, showfliers=False)
plot_5.set_title("Graph of radius worst vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['radius_worst', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# p-value very small, so the difference in means is statistically significant

plot_5 = sns.boxplot(x='diagnosis', y='area_worst', data=df, showfliers=False)
plot_5.set_title("Graph of area worst vs diagnosis of tumor")

new_d = pd.DataFrame(data=df[['area_worst', 'diagnosis']])
new_d = new_d.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])

# Very small p-value (<0.01), statitically significant difference in means for M and B samples.

# <h3>VIF SCORES for all the FEATURES

# creating copy of series 
new = df.copy(deep=True)

new.columns

new = new.rename(columns= {'concave points_mean': 'concave_points_mean', 'concave points_se':'concave_points_se', 'concave points_worst':'concave_points_worst'})

new

new.isna().sum()

# get y and X dataframes based on this regression:
y_vif, X_vif = dmatrices('diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + smoothness_mean + compactness_mean + concavity_mean + concave_points_mean + symmetry_mean + fractal_dimension_mean + radius_se + texture_se + perimeter_se + area_se + smoothness_se +compactness_se + concavity_se + concave_points_se + symmetry_se +fractal_dimension_se + radius_worst + texture_worst +perimeter_worst + area_worst + smoothness_worst +compactness_worst + concavity_worst + concave_points_worst + symmetry_worst + fractal_dimension_worst', new, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif["features"] = X_vif.columns

vif.round(1)

# The VIF scores are extremely high for a large number of features indicating multicollinearity. Multicollinearity makes it hard to assess the relative importance of independent variables, but it does not affect the usefulness of the regression equation for prediction. Even when multicollinearity is great, the least-squares regression equation can be highly predictive. We are only interested in prediction, multicollinearity is not a problem.

# ELIMINATING HIGHLY CORELATED FEATURES

# y includes our labels and x includes our features
y = df.diagnosis                          # M or B 
list = ['diagnosis']
X = df.drop(list,axis = 1 )
X.head()

# +
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# -

to_drop

# Drop features 
X = X.drop(X[to_drop], axis=1)
X.columns

# We need to find the optimal number of features for best classification results and the best features for classification too

# Transform categorical value of diagnosis column using LabelEncoder

y

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y= labelencoder_Y.fit_transform(y)
print(labelencoder_Y.fit_transform(y))

# <h2>Feature Selection</h2>

# Insert noise in dataset to check how feature selection performs

np.random.seed(100)
E = np.random.uniform(0, 1, size=(len(X), 15))
X = np.hstack((X, E))
print(X.shape)

# train test split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

# Try Univariate feature selection: chi2 test

sel_chi2 = SelectKBest(chi2, k=10)    # select 4 features
X_train_chi2 = sel_chi2.fit_transform(X_train, y_train)
print(sel_chi2.get_support())

# f test

sel_f = SelectKBest(f_classif, k=10)
X_train_f = sel_f.fit_transform(X_train, y_train)
print(sel_f.get_support())

# mutual_info_classif test

sel_mutual = SelectKBest(mutual_info_classif, k=10)
X_train_mutual = sel_mutual.fit_transform(X_train, y_train)
print(sel_mutual.get_support())

# SelectFromMode: L1 based feature selection

model_logistic = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=10000, penalty='l1')
sel_model_logistic = SelectFromModel(estimator=model_logistic)
X_train_sfm_l1 = sel_model_logistic.fit_transform(X_train, y_train)
print(sel_model_logistic.get_support())

# Does not work. Includes noise in features.

# RFE on logistic regression

model_logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
sel_rfe_logistic = RFE(estimator=model_logistic, n_features_to_select=10, step=1)
X_train_rfe_logistic = sel_rfe_logistic.fit_transform(X_train, y_train)
print(sel_rfe_logistic.get_support())
print(sel_rfe_logistic.ranking_)

# RFE on random Forest

model_tree = RandomForestClassifier(random_state=100, n_estimators=50)
sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select=10, step=1)
X_train_rfe_tree = sel_rfe_tree.fit_transform(X_train, y_train)
print(sel_rfe_tree.get_support())
print(sel_rfe_tree.ranking_)

# <h2>Before feature selection

model_logistic = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=10000)
model_logistic.fit(X_train, y_train)
predict = model_logistic.predict(X_test)
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

# <h2>After feature selection

model_logistic = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=17)
model_logistic.fit(X_train_chi2, y_train)

X_test_chi2 = sel_chi2.transform(X_test)
print(X_test.shape)
print(X_test_chi2.shape)

predict = model_logistic.predict(X_test_chi2)
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))

# +
# Create first pipeline for base without reducing features.
ftwo_scorer = make_scorer(fbeta_score, beta=2)
# Create logistic regression
#logistic = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.arange(0, 1, 0.001)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(model_logistic, hyperparameters, cv=5, scoring=ftwo_scorer, verbose=0)
# -

# Fit grid search
best_model = clf.fit(X_train_chi2, y_train)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

predictions = best_model.predict(X_test_chi2)
print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# <h3>Custom Threholding to increase recall

y_scores = best_model.predict_proba(X_test_chi2)[:, 1]

p, r, thresholds = precision_recall_curve(y_test, y_scores)


# +
def adjusted_classes(y_scores, t):

    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):

    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    print(classification_report(y_test, y_pred_adj))
# -

precision_recall_threshold(p, r, thresholds, 0.40)

list = ['diagnosis']
X = df.drop(list,axis = 1 )
X = X.drop(X[to_drop], axis=1)
X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, stratify=y, random_state = 17)

#Feature Scaling
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
from sklearn.preprocessing import LabelBinarizer
df = pd.read_csv('data.csv')
# by default majority class (benign) will be negative
lb = LabelBinarizer()
df['diagnosis'] = lb.fit_transform(df['diagnosis'].values)
targets = df['diagnosis']

df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, targets, stratify=targets)
'''


# <h3>Train model in Logistic Regression, KNeighborsClassifier, SVM, GaussianNB, Decision Tree and Random Forest

# Define a function which trains models
def models(X_train,y_train):
    
  
  #Using Logistic Regression 
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, y_train)

  #Using SVC linear
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0)
    svc_lin.fit(X_train, y_train)

  #Using SVC rbf
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    svc_rbf.fit(X_train, y_train)

  #Using DecisionTreeClassifier 
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, y_train)
  
  #print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, y_train))
    #print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, y_train))
    print('[1]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, y_train))
    print('[2]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, y_train))
    #print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, y_train))
    print('[3]Decision Tree Classifier Training Accuracy:', tree.score(X_train, y_train))
    print('[4]Random Forest Classifier Training Accuracy:', forest.score(X_train, y_train))
  
    return log, svc_lin, svc_rbf, tree, forest

model = models(X_train,y_train)

# <h2>Confusion Matrix

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    
    cm = confusion_matrix(y_test, model[i].predict(X_test))
  
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
  
    print(cm)
    print('Model[{}] Testing Accuracy = "{}"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
    print()# Print a new line

# +
#Show other ways to get the classification accuracy & other metrics 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print('Model ',i)
  #Check precision, recall, f1-score
    print(classification_report(y_test, model[i].predict(X_test)))
  #Another way to get the models accuracy on the test data
    print(accuracy_score(y_test, model[i].predict(X_test)))
    print()#Print a new line
# -

# <h4>From all the models trained and tested above, Random Forest Classifier gives us the best accuracy at 0.986 on the test set. However it seems to make a few wrong predictions for patients who have cancer and those who don't. SVM also performs well with test accuracy of 0.96

# We now choose the SVM and Random Forest model for hyper parameter tuning which might improve its performance further, and check with cross validation. We also want to know which and how many important features to include from amongst the 32 features for optimal model performance.

# <h2>Grid Search on Logistic Regression

# +
# Create first pipeline for base without reducing features.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
# Create logistic regression
logistic = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.arange(0, 1, 0.001)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, scoring=ftwo_scorer, verbose=0)
# -

# Fit grid search
best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

predictions = best_model.predict(X_test)
print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Custom Thresholding to increase recall

y_scores = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_test, y_scores)


# +
def adjusted_classes(y_scores, t):
    
    return [1 if y >= t else 0 for y in y_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'], 
                       index=['neg', 'pos']))
    print(classification_report(y_test, y_pred_adj))


# -

precision_recall_threshold(p, r, thresholds, 0.42)


# After thresholding the number of FNs reduce to 1

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
  
    plt.figure(figsize=(8, 8))
    plt.title("Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.axvline(x=.42, color='black')
    plt.text(.39,.50,'Optimal Threshold for best Recall',rotation=90)
    plt.ylabel("Recall Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

# +
from sklearn import metrics
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = best_model.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print(metrics.auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.show()
# -

# <h3>Grid Search on Decision tree classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dt= DecisionTreeClassifier(random_state=17)

# +
# Define the grid of hyperparameters 'params_dt'
params_dt = {'max_depth': [3, 4, 5, 6],'min_samples_leaf': [0.04, 0.06, 0.08],'max_features': [0.2, 0.4,0.6, 0.8]}

# Instantiate a 10-fold CV grid search object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring= ftwo_scorer, cv=10, n_jobs=-1)

# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)
# -

# Extract best hyperparameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyerparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy', best_CV_score)

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

predictions = best_model.predict(X_test)

print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# <h3>Custom Threholding to increase recall

y_scores = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_test, y_scores)

precision_recall_threshold(p, r, thresholds, 0.30)

# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

# +
from sklearn import metrics
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = best_model.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print(metrics.auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decison Tree')
plt.show()
# -

# <h2>Grid Search on SVC

# +
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)

c_values = np.arange(0, 1, 0.001)
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(random_state=0)
kfold = KFold(n_splits=5, random_state=None)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=ftwo_scorer, cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# -

best_model = grid_result.best_estimator_

best_model.fit(X_train, y_train)

# estimate accuracy on test dataset
predictions = best_model.predict(X_test)

print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

# SVM is misclassifying 10 cases amongst the 230 test group.

# <h2>Custom Thresholding to increase recall

y_scores = best_model.decision_function(X_test)

p, r, thresholds = precision_recall_curve(y_test, y_scores)

precision_recall_threshold(p, r, thresholds, 0.05)


# +

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
 
 plt.figure(figsize=(8, 8))
 plt.title("Recall Scores as a function of the decision threshold")
 plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
 plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
 plt.axvline(x=.42, color='black')
 plt.text(.39,.50,'Optimal Threshold for best Recall',rotation=90)
 plt.ylabel("Recall Score")
 plt.xlabel("Decision Threshold")
 plt.legend(loc='best')
# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

# +
from sklearn import metrics
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
#y_pred_prob = best_model.(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
print(metrics.auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVC')
plt.show()
# -

# <h2>Recursive Feature Elimination for Random Forest Classifier with cross validation

# +
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
ftwo_scorer = make_scorer(fbeta_score, beta=2)
# split data train 70 % and test 30 %
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring=ftwo_scorer)   #5-fold cross-validation
rfecv = rfecv.fit(X, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X.columns[rfecv.support_])
# -

# <h3>Include only the best features

#to_drop = ['symmetry_mean', 'smoothness_mean', 'symmetry_se', 'texture_se', 'fractal_dimension_worst']
#X_train_new = X_train.drop(X[to_drop], axis=1)
#X_new.columns
X_new = X[['radius_mean', 'texture_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'radius_se', 'concavity_se',
       'fractal_dimension_se', 'texture_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst']]

# <h3>Train Test Split again

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, stratify=y, random_state = 17)

# Scale the dataset Again

#Feature Scaling
from sklearn.preprocessing import StandardScaler, RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Run random forest Again

forest = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 17)
rf = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print('Accuracy with Scaling: {}'.format(rf.score(X_test, y_test)))

confusion_matrix(y_test, y_pred)

# <h2>Grid Search on Random Forest

# +
from sklearn.metrics import precision_score, recall_score, accuracy_score
clf = RandomForestClassifier(n_jobs=-1)

param_grid = {
    'min_samples_split': [3, 5, 10], 
    'n_estimators' : [100, 300],
    'max_depth': [3, 5, 15, 25],
    'max_features': [3, 5, 10, 20]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    #'accuracy_score': make_scorer(accuracy_score)
}
# -

from sklearn.model_selection import StratifiedKFold
def grid_search_wrapper(refit_score='recall_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {}:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


grid_search_clf = grid_search_wrapper(refit_score='recall_score')

# <h3>Custom Threholding to increase recall

y_scores = grid_search_clf.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_test, y_scores)

precision_recall_threshold(p, r, thresholds, 0.43)

# use the same p, r, thresholds that were previously calculated
plot_precision_recall_vs_threshold(p, r, thresholds)

# +
from sklearn import metrics
from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = grid_search_clf.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
print(metrics.auc(fpr, tpr))
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.show()

# +
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(X_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
