import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import time

from xgboost import XGBClassifier

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('BurstCollapse.csv')
# Only extract the columns we need
df = df[['Pass/Fail', 'Material', 'Yield Strength (ksi)', 'OD (in)', 'Thickness (in)', 'PE (psi)', 'PI (psi)', 'T (C)']]
df.head()



df['Material'] = LabelEncoder().fit_transform(df['Material']) # Encode string to float
df['T (C)'] = LabelEncoder().fit_transform(df['T (C)']) # Encode RF to int

y, x = df.values[:, 0], df.values[:, 1:len(df.columns)] # y is P/F, x is other stuff
feature_x = []
for i in range(1, len(df.columns)):
        feature_x.append(df.keys()[i])
    
print(feature_x)
print(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.3, random_state = 100)

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#Calculate importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
importances = sorted(importances, reverse=True)

counter = 0
print("\n")
feature_p = []
for index in indices:
    print("Rank {} importance: {}".format(counter + 1, feature_x[index]))
    feature_p.append(feature_x[index])
    counter += 1
    
# Plot feature importance
plt.figure(figsize=(12,6))
plt.title("Feature importances")
plt.bar(feature_p, importances, align="center")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()

# Uncomment below to see the training accuracy of a random forest classifier (similar performance to XGBoost for smaller datasets). The main reason why we use 
# random boost here is to identify feature importance to see if it matches correctly with engineering intuition. Comparisons are still included, but the main thing we
# will use is still XGBoost.

# acc = round(accuracy_score(y_test, y_pred) * 100, 2) 
# acc1 = accuracy_score(y_test, y_pred, normalize=False)
# print("Correctly classified: {} / {} \nTraining accuracy: {}%".format(acc1, len(y_pred), acc))

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []

models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('kNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('ADA', AdaBoostClassifier(DecisionTreeClassifier(criterion='gini'))))
models.append(('NB', GaussianNB()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('MLP', MLPClassifier())) ## Multi-layer Perceptron neural network

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
print("Algorithm | Accuracy | Standard Deviation | Training Time\n")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    start_time = time.time()
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print("{}: {}% ({}) | {} sec".format(name, round(cv_results.mean() * 100, 2), round(cv_results.std() * 100, 2),  round(time.time() - start_time, 2)))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

