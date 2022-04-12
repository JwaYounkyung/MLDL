print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

# #############################################################################
# Load data
df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
#X_train = df_x.values

df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)
#y_train = df_y.values
df_y_time = df_y.iloc[:,1]
df_y_spent = df_y.iloc[:,2]

df_x_train = df_x.sample(frac=0.8,random_state=0)
df_x_test = df_x.drop(df_x_train.index)
X_train = df_x_test.values
X_test = df_x_test.values

df_y_train = df_y_time.sample(frac=0.8,random_state=0) #change this 
df_y_test = df_y_time.drop(df_y_train.index)
y_train = df_y_test.values
y_test = df_y_test.values

# #############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(6, 6))
#plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
#plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()


# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, array(df_x.columns)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
