from numpy import *

from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
    
def findEstimator():
    #find the good estimator of GradientBoostingRegressor
    df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
    X_train = df_x.values

    df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)
    Y = df_y.values

    Y_time = Y[:,1].reshape(-1,1)
    Y_spent = Y[:,2].reshape(-1,1)

    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    param_grid = dict(n_estimators=array([50,100,200,300,400]))
    model = GradientBoostingRegressor(random_state=21)
    kfold = KFold(n_splits=10, random_state=21)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid.fit(rescaledX, Y_spent)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


findEstimator()