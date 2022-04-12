from numpy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import pandas as pd

import pickle
from sklearn.externals import joblib
    
def trainskitRidge():
    df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
    X = df_x.values

    df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)
    Y = df_y.values

    Y_time = Y[:,1].reshape(-1,1)
    Y_spent = Y[:,2].reshape(-1,1)

    modelRidge_time = linear_model.Ridge(alpha=5)
    modelRidge_time.fit(X, Y_time)

    pickle.dumps(modelRidge_time)
    joblib.dump(modelRidge_time, './model/model_Ridge_time.pkl')

    modelRidge_spent = linear_model.Ridge(alpha=1)
    modelRidge_spent.fit(X, Y_spent)

    pickle.dumps(modelRidge_spent)
    joblib.dump(modelRidge_spent, './model/model_Ridge_spent.pkl')

trainskitRidge()