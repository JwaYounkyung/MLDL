from numpy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import pandas as pd

import pickle
from sklearn.externals import joblib
    
def trainskitLasso():
    df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
    X = df_x.values

    df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)
    Y = df_y.values

    Y_time = Y[:,1].reshape(-1,1)
    Y_spent = Y[:,2].reshape(-1,1)

    model_Lasso_time = linear_model.Lasso(alpha=0.1)
    model_Lasso_time.fit(X, Y_time)

    pickle.dumps(model_Lasso_time)
    joblib.dump(model_Lasso_time, './model/model_Lasso_time.pkl')

    modelLasso_spent = linear_model.Lasso(alpha=0.1)
    modelLasso_spent.fit(X, Y_spent)

    pickle.dumps(modelLasso_spent)
    joblib.dump(modelLasso_spent, './model/model_Lasso_spent.pkl')

trainskitLasso()