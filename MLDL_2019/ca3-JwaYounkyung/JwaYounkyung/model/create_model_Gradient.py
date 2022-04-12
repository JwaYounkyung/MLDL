from numpy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import pandas as pd

import pickle
from sklearn.externals import joblib

import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
    
def trainskitGradient():
    df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
    X_train = df_x.values

    df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)
    Y = df_y.values

    Y_time = Y[:,1]#.reshape(-1,1)
    Y_spent = Y[:,2]#.reshape(-1,1)

    model_Grad_time = GradientBoostingRegressor(random_state=21, n_estimators=400)
    model_Grad_time.fit(X_train, Y_time)

    pickle.dumps(model_Grad_time)
    joblib.dump(model_Grad_time, './model/model_Grad_time.pkl')

    model_Grad_spent = GradientBoostingRegressor(random_state=21, n_estimators=50)
    model_Grad_spent.fit(X_train, Y_spent)

    pickle.dumps(model_Grad_spent)
    print("hi")
    joblib.dump(model_Grad_spent, './model/model_Grad_spent.pkl')

trainskitGradient()