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

def predictskitGradient(num):

    df_x = pd.read_csv('./preprocess/test' + num + '_preprocess_1.csv', header=0)
    X_test = df_x.values

    model_Grad_spent = joblib.load('./model/model_Grad_spent.pkl') 
    
    predict_spent = model_Grad_spent.predict(X_test)
    predict_spent = predict_spent.clip(min=0).reshape(-1,1)

    model_Grad_time = joblib.load('./model/model_Grad_time.pkl') 
    
    predict_time = model_Grad_time.predict(X_test)
    predict_time = predict_time.clip(min=0)
    predict_time = predict_time.astype(int) #convert to int
    predict_time[predict_time > 64] = 64
    predict_time = predict_time.reshape(-1,1)

    df_y = pd.read_csv('./preprocess/test'+ num + '_id_1.csv', header=0)
    Y = df_y.values
    acc_id = Y[:,0].reshape(-1,1)

    predict = concatenate((acc_id, predict_time, predict_spent), axis=1) # array condatenate
    df_predict = pd.DataFrame(predict, columns=['acc_id', 'survival_time','amount_spent'])

    #save to csv
    df_predict.to_csv('./predict/test' + num + '_predict_Grad.csv', index=False)
    print(num + 'sucess')

    
predictskitGradient('1')
predictskitGradient('2')