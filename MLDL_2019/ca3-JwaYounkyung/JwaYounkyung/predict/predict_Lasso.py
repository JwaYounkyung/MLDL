from numpy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import pandas as pd

import pickle
from sklearn.externals import joblib

def predictskitLasso(num):

    df_x = pd.read_csv('./preprocess/test' + num + '_preprocess_1.csv', header=0)
    X = df_x.values

    model_Lasso_spent = joblib.load('./model/model_Lasso_spent.pkl') 
    predict_spent = model_Lasso_spent.predict(X)
    predict_spent = predict_spent.clip(min=0).reshape(-1,1)
    #print(predict_spent.shape)

    model_Lasso_time = joblib.load('./model/model_Lasso_time.pkl') 
    predict_time = model_Lasso_time.predict(X)
    predict_time = predict_time.clip(min=0)
    predict_time = predict_time.astype(int) #convert to int
    predict_time[predict_time > 64] = 64
    predict_time = predict_time.reshape(-1,1)
    #print(predict_time.shape)

    df_y = pd.read_csv('./preprocess/test'+ num + '_id_1.csv', header=0)
    Y = df_y.values
    acc_id = Y[:,0].reshape(-1,1)
    #print(acc_id.shape)

    predict = concatenate((acc_id, predict_time, predict_spent), axis=1) # array condatenate
    df_predict = pd.DataFrame(predict, columns=['acc_id', 'survival_time','amount_spent'])

    #save to csv
    df_predict.to_csv('./predict/test' + num + '_predict_Lasso.csv', index=False)
    print(num + 'sucess')

    
predictskitLasso('1')
predictskitLasso('2')