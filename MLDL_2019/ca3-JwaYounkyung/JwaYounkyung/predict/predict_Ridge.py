from numpy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import pandas as pd

import pickle
from sklearn.externals import joblib

def predictskitRidge(num):

    df_x = pd.read_csv('./preprocess/test' + num + '_preprocess_1.csv', header=0)
    X = df_x.values

    model_Ridge_spent = joblib.load('./model/model_Ridge_spent.pkl') 
    predict_spent = model_Ridge_spent.predict(X)
    predict_spent = predict_spent.clip(min=0)

    model_Ridge_time = joblib.load('./model/model_Ridge_time.pkl') 
    predict_time = model_Ridge_time.predict(X)
    predict_time = predict_time.clip(min=0)
    predict_time = predict_time.astype(int) #convert to int
    predict_time[predict_time > 64] = 64

    df_y = pd.read_csv('./preprocess/test'+ num + '_id_1.csv', header=0)
    Y = df_y.values
    acc_id = Y[:,0].reshape(-1,1)

    predict = concatenate((acc_id, predict_time, predict_spent), axis=1) # array condatenate
    df_predict = pd.DataFrame(predict, columns=['acc_id', 'survival_time','amount_spent'])

    #save to csv
    df_predict.to_csv('./predict/test' + num + '_predict_Ridge.csv', index=False)
    print(num + 'sucess')

    
predictskitRidge('1')
predictskitRidge('2')