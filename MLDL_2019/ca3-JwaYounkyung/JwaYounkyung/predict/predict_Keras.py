from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import load_model

from numpy import *

def predictKears(num):

    df_x = pd.read_csv('./preprocess/test' + num + '_preprocess_1.csv', header=0)

    model_Keras_spent = load_model('./model/model_Keras_spent.h5')
    predict_spent = model_Keras_spent.predict(df_x).flatten()
    predict_spent = predict_spent.clip(min=0).reshape(-1,1)

    model_Keras_time = load_model('./model/model_Keras_time.h5')
    predict_time = model_Keras_time.predict(df_x).flatten()
    predict_time = predict_time.clip(min=0)
    predict_time = predict_time.astype(int) #convert to int
    predict_time[predict_time > 64] = 64
    predict_time = predict_time.reshape(-1,1)

    df_y = pd.read_csv('./preprocess/test'+ num + '_id_1.csv', header=0)
    Y = df_y.values
    acc_id = Y[:,0].reshape(-1,1)
    print(acc_id.shape)

    predict = concatenate((acc_id, predict_time, predict_spent), axis=1) # array condatenate
    df_predict = pd.DataFrame(predict, columns=['acc_id', 'survival_time','amount_spent'])

    #save to csv
    df_predict.to_csv('./predict/test' + num + '_predict_Keras.csv', index=False)
    print(num + 'sucess')

    
predictKears('1')
predictKeras('2')

#keras version에 따른 에러 발생 가능 model을 만든 버전과 같은 keras버전을 사용해야함