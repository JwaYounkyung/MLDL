from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(df_x):
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(df_x.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    return model

# 에포크가 끝날 때마다 점(.)을 출력해 훈련 진행 과정을 표시합니다
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def trainKeras():
    EPOCHS = 1000
    df_x = pd.read_csv('./preprocess/train_preprocess_1.csv', header=0)
    df_y = pd.read_csv('./preprocess/train_label_1.csv', header=0)

    df_y_time = df_y.iloc[:,1]
    df_y_spent = df_y.iloc[:,2]

    # patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model_time = build_model(df_x)
    history_time = model_time.fit(df_x, df_y_time, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    
    model_spent = build_model(df_x)
    history_spent = model_spent.fit(df_x, df_y_spent, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    model_time.save('./model/model_Keras_time.h5')
    model_spent.save('./model/model_Keras_spent.h5')

trainKeras()