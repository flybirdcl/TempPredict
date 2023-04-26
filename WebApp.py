#!/usr/bin/env python
# coding: utf-8

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Import Meteostat library and dependencies
from datetime import datetime
from meteostat import Hourly
from meteostat import Point
from meteostat import Stations


import gradio as gr

def predict(text):
  def normalize(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return (data - data_mean) / data_std

  coor = text.split()

  sd = Stations().nearby(float(coor[0]),float(coor[1]))
  sd = sd.fetch(1)


  # data units
  # ① Humidity (%), ② Pressure (hPa), ③ Temperature (K), 4) Wind Speed (m/s).

  # sd coordinates: 32.7157° N, 117.1611° W


  # Set time period
  start = datetime(2021, 1, 1)
  end = datetime(2023, 4, 23, 23, 59)

  # Get hourly data
  data = Hourly(sd, start, end)
  data = data.fetch()

  # Print DataFrame
  features = data[['temp','pres','rhum','wspd']]


  def f(x):
      x = x + 273.15
      return float(x)


  features['temp'] = features['temp'].map(lambda a: a+273.15)

  features




  training_size = int ( 0.8 * features.shape[0])  

  dataset=features.values
  data_mean = dataset[:training_size].mean(axis=0)
  data_std = dataset[:training_size].std(axis=0)
  dataset = (dataset-data_mean)/data_std

  dataset.shape


  features = normalize(features.values)
  features = pd.DataFrame(features)
  features





  past_history = 72
  future_target = 24
  STEP = 1



  def multivariate_data(dataset, target, start_index, end_index, history_size,
                        target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      data.append(dataset[indices])

      if single_step:
        labels.append(target[i+target_size])
      else:
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)



  x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                  training_size, past_history,
                                                  future_target, STEP)
  x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                              training_size, None, past_history,
                                              future_target, STEP)

  print(x_train_multi.shape)
  print(y_train_multi.shape)




  print ('Single window of past history : {}'.format(x_train_multi[0].shape))
  print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))




  BATCH_SIZE = 256
  BUFFER_SIZE = 10000

  train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
  train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat() # .repeat()

  val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
  val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat() # .repeat()




  def multi_step_plot(history, true_future, prediction):
    fig = plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
              label='Predicted Future')
    plt.legend(loc='upper left')
    plt.xlabel('Time (hour)', fontsize=12)
    plt.ylabel('Temperature (Celcius)', fontsize=12)
    plt.title('Temp vs. time')
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')
    return fig
    # plt.show()


  multi_step_model = tf.keras.models.Sequential()
  multi_step_model.add(tf.keras.layers.LSTM(32,
                                            return_sequences=True,
                                            input_shape=x_train_multi.shape[-2:]))
  multi_step_model.add(tf.keras.layers.LSTM(16, activation='tanh'))
  multi_step_model.add(tf.keras.layers.Dense(future_target))

  multi_step_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss="mse")




  EVALUATION_INTERVAL = 250
  EPOCHS = 6

  multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_multi,
                                            validation_steps=50)


  # Plot training loss and validation
  fig1=plt.figure()
  plt.plot(multi_step_history.history['loss'])
  plt.plot(multi_step_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  # fig1.show()




  def create_time_steps(length):
    return list(range(-length, 0))



  def denormalize(a):
      #     data_mean = dataset[:training_size].mean(axis=0)
      #     data_std = dataset[:training_size].std(axis=0)
      #     dataset = (dataset-data_mean)/data_std
      return a*data_std[0]+data_mean[0]-273.15




  denorm = np.vectorize(denormalize)



  for x, y in val_data_multi.take(5):
    return multi_step_plot(denorm(x[0]), denorm(y[0]), denorm(multi_step_model.predict(x)[0])), fig1



  # return multi_step_plot(denorm(x[0]), denorm(y[0]), denorm(multi_step_model.predict(x)[0]))
  

iface = gr.Interface(
  fn=predict, 
  inputs=gr.Textbox(lines=5, label="City Coordinates temperature predictor", placeholder="Put coordinate of city you're interested in predicting the temp and make sure to have space in between coordinate: i.e. 32.7338 117.1933"),
  examples = [['32.7338 117.1933'], ['29.4252 98.4946']],  
  outputs=['plot','plot'],
  cache_examples=True,
  title="Temperature Predictor for given coordinates"
)

iface.queue().launch(share=True) 





