# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:43:50 2018

@author: SERGIO
"""
# data source: http://www.macrotrends.net/2532/corn-prices-historical-chart-data

from pandas import read_csv, concat
from numpy import array, reshape
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

dataSet = read_csv('corn-prices-historical-chart-data-train.csv')
trainingSet = dataSet.iloc[:, 1:2].values
#trainingSet = dataSet.iloc[32:, 1:2].values

scaler = MinMaxScaler(feature_range = (0, 1))
trainingSetScale = scaler.fit_transform(trainingSet)

xTrain, yTrain = [], []
for i in range(128, 2048 + 32):
    xTrain.append(trainingSetScale[i-128:i, 0])
    yTrain.append(trainingSetScale[i, 0])
xTrain, yTrain = array(xTrain), array(yTrain)
xTrain = reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

lstm = Sequential()
lstm.add(LSTM(units = 64, return_sequences = True, input_shape = (xTrain.shape[1], 1)))
lstm.add(Dropout(0.2))
lstm.add(LSTM(units = 64, return_sequences = True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(units = 64, return_sequences = True))
lstm.add(Dropout(0.2))
lstm.add(LSTM(units = 64))
lstm.add(Dropout(0.2))
lstm.add(Dense(units = 1))

lstm.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
lstm.fit(xTrain, yTrain, epochs = 128, batch_size = 32)

dataSetToPred = read_csv('corn-prices-historical-chart-data-pred.csv')
dataSetReal = dataSetToPred.iloc[:, 1:2].values
#dataSetReal = dataSet.iloc[0:32, 1:2].values
#toPred = scaler.transform(dataSetReal)
#xPred = []
#for i in range(32, 128 + 32):
#    xPred.append(toPred[i-128:i, 0])
#xPred = array(xPred)
#xPred = reshape(xPred, (xPred.shape[0], xPred.shape[1], 1))
#yPred = scaler.inverse_transform(lstm.predict(xPred))

dataSetFull = concat((dataSet['values'], dataSetToPred['values']), axis = 0)
toPred = dataSetFull[len(dataSetFull) - len(dataSetToPred) - 128:].values
toPred = scaler.transform(toPred.reshape(-1,1))
xPred = []
for i in range(128, 128 + len(dataSetToPred)):
    xPred.append(toPred[i-128:i, 0])
xPred = array(xPred)
xPred = reshape(xPred, (xPred.shape[0], xPred.shape[1], 1))
yPred = scaler.inverse_transform(lstm.predict(xPred))