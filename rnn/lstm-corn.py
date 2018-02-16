# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:43:50 2018

@author: SERGIO
"""
# data source: http://www.macrotrends.net/2532/corn-prices-historical-chart-data

from pandas import read_csv
from numpy import array, reshape
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

toPredNum = 29
toTrainNum = 2520
timeSteps = 128

dataSet = read_csv('corn-prices-historical-chart-data.csv')
trainingSet = dataSet.iloc[len(dataSet)-toTrainNum-toPredNum:len(dataSet)-toPredNum,1:2].values
predSet = dataSet.iloc[len(dataSet)-(toPredNum+timeSteps):len(dataSet)+1, 1:2].values

toTrainTh = len(dataSet) - (toTrainNum + toPredNum + 1)
toPredTh = len(dataSet) - (toPredNum + 1)

scaler = MinMaxScaler(feature_range = (0, 1))
trainingSetScale = scaler.fit_transform(trainingSet)

xTrain, yTrain = [], []
for i in range(timeSteps+toTrainTh,len(dataSet)-toPredNum):
    xTrain.append(trainingSetScale[i-(timeSteps+toTrainTh):i-toTrainTh, 0])
    yTrain.append(trainingSetScale[i-(toTrainTh+1), 0])
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

predSet = predSet.reshape(-1,1)
predSet = scaler.transform(predSet)
xPred = []
for i in range(timeSteps, timeSteps + toPredNum):
    xPred.append(predSet[i-timeSteps :i, 0])
xPred = array(xPred)
xPred = reshape(xPred, (xPred.shape[0], xPred.shape[1], 1))
yPred = scaler.inverse_transform(lstm.predict(xPred))
