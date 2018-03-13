# data source: http://www.macrotrends.net/2532/corn-prices-historical-chart-data

from pandas import read_csv
from numpy import array, reshape, zeros
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

toPredNum = 10
timeSteps = 128
train = False
test = False

dataSet = read_csv('corn-oil-prices-1997-2017.csv')
trainingSet = dataSet.iloc[:len(dataSet)+1,1:3].values

scaler = MinMaxScaler(feature_range = (0, 1))
trainingSet = scaler.fit_transform(trainingSet)

xTrain, yTrain, xPred = [], [], []
for i in range(timeSteps,len(dataSet)+1):
    if(i<len(dataSet)-toPredNum+1):
        xTrain.append(trainingSet[i-timeSteps:i, 0:2])
        yTrain.append(trainingSet[i-1, 0])
    else:
        xPred.append(trainingSet[i-timeSteps:i, 0:2])
xTrain, yTrain, xPred = array(xTrain), array(yTrain), array(xPred)
xTrain = reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 2))

if train:
    lstm = Sequential()
    lstm.add(LSTM(units = 64, return_sequences = True, input_shape = (xTrain.shape[1], 2)))
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
    lstm.save('lstm-corn-1997-2017.h5')
else:
    lstm = load_model('lstm-corn-1997-2017.h5')

if test:
    xPred = reshape(xPred, (xPred.shape[0], xPred.shape[1], 2))
    yPred = lstm.predict(xPred)
    y = zeros((len(yPred),2))
    y[:,:-1] = yPred
    yPred = scaler.inverse_transform(y)[:,0:1]
else:
    yPred = []
    trainingSet[len(dataSet)-toPredNum: len(dataSet) + 1, 0] = 0
    for i in range(len(dataSet)-toPredNum, len(dataSet)):
        xPred = []
        xPred.append(trainingSet[i-timeSteps:i, 0:2])
        xPred = array(xPred)
        xPred = reshape(xPred, (xPred.shape[0], xPred.shape[1], 2))
        y = lstm.predict(xPred)[0,0]
        trainingSet[i, 0] = y
        y = reshape(array([y, 0]), (1,2))
        yPred.append(scaler.inverse_transform(y)[0,0])
    yPred = array(yPred)