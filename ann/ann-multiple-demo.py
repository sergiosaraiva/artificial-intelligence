# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:17:49 2018
@author: SERGIO
"""

from pandas import read_csv
from numpy import argmax

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

# Import and prepare data
columnIndVar = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 77, 78, 79, 80, 81, 82, 83, 84, 85]
columnDepVar = [127]
rowBaseline = 3000

originalDataSet = read_csv('BGRI2011_1106_multiple.csv')
sourceIndVar = originalDataSet.iloc[0: rowBaseline, columnIndVar].values
sourceDepVar = originalDataSet.iloc[0: rowBaseline, columnDepVar].values.astype(str)
sourcePredIndVar = originalDataSet.iloc[rowBaseline: , columnIndVar].values 

# Encode data if needed (beware dummy variable trap)
encoderDepVar1 = LabelEncoder()
sourceDepVar = encoderDepVar1.fit_transform(sourceDepVar[:])
encoder = OneHotEncoder(categorical_features = [0])
sourceDepVar = encoder.fit_transform(sourceDepVar.reshape(-1, 1)).toarray()

# Split train, test (20%) and predict data sets (apply feature scaling)
trainIndVar, testIndVar, trainDepVar, testDepVar = train_test_split(sourceIndVar, sourceDepVar, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
trainIndVar = scaler.fit_transform(trainIndVar)
testIndVar = scaler.transform(testIndVar)
sourcePredIndVar = scaler.transform(sourcePredIndVar)

# Create the ANN: neurons = 1/2 independent variables (not exact science...): relu + sigmoid (last hidden layer: percentage)
neurons = int(round(len(sourceIndVar[0]) / 2, 0))
classifier = Sequential()
classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(sourceIndVar[0])))
classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train the ANN
classifier.fit(trainIndVar, trainDepVar, batch_size = 32, epochs = 512)

# Check against test set and do some predictions
testPredDepVar = encoderDepVar1.classes_[argmax(classifier.predict(testIndVar), axis = 1)]

predDepVar = encoderDepVar1.classes_[argmax(classifier.predict(sourcePredIndVar), axis = 1)]
