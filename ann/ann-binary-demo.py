# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:17:49 2018

@author: SERGIO
"""

from pandas import read_csv

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense

# Import and prepare data
columnMinX = 104
columnMaxX = 126
columnY = 127
rowX = 3000
originalDataSet = read_csv('BGRI2011_1106_sample.csv')
originalX = originalDataSet.iloc[0:rowX, columnMinX:columnMaxX+1].values
originalY = originalDataSet.iloc[0:rowX, columnY].values

# Encode data if needed
#encoderX1 = LabelEncoder()
#originalX[:, <column to encode>] = encoderX1.fit_transform(originalX[:, <column to encode>])
#encoder = OneHotEncoder(categorical_features = [<column>])
#originalX = encoder.fit_transform(originalX).toarray()
#originalX = originalX[:, <column to remove>:]

# Train and test sets
trainX, testX, trainY, testY = train_test_split(originalX, originalY, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Create the ANN
classifier = Sequential()
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
classifier.fit(trainX, trainY, batch_size = 16, epochs = 128)

# Predict
predTest = classifier.predict(testX)
predTest = (predTest > 0.5)
cm = confusion_matrix(testY, predTest)

predX = originalDataSet.iloc[rowX:, columnMinX:columnMaxX+1].values
predY = classifier.predict(scaler.transform(predX))
predY = (predY > 0.5)
