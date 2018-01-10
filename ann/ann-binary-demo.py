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
columnMinIndVar = 104
columnMaxIndVar = 126
columnDepVar = 127
rowBaseline = 3000

originalDataSet = read_csv('BGRI2011_1106_sample.csv')
sourceIndVar = originalDataSet.iloc[0:rowBaseline, columnMinIndVar:columnMaxIndVar+1].values
sourceDepVar = originalDataSet.iloc[0:rowBaseline, columnDepVar].values
sourceToPredIndVar = originalDataSet.iloc[rowBaseline:, columnMinIndVar:columnMaxIndVar+1].values

# Encode data if needed (beware dummy variable trap)
#encoderIndVar1 = LabelEncoder()
#sourceIndVar[:, <column to encode>] = encoderIndVar1.fit_transform(sourceIndVar[:, <column to encode>])
#encoder = OneHotEncoder(categorical_features = [<column>])
#sourceIndVar = encoder.fit_transform(sourceIndVar).toarray()
#sourceIndVar = sourceIndVar[:, <column to remove>:]

# Split train, test and to predict data sets (apply feature scaling)
trainIndVar, testIndVar, trainDepVar, testDepVar = train_test_split(sourceIndVar, sourceDepVar, test_size = 0.2, random_state = 0)
scaler = StandardScaler()
trainIndVar = scaler.fit_transform(trainIndVar)
testIndVar = scaler.transform(testIndVar)
sourceToPredIndVar = scaler.transform(sourceToPredIndVar)

# Create the ANN
neurons = 12
classifier = Sequential()
classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))
classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dense(units = neurons, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
classifier.fit(trainIndVar, trainDepVar, batch_size = 16, epochs = 512)

# Check against test set
#testPredDepVar = classifier.predict(testIndVar)
#testPredDepVar = (testPredDepVar > 0.5)
#confusionMatrix = confusion_matrix(testDepVar, testPredDepVar)

# Predict
predDepVar = classifier.predict(sourceToPredIndVar)
predDepVar = (predDepVar > 0.5)
