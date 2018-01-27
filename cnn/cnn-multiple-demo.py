from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from numpy import argmax

imageWidth = 100
imageHeight = 100
kernelSize = 3 # feature vector (3x3)
downscale = 2 # max pooling vector (2x2)
datasetPath = 'dataset'

# initialize the CNN
cnn = Sequential()
# add first CNN layer (feature detector + relu layer) and apply max pooling
cnn.add(Conv2D(32, kernelSize, input_shape = (imageWidth, imageHeight, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (downscale, downscale)))
# add second CNN layer (feature detector + relu layer) and apply max pooling
cnn.add(Conv2D(64, kernelSize, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (downscale, downscale)))
# apply flattening
cnn.add(Flatten())
# add conventional ANN to apply flattening layer
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 6, activation = 'softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# prepare training and test images using image generator
trainImages = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
testImages = ImageDataGenerator(rescale = 1./255)
# prepare training and test sets from folder
trainSet = trainImages.flow_from_directory(datasetPath + '/train', target_size = (imageWidth, imageHeight), batch_size = 128, class_mode = 'categorical')
testSet = testImages.flow_from_directory(datasetPath + '/test', target_size = (imageWidth, imageHeight), batch_size = 128, class_mode = 'categorical')

# execute the CNN
cnn.fit_generator(trainSet, steps_per_epoch = 1024, epochs = 2, validation_data = testSet, validation_steps = 128)

# single experiment
classList = list(trainSet.class_indices.keys())
for i in range(0, 6):
    validationFile = datasetPath + '/validate' + str(i) + '.jpg'
    validationImage = image.load_img(validationFile, target_size = (imageWidth, imageHeight))
    validationResult = cnn.predict(expand_dims(image.img_to_array(validationImage), axis = 0))
    evaluationResult = cnn.evaluate(expand_dims(image.img_to_array(validationImage), axis = 0), validationResult, verbose = 0)
    print('Fruit in:', validationFile, ':', classList[argmax(validationResult)].ljust(6), '[', round(evaluationResult[1] * 100), '% ]')
