from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from numpy import expand_dims

imageWidth = 64
imageHeight = 64
stride = 3
downscale = 2
datasetPath = 'dataset-binary'

# initialize the CNN
cnn = Sequential()
# add first CNN layer and apply max pooling
cnn.add(Conv2D(32, (stride, stride), input_shape = (imageWidth, imageHeight, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (downscale, downscale)))
# add second CNN layer and apply max pooling
cnn.add(Conv2D(32, (stride, stride), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (downscale, downscale)))
# apply flattening
cnn.add(Flatten())
# add conventional ANN to apply flattening layer
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# prepare training and test images using image generator
trainImages = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
testImages = ImageDataGenerator(rescale = 1./255)
# prepare training and test sets from folder
trainSet = trainImages.flow_from_directory(datasetPath + '/train', target_size = (imageWidth, imageHeight), batch_size = 32, class_mode = 'binary')
testSet = testImages.flow_from_directory(datasetPath + '/test', target_size = (imageWidth, imageHeight), batch_size = 32, class_mode = 'binary')

# execute the CNN
cnn.fit_generator(trainSet, steps_per_epoch = 500, epochs = 2, validation_data = testSet, validation_steps = 250)

# single experiment
validationFile = datasetPath + '/validate2.jpg'
validationImage = image.load_img(validationFile, target_size = (imageWidth, imageHeight))
validationImage = expand_dims(image.img_to_array(validationImage), axis = 0)
validationResult = cnn.predict(validationImage)
print('Fruit in: ' + validationFile + ': ' + list(trainSet.class_indices.keys())[list(trainSet.class_indices.values()).index(validationResult[0][0])])
