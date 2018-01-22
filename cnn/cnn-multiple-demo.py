from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

imageWidth = 64
imageHeight = 64
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
trainSet = trainImages.flow_from_directory(datasetPath + '/train', target_size = (imageWidth, imageHeight), batch_size = 64, class_mode = 'categorical')
testSet = testImages.flow_from_directory(datasetPath + '/test', target_size = (imageWidth, imageHeight), batch_size = 64, class_mode = 'categorical')

# execute the CNN
cnn.fit_generator(trainSet, steps_per_epoch = 512, epochs = 2, validation_data = testSet, validation_steps = 128)

# single experiment. can use predict_generator instead
for i in range(0, 5):
    validationFile = datasetPath + '/validate' + str(i) + '.jpg'
    validationImage = image.load_img(validationFile, target_size = (imageWidth, imageHeight))
    validationImage = expand_dims(image.img_to_array(validationImage), axis = 0)
    validationResult = cnn.predict(validationImage)
    highest = 0
    for i in range(len(validationResult[0])):    
        if(validationResult[0][i] > highest):
            highest = i
    print('Fruit in: ' + validationFile + ': ' + list(trainSet.class_indices.keys())[list(trainSet.class_indices.values()).index(highest)])