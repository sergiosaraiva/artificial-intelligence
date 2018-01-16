from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from numpy import expand_dims

# initialize the CNN
cnn = Sequential()
# add first CNN layer + max pooling
cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
# add second CNN layer + max pooling
cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
# apply flattening
cnn.add(Flatten())
# add conventional ann to apply flatening layer
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# prepare training and test images using image generator
trainImages = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
testImages = ImageDataGenerator(rescale = 1./255)
# prepare training and test sets from folder
trainSet = trainImages.flow_from_directory('dataset/train', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
testSet = testImages.flow_from_directory('dataset/test', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# execute the cnn
cnn.fit_generator(trainSet, steps_per_epoch = 2000, epochs = 4, validation_data = testSet, validation_steps = 500)

# single experiment
image = image.load_img('dataset/validade1.jpg', target_size = (64, 64))
image = image.img_to_array(image)
image = expand_dims(image, axis = 0)
result = cnn.predict(image)
trainSet.class_indices
if result[0][0] == 1:
    print('apple')
else:
    print('orange')

image = image.load_img('dataset/validade2.jpg', target_size = (64, 64))
image = image.img_to_array(image)
image = expand_dims(image, axis = 0)
result = cnn.predict(image)
trainSet.class_indices
if result[0][0] == 1:
    print('apple')
else:
    print('orange')
