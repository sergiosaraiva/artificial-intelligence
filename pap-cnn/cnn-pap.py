from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims, argmax

imgW, imgH = 100, 100
kernel, downscale = 3, 2
path = 'dataset'

trainImgs = ImageDataGenerator(
        rescale = 1./255, shear_range = 0.2, 
        zoom_range = 0.2, horizontal_flip = True)
testImgs = ImageDataGenerator(rescale = 1./255)

train = trainImgs.flow_from_directory(
        path + '/train', target_size = (imgW, imgH),
        batch_size = 32, class_mode = 'categorical')
test = testImgs.flow_from_directory(
        path + '/test', target_size = (imgW, imgH),
        batch_size = 32, class_mode = 'categorical')

classes = list(train.class_indices.keys())

cnn = Sequential()

cnn.add(Conv2D(
        32, kernel, input_shape = (imgW, imgH, 3),
        activation = 'relu'))
cnn.add(MaxPooling2D(
        pool_size = (downscale, downscale)))
cnn.add(Conv2D(
        64, kernel, activation = 'relu'))
cnn.add(MaxPooling2D(
        pool_size = (downscale, downscale)))
cnn.add(Flatten())

cnn.add(Dense(
        units = 128, activation = 'relu'))
cnn.add(Dense(
        units = len(classes), activation = 'softmax'))
cnn.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

cnn.fit_generator(
        train, steps_per_epoch = 2048, epochs = 4, 
        validation_data = test, validation_steps = 512)
cnn.save('cnn-fruit.h5')

cnn = load_model('cnn-fruit.h5')

for i in range(0, 12):
    file = path + '/unknown-' + str(i) + '.jpg'
    img = image.load_img(
            file, target_size = (imgW, imgH))
    y = cnn.predict(expand_dims(
            image.img_to_array(img), axis = 0))
    print('Fruit in:', file, ':', classes[argmax(y)])
