from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional
import h5py


def data():
    """ """
    # dimensions of our images.
    img_width, img_height = 128, 128

    batch_size = 5
    train_data_dir = '/data/arango/thesis_workspace/data/Training'
    validation_data_dir = '/data/arango/thesis_workspace/data/Validation'



    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')



    return train_generator, validation_generator

def create_model(train_generator, validation_generator):
    """ """
    img_width, img_height = 128, 128
    epochs = 50
    nb_train_samples = 780
    nb_validation_samples = 200
    batch_size = 5


    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))

    model.compile(loss={{choice(['categorical_crossentropy', 'binary_crossentropy','mean_squared_error'])}},
                  optimizer={{choice(['adam', 'rsmprop', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    score, acc = model.evaluate_generator(generator=validation_generator,
                                          steps=nb_validation_samples // batch_size)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    """ """
    train_generator, validation_generator = data()

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Best performing model:")

    print(best_model.evaluate(validation_generator))

