from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space
import h5py
import csv
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn import metrics


from platform import python_version_tuple

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

import numpy as np

def data():
    """ """
    # dimensions of our images.
    img_width, img_height = 128, 128

    batch_size = 5
    train_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\Training'
    validation_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\Validation'
    test_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\Test'


    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=True,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills', 'Noise'])

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        shuffle=True,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills', 'Noise'])

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=True,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills', 'Noise'])



    return train_generator, validation_generator, test_generator

def create_model(train_generator, validation_generator):
    """ """
    img_width, img_height = 128, 128
    epochs = {{choice(range(75,150))}}
    nb_train_samples = 600
    nb_validation_samples = 140
    batch_size = 5


    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size={{choice(range(3, 8))}},
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size={{choice([1, 2, 3, 4])}}))

    model.add(Conv2D(32, kernel_size={{choice(range(3,8))}}))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size={{choice([1, 2])}}))

    model.add(Conv2D(64, kernel_size={{choice(range(3, 8))}}))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size={{choice([1, 2])}}))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss= {{choice(['categorical_crossentropy','mean_squared_error'])}}, 
                  optimizer={{choice(['adam','sgd', 'rmsprop'])}},
                  metrics=['accuracy'])

    """Implementing Early Stopping for better results"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, baseline=94.5)
    """
     validation_data: The model will not be trained on this data
    """
    model.fit_generator(
        train_generator,
        callbacks=[early_stopping],
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    score, acc = model.evaluate_generator(generator=validation_generator,

                                          steps=nb_validation_samples // batch_size)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    """ """
    trials = Trials()
    train_generator, validation_generator, test_generator = data()

    best_run, best_model, space = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=trials,
                                          eval_space=True,
                                          return_space=True)

    print('Saving Best Model')
    best_model.save('best_run_four_class_es_trial.h5')
    print("Best performing model:")
    print(best_run)
    print("Validation %", best_model.evaluate_generator(validation_generator))
    print("Test ", best_model.evaluate_generator(test_generator))

    x, y = izip(*(test_generator[i] for i in xrange(len(test_generator))))
    x_test, y_test = np.vstack(x), np.vstack(y)
    """
    test_predictions = best_model.predict(x_test)
    confusion = metrics.confusion_matrix(y_test, test_predictions.round())"""

    predictions_test_generator = best_model.predict_generator(test_generator)
    """
    print("Testing accuracy = ", metrics.accuracy_score(y_test, predictions_test_generator.round()))
    print("Recall = ", metrics.recall_score(y_test, predictions_test_generator.round(), average='samples' )) 
    """
    hyperparams = []
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        vals = eval_hyperopt_space(space, vals)
        vals['accuray'] = -trial.get('result')['loss']
        hyperparams.append(vals)
    
    # write hyperparameters to a csv file
    keys = hyperparams[0].keys()
    with open("hyperparameters_all_four_class_es_trial.txt", 'w') as output:
        dic_writer = csv.DictWriter(output, keys)
        dic_writer.writeheader()
        dic_writer.writerows(hyperparams)
