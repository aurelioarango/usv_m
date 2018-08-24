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

def data():
    """ """
    # dimensions of our images.
    img_width, img_height = 128, 128

    batch_size = 5
    train_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\Training'
    validation_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\Validation'



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
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills'])

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills'])



    return train_generator, validation_generator

def create_model(train_generator, validation_generator):
    """ """
    img_width, img_height = 128, 128
    epochs = {{choice(range(50,150))}}
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
    model.add(Dense(3, activation={{choice(['softmax','sigmoid'])}}))

    model.compile(loss= {{choice(['categorical_crossentropy','mean_squared_error'])}}, 
                  optimizer={{choice(['adam','sgd', 'rmsprop'])}},
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
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
    train_generator, validation_generator = data()

    best_run, best_model, space = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=trials,
					  eval_space=True,
                                          return_space=True)

    print("Best performing model:")
    print(best_run)
    print(best_model.evaluate_generator(validation_generator))

    hyperparams = []
    for t, trial in enumerate(trials):
        #print(trial)
        vals = trial.get('misc').get('vals')
        #print("Trial %s vals: %s" %(t, vals))
        vals = eval_hyperopt_space(space, vals)
        vals['accuray'] = -trial.get('result')['loss']
        hyperparams.append(vals)
        #print(vals)
        #print("Accuracy: ", vals['accuray'])
    
    # write hyperparameters to a csv file
    keys = hyperparams[0].keys()
    with open("hyperparameters_all_Three_classes.txt", 'w') as output:
        dic_writer = csv.DictWriter(output, keys)
        dic_writer.writeheader()
        dic_writer.writerows(hyperparams)
