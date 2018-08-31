# Construct models using keras library
import csv
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras import regularizers
from six.moves import cPickle as pickle
from seq_util import *
from sklearn import metrics
import time

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperas.utils import eval_hyperopt_space

from optparse import OptionParser

def data():
    pickle_file = 'promoters.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        x_train = save['train_dataset']
        y_train = save['train_labels']
        x_val = save['valid_dataset']
        y_val = save['valid_labels']
        x_test = save['test_dataset']
        y_test = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', x_train.shape, y_train.shape)
        print('Validation set', x_val.shape, y_val.shape)
        print('Test set', x_test.shape, y_test.shape)
        print(reverse_convert(x_train[-1]))
        print(reverse_convert(x_test[-1]))
        print(y_test[0:10])

        return x_train, y_train, x_val, y_val, x_test, y_test


def create_model(x_train, y_train, x_val, y_val, x_test, y_test):
    print('Building model...')
    model = Sequential()
    model.add(Conv1D(filters={{choice([10, 20, 40, 60, 80, 100])}},
                 kernel_size={{choice(range(6, 21))}}, #{{choice([7, 11, 15, 19, 21])}},
                 strides=1,
                 padding="valid",
                 use_bias=True,
                 kernel_regularizer=regularizers.l2({{choice([0.0001, 0.0005, 0.001, 0.005, 0.01])}}),
                 input_shape=(100, 4),
                 ))
    model.add(MaxPooling1D(pool_size={{choice([1, 2, 3, 4])}}))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    model.add(Conv1D(filters={{choice([5, 10, 15, 20, 25, 30])}},
                 kernel_size= {{choice(range(4,17))}}, #{{choice([4, 6, 9, 13, 17])}},
                 strides=1,
                 padding="valid",
                 use_bias=True,
                 kernel_regularizer=regularizers.l2({{choice([0.0001, 0.0005, 0.001, 0.005, 0.01])}})
                 ))
    #model.add(MaxPooling1D(pool_size={{choice([1, 2, 4])}}))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(Activation({{choice(['sigmoid', 'relu'])}}))
    #model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid',
                kernel_regularizer=regularizers.l2({{choice([0.0001, 0.0005, 0.001])}}),
                activity_regularizer=regularizers.l1({{choice([0.0001, 0.0005, 0.001])}})))

    print("Compiling model...")
    model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

    print("Training model...")
    model.fit(x_train, y_train, epochs={{choice(range(5, 15))}},
              batch_size={{choice([16, 24, 32])}},
              shuffle=True, validation_data=(x_val, y_val))

    score, acc = model.evaluate(x_val, y_val)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    parser = OptionParser(description='Select hyperparameters for promoter prediction')
    parser.add_option("-n", "--numevals", dest="nevals",
                      help="Number of hyperparameter evaluations",
                      type="int", default="5")
    parser.add_option("-m", "--savemodelfile", dest="modelfile",
                      help="File to save the best model",
                      metavar='FILE', default="promoters_best_model.hdf5")
    parser.add_option("-p", "--hyperparamfile", dest="hyperfile",
                      help="File to save the evaluated hyperparameters",
                      metavar='FILE', default="hyper_parameters.csv"
                      )
    (options, args) = parser.parse_args()

    from numpy.random import seed
    seed(1)

    start = time.time()
    trials = Trials()
    x_train, y_train, x_val, y_val, x_test, y_test = data()
    print("Select model hyperparameters...")

    best_run, best_model, space = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=options.nevals,
                                          trials=trials,
                                          eval_space=True,
                                          return_space=True
                                          )
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    # save the best model to a file
    best_model.save(options.modelfile)
    #model = load_model(options.modelfile)

    print("Best performing model chosen hyper-parameters:")
    #real_param_values = eval_hyperopt_space(space, best_run)
    print(best_run)
    test_predictions = best_model.predict(x_test)
    confusion = metrics.confusion_matrix(y_test, test_predictions.round())
    print(confusion)
    print("testing accuracy = ", metrics.accuracy_score(y_test, test_predictions.round()))
    print("recall = ", metrics.recall_score(y_test, test_predictions.round()))
    print(metrics.classification_report(y_test, test_predictions.round()))

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
    with open(options.hyperfile, 'w') as output:
        dic_writer = csv.DictWriter(output, keys)
        dic_writer.writeheader()
        dic_writer.writerows(hyperparams)

    end = time.time()
    print("Run time = ", end - start)

