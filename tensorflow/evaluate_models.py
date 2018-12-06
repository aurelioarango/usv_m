# Evaluate Models using ensemble model
"""
Use: Prediction
source: https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import pandas as pd

import h5py

def data():
    """ """
    # dimensions of our images.
    img_width, img_height = 128, 128

    batch_size = 5
    test_data_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\archive\\old_data_93_94\\Test'

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=False,
        color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['FF', 'FM', 'Trills', 'Noise'])

    return test_generator

def print_predictions(generator, model, filename):
    generator.reset()
    """Predicted class"""
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    """labels"""
    labels = (generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions})
    results.to_csv(filename+".csv", index=False)


if __name__ == '__main__':

    test_generator = data()
    server_model = load_model('best_run_four_class.h5')
    server_model_2 = load_model('best_run_four_class_95_1.h5')
    cnn_model = load_model('best_run_four_class_es_trial_95_14.h5')

    print("Server Test ", server_model.evaluate_generator(test_generator))
    print("Server Test ", server_model_2.evaluate_generator(test_generator))
    print("cnn Test ", cnn_model.evaluate_generator(test_generator))

    print_predictions(test_generator, cnn_model, 'cnn_model_results')
    print_predictions(test_generator, server_model, 'server_model_results')
    print_predictions(test_generator, server_model_2, 'server2_model_results')








