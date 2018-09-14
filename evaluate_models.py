# Evaluate Models using ensemble model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

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


if __name__ == '__main__':

    test_generator = data()
    server_model = load_model('best_run_four_class.h5')
    cnn_model = load_model('best_run_four_class_es_trial_95_14.h5' )
    print("Server Test ", server_model.evaluate_generator(test_generator))
    print("cnn Test ",cnn_model.evaluate_generator(test_generator))