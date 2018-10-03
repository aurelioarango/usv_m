#!/usr/bin/python
import csv
from PIL import Image
import os
import pathlib

def parse_csv_file(filename):
    """ Read and extract images file """
    """print(filename)
    with open(filename) as usv_prediction_file:
        file_reader = csv.reader(usv_prediction_file, delimiter=',')
        line_count = 0
        print("before reading rows")
        for row in file_reader:
            if line_count > 0:
                usv_name = {row[0]}
                file_prediction = {row[1]}
                print(usv_name)
                print(file_prediction)
            line_count += 1"""
    line_count =0
    usv_prediction_file = open(filename, 'r')
    file_names = []
    file_predictions = []
    for line in usv_prediction_file:
        if line_count > 0:
            #print(line)
            line = line.rstrip()
            data = line.split(",")
            usv_name = data[0]
            file_names.append(usv_name)
            file_prediction = data[1]
            file_predictions.append(file_prediction)
            #print(usv_name)
            #print(file_prediction)
        line_count += 1
    return file_names, file_predictions
def save_images(path,usv_image_source_dir, prediction_folder):

    #print(dir_data)

    # Make directories
    fm_path = path+'classified\\FM\\'
    # print(fm_path)
    ff_path = path+'classified\\FF\\'
    trills_path = path+'classified\\Trills\\'
    noise_path = path+'classified\\Noise\\'


    pathlib.Path(fm_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ff_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trills_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(noise_path).mkdir(parents=True, exist_ok=True)

    for index in range (len(usv_image_source_dir)):
        dir_data = usv_image_source_dir[index].split("\\")
        #print(dir_data)
        path_usv = dir_data[0]
        image_filename = dir_data[1]
        fullpath = path + path_usv
        # print("Current Dir: ",os.getcwd())
        os.chdir(fullpath)
        # print(fullpath)
        img = Image.open(image_filename, 'r')
        # Delete File
        #os.remove(image_filename)

        # print("Image file name: ", image_filename)
        # print("Image prediction: ", prediction_folder[index])

        if prediction_folder[index].lower() == "FM".lower():
            # print(fm_path)
            # print(os.getcwd())
            os.chdir(fm_path)
            img.save(image_filename)
        elif prediction_folder[index].lower() == "FF".lower():
            # print(fm_path)
            os.chdir(ff_path)
            # print(os.getcwd())
            img.save(image_filename)
        elif prediction_folder[index].lower() == "Trills".lower():
            # print(trills_path)
            # print(os.getcwd())
            os.chdir(trills_path)
            img.save(image_filename)
        elif prediction_folder[index].lower() == "Noise".lower():
            # print(noise_path)
            # print(os.getcwd())
            os.chdir(noise_path)
            img.save(image_filename)
        else:
            print("not found")
    # test image
    # img.show()
"""-------------MAIN FUNCTION--------------"""
if __name__ == "__main__":
    usv_images, prediction = parse_csv_file("cnn_model_results.csv")
    path_to_dir = 'C:\\Users\\keith-la\\USV_Project\\matlab\\sorted_\\archive\\old_data_93_94\\Test\\'

    save_images(path_to_dir, usv_images, prediction)


