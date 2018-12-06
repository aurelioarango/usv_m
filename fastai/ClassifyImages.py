import csv
from PIL import Image
import os
import pathlib
import platform
from os import walk


def parse_file(filename):
    """Read and Extract images from file"""
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
            # print(usv_name)
            # print(file_prediction)
        line_count += 1
    return file_names, file_predictions

def save_images(path,usv_image_source_dir, prediction_folder):
    """ Look at the path provided and create a new directory to save the images"""
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)

    last_position =[]
    image_last_position=[]
    # Get dir name and image name
    if platform.system() == "Windows":
        last_position = path.rfind("\\")
    else:
        last_position = path.rfind("/")

    dir_name = path[last_position:]
    local_path = path[:last_position]

    fm_path =[]
    ff_path =[]
    trills_path = []
    noise_path = []
    
    # Make path dir according to windows or linux	
    if platform.system() == "Windows":
        # Create Paths for directories
        fm_path = path+'classified\\FM\\'
        #print(fm_path)
        ff_path = path+'classified\\FF\\'
        trills_path = path+'classified\\Trills\\'
        noise_path = path+'classified\\Noise\\'
    else:
        # Make path for directories
        fm_path = path+'classified/FM/'
        #print(fm_path)
        ff_path = path+'classified/FF/'
        trills_path = path+'classified/Trills/'
        noise_path = path+'classified/Noise/'
    # Create directories for image classes
    pathlib.Path(fm_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ff_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(trills_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(noise_path).mkdir(parents=True, exist_ok=True)
    
    for index in range (len(usv_image_source_dir)):
        # Grab path to image
        dir_data = usv_image_source_dir[index]
        if platform.system() == "Windows":
            image_last_position = dir_data.rfind('\\')
            
        else:
            image_last_position = dir_data.rfind('/')
        #print('dir data:, ',dir_data)
        # grab path to directory containing image
        path_usv = dir_data[:image_last_position+1]
        # grab file image
        image_filename = dir_data[image_last_position+1:]
        # change to directory containing image
        os.chdir(path_usv)
        try:
            # load image
            img = Image.open(image_filename, 'r')
            # Delete image file
            #os.remove(image_filename)

            # get predition status of the image
            pred = prediction_folder[index].strip().lower()
            # change to correct directory and save image
            if pred == "FM".lower():
                # print(fm_path)
                # print(os.getcwd())
                os.chdir(fm_path)
                img.save(image_filename)
            elif pred == "FF".lower():
                #print(ff_path)
                os.chdir(ff_path)
                #print(os.getcwd())
                img.save(image_filename)
            elif pred == "Trills".lower():
                # print(trills_path)
                # print(os.getcwd())
                os.chdir(trills_path)
                img.save(image_filename)
            elif pred == "Noise".lower():
                # print(noise_path)
                # print(os.getcwd())
                os.chdir(noise_path)
                img.save(image_filename)
            else:
                print("not found")
        except IOError:
            print("Image not found")

"""-------------MAIN FUNCTION--------------"""
if __name__ == "__main__":
    usv_images, prediction = parse_file("predictions.txt")
    path_to_dir = '/data/arango/thesis_workspace/data/'
    save_images(path_to_dir, usv_images, prediction)



