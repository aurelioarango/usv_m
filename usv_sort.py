#!/usr/bin/python
"""Author: Aurelio Arango"""
#from os import listdir
#from os.path import isfile, join

from os import walk
import sys, getopt
import pathlib
import random
import matplotlib.image as mpimg
from PIL import Image
import os
import platform
import numpy as np

"""READ PATH
PARAM list of arguments from input
RETURN Path to Directory & Percent to sort in valiation"""
def read_path(argv):

    dir_path = ''
    percent = .2
    try:
        opts, args = getopt.getopt(argv, "hi:p:", ["ipath", "ppercent"])
    except getopt.GetoptError:
        print("usv_sort.py -i <usv_dir_path> - p <validation_percent>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usv_sort.py -i <usv_dir_path>')
        elif opt in ("-i", "--ipath"):
            dir_path = arg
        elif opt in ("-p","--ppercent" ):
            percent = int(arg)/100.0
        else:
            print("usv_sort.py -i <usv_dir_path> - p <validation_percent>")

    print('Dir Path is ', dir_path)
    print("Validation Percent ", percent)
    return dir_path, percent

def split_data(path, percent):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)

    # Get dir name
    last_position =path.rfind("\\")
    dir_name = path[last_position+1:]
    local_path = path[:last_position]

    val_path = ''
    test_path = ''
    if platform.system() == "Windows":
        # Create Paths
        val_path = local_path + "\\Validation\\" + dir_name
        test_path = local_path + "\\Training\\" + dir_name
    else:
        val_path = local_path + "/Validation/" + dir_name
        test_path = local_path + "/Training/" + dir_name
    # Create Dir
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)

    # Iterate through file list
    for index in range (len(filenames)):
        # make sure we are in the correct directory
        os.chdir(path)
        # read image
        img = Image.open(filenames[index])
        # Randomly decide where to store image
        if random.uniform(0, 1) < percent:
            os.chdir(val_path)
            img.save(filenames[index])
        else:
            os.chdir(test_path)
            img.save(filenames[index])


"""-------------MAIN FUNCTION--------------"""
if __name__ == "__main__":

    dir_path, percent = read_path(sys.argv[1:])
    split_data(dir_path, percent)
