#!/usr/bin/python
"""Author: Aurelio Arnago"""
#from os import listdir
#from os.path import isfile, join

from os import walk
import sys, getopt
import pathlib
import random
import matplotlib.image as mpimg
from PIL import Image
import os
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

    print('Dir Path is ', dir_path)
    print("Validation Percent ", percent)
    return dir_path, percent

def split_data(path, percent):
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    #break
    print(filenames)
    # Get dir name
    last_position =path.rfind("\\")
    dir_name = path[last_position+1:]
    local_path = path[:last_position]
    #print(dir_name)
    #print(local_path)

    # Create Paths
    val_path = local_path + "\\Validation\\" + dir_name
    test_path = local_path + "\\Training\\" +dir_name
    # Create Dir
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
    #make sure we are in the correct directory


    # Iterate through file list
    for index in range (len(filenames)):
        #print(filenames[index])
        # read image
        #img = mpimg.imread(filenames[index])
        os.chdir(path)
        img = Image.open(filenames[index])

        #im = Image.open(filenames[index])
        #im.show()
        if random.uniform(0, 1) < percent:
            print("less than")
            os.chdir(val_path)
            #mpimg.imsave(filenames[index])
            img.save(filenames[index])
        else:
            print("grater")
            os.chdir(test_path)
            #mpimg.imsave(filenames[index])
            #mpimg.
            img.save(filenames[index])


"""-------------MAIN FUNCTION--------------"""
if __name__ == "__main__":

    dir_path, percent = read_path(sys.argv[1:])
    split_data(dir_path, percent)
