from fastai.vision import *
from fastai import *
from fastai.docs import *


import torchvision
from torchvision import transforms
import torchvision.models as models
import sys


def load_training(path):
    data = image_data_from_folder(path)

    return data

def training(data, path, model,epochs):
    learn=[]
    arch =[]
    #print("Model: ", model)
    if model == "resnet34".lower():
        #arch = tvm.resnet34()
        print("Resnet34")
        learn = ConvLearner(data, models.resnet34, metrics = accuracy) 
    elif model == "resnet50".lower():
        #arch = tvm.inception_v3()
        print("Resnet50")
        learn = ConvLearner(data, models.resnet50, metrics = accuracy)
    elif model == "resnet152".lower():
        print("Resnet152")
        learn = ConvLearner(data, models.resnet152, metrics = accuracy)
    elif model == "resnet101".lower():
        #arch = tvm.resnet34()
        print("Resnet101")
        learn = ConvLearner(data, models.resnet101, metrics = accuracy)
    elif:
        print("Default: Resnet18")
        #arch = tvm.resnet18
        learn = ConvLearner(data, models.resnet18, metrics = accuracy)
    else:
        print("No Model")
        sys.exit(2)
 
    print("Creating learn from model")
    #learn = ConvLearner(data, tvm.vgg13, metrics = accuracy)
    #print("fit")
    learn.fit(epochs)
    #print(path)
    #print(model)

    path = path+model+".h5"
    print(path)
    learn.save(path)
    return learn


