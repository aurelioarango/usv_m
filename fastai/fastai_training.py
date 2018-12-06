#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets 

from fastai.vision import *
from fastai import *
from fastai.docs import *


import torchvision 
from torchvision import transforms

import csv
import numpy

class ImagePathFolder(datasets.ImageFolder):

    def __getitem__(self,index):
        # Original ImageFolder  
        original_tuple = super(ImagePathFolder, self).__getitem__(index)
        # Image Path
        path = self.imgs[index][0]
        # New tuple that includes path
        path_tuple = (original_tuple + (path,))
        return path_tuple


def load_training():
    data = image_data_from_folder("/data/arango/thesis_workspace/data")

    #learn = ConvLearner(data, tvm.resnet18, metrics=accuracy)
    return data

def load_test():
    path='/data/arango/thesis_workspace/data/test'
    # test_data = image_data_from_folder("/data/arango/thesis_workspace/test_data/test", test_name=‘test’)
    #test_data = torchvision.datasets.ImageFolder(path,
    #    transform = transforms.Compose([transforms.ToTensor()]) )
    test_data = ImagePathFolder(path, transform = transforms.Compose([transforms.ToTensor()]) )
    return test_data

def load_model(data, PATH):
    learner = ConvLearner(data, tvm.resnet34, metrics=accuracy)
    learner.load(PATH)
    #resnet32.load_state(state)
    model = learner.model.to("cpu")
    model.eval()
    return model

def evaluate(test_data, model):
    #model.eval(data)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    classes  = ('FF', 'FM', 'Noise', 'Trills')
    # learn = ConvLearner( data,model)
    # learn.predict(data, is_test=True) 

    # to save to file
    to_file = []
    for data in testloader:
        # print(len(data))
        images, labels, path = data
        
        if len(labels) >= 1 :
            #print('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
            # to_file.append('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
            outputs = model(images)
            # print('images ',images) # actual images
            _, predicted = torch.max(outputs, 1)
            # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))
            out = 'Predicted, '+path[0]+', ' + classes[labels[0]] +',' + classes[predicted[0]]
            #to_file.append(out)
            print(out)
           
        else:
            print(path)
            #print(classes[labels[0]])
    # print(to_file[0])
    numpy.savetxt('preditions.txt',to_file,fmt='%6s', delimiter=',') 

def training(data):
    learn = ConvLearner(data, tvm.resnet34, metrics = accuracy)

    learn.fit(100)
    learn.save('/data/arango/thesis_workspace/data/models/resnet34usv.h5')
    return learn


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data = load_training()
    #learn = training(data)
    test_data = load_test()
    model = load_model(data, '/data/arango/thesis_workspace/data/models/resnet34usv.h5')
    evaluate(test_data, model)


