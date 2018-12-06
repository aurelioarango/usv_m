#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastai.vision import *

#import TrainImageModel
from TestImageModel import *
from TrainImageModel import *

if __name__ == '__main__':

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
   
    tpath = "/data/arango/thesis_workspace/data"
    test_path = '/data/arango/thesis_workspace/data/test'

    data = load_training(tpath)
    #print(data)
    #learn = training(data)
    
    test_data = load_test(test_path)
    model = load_model(data, '/data/arango/thesis_workspace/data/models/resnet34usv.h5')
    evaluate(test_data, model)

