#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Aurelio Arango
USV Training
"""
from fastai.vision import *

import sys

from TestImageModel import *
from TrainImageModel import *
from ReadParams import *

if __name__ == '__main__':

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    model, load_training_data, save_model_path, epochs = train_read_params(sys.argv[1:])  
    #tpath = "/data/arango/thesis_workspace/data"
    
    #test_path = '/data/arango/thesis_workspace/data/test'

    #data = load_training(tpath)
    data = load_training(load_training_data)
    #print(data)
    #learn = training(data,'/arango/thesis_workspace/data/models/', "resnet152")
    print(model)
    learn = training(data, save_model_path, model, epochs)
    

