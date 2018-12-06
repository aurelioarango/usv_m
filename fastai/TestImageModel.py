
from torchvision import datasets
from fastai.vision import *
from fastai import *
from fastai.docs import *

import torchvision
from torchvision import transforms

import csv
import numpy

class ImagePathFolder(datasets.ImageFolder):
#    def __init__(self):
#        """BLANK"""
    def __getitem__(self,index):
        # Original ImageFolder  
        original_tuple = super(ImagePathFolder, self).__getitem__(index)
        # Image Path
        path = self.imgs[index][0]
        # New tuple that includes path
        path_tuple = (original_tuple + (path,))
        return path_tuple

def load_model(data, PATH):
    learner = ConvLearner(data, tvm.resnet34, metrics=accuracy)
    learner.load(PATH)
        #resnet32.load_state(state)
    model = learner.model.to("cpu")
    model.eval()
    return model

def load_test(path):
    test_data = ImagePathFolder(path, transform = transforms.Compose([transforms.ToTensor()]) )
    return test_data

def evaluate(test_data, model):
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    classes  = ('FF', 'FM', 'Noise', 'Trills')
    # to save to file
    to_file = []
    to_file.append('Image Path, GroundTruth, Predicted')
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
            out = path[0]+', ' + classes[labels[0]] +',' + classes[predicted[0]]
            to_file.append(out)
            print(out)
           
        #else:
           # print(path)
            #print(classes[labels[0]])
    # print(to_file[0])
    numpy.savetxt('predictions.txt',to_file,fmt='%6s', delimiter=',') 


