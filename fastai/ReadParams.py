#Aurelio Arango
#Date 11/28/2018


import sys, getopt


def train_read_params(argv):

    load_path='/data/arango/thesis_workspace/data'
    save_path='/data/arango/thesis_workspace/model/'
    model='resnet18'
    epochs = 10
    print (argv)

    try:
        opts, args = getopt.getopt(argv, "hH:l:s:m:e:",["help","loadpath","savepath", "model", "epochs"])
    except getopt.GetoptError:
        print("main_train.py -m <model>")
        sys.exit(2)
    print("opts: ",opts)

    #if len(opts) == 0:
    #    sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h","-H", "help"):
            print("-m model to use <resnet18, resnet34, resnet50, resnet101, resnet152>")
            print("-l <load path>")
            print("-s <save path>")
            print("-e <number epochs>")
            sys.exit(2)
        elif opt in ("-l","--loadpath"):
            load_path = arg
        elif opt in ("-s","--savepath"):
            save_path = arg
        elif opt in ("-m","--model"):
            model = arg
        elif opt in ("-e","--epochs"):
            print("arg", arg)
            epochs = int(arg)
        else:
            print("-m model to use <resnet18, resnet34, resnet50, resnet101, resnet152>")
            print("-l <load path>")
            print("-s <save path>")
            print("-e <number epochs>") 
            sys.exit(2)

    print("Load Path: ", load_path)
    print("Save Path: ", save_path)
    print("Model: ", model)
    print("Epochs: ", epochs)

    return model, load_path, save_path, epochs


#def read_test_params(argv):
"""Function to read parameters for testing"""


