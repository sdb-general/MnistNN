# MnistNN
some attempts at making learning models for the mnist dataset

improved_mnist.py contains an attempt to attach LIME to a pretrained NN. running this script should produce an image explaining
why the net thought it made its classification.

CNN attempt learns the MNIST dataset through convolutions - at the end of the script it has an attempt to generate a saliency
map. Need to update this to load the data through torchvision.

you can download the dataset it refers to here http://yann.lecun.com/exdb/mnist/
