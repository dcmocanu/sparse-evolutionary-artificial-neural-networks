# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of a standard dense Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.
# This implementation serves just as a comparison for the SET-MLP and MLP-FixProb models

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
#Please note that in newer versions of keras_contrib you may encounter some import errors. You can find a fix for it on the Internet, or as an alternative you can try other activations functions.
from keras_contrib.layers.advanced_activations.srelu import SReLU
from keras.datasets import cifar10
from keras.utils import np_utils


class MLP_CIFAR10:
    def __init__(self):
        # set model parameters
        self.epsilon = 20 # control the sparsity level as discussed in the paper
        self.batch_size = 100 # batch size
        self.maxepoches = 1000 # number of epochs
        self.learning_rate = 0.01 # SGD learning rate
        self.num_classes = 10 # number of classes
        self.momentum=0.9 # SGD momentum

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        # initialize weights for SReLu activation function
        self.wSRelu1 = None
        self.wSRelu2 = None
        self.wSRelu3 = None

        # create a MLP-FixProb model
        self.create_model()

        # train the MLP-FixProb model
        self.train()


    def create_model(self):

        # create a dense MLP model for CIFAR10 with 3 hidden layers
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(32, 32, 3)))
        self.model.add(Dense(4000, name="dense_1", weights=self.w1))
        self.model.add(SReLU(name="srelu1", weights=self.wSRelu1))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1000, name="dense_2", weights=self.w2))
        self.model.add(SReLU(name="srelu2", weights=self.wSRelu2))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4000, name="dense_3", weights=self.w3))
        self.model.add(SReLU(name="srelu3", weights=self.wSRelu3))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, name="dense_4", weights=self.w4))
        self.model.add(Activation('softmax'))

    def train(self):

        # read CIFAR10 data
        [x_train,x_test,y_train,y_test]=self.read_data()

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        self.model.summary()

        sgd = optimizers.SGD(lr=self.learning_rate, momentum=self.momentum)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        historytemp = self.model.fit_generator(datagen.flow(x_train, y_train,
                                                            batch_size=self.batch_size),
                                               steps_per_epoch=x_train.shape[0] // self.batch_size,
                                               epochs=self.maxepoches,
                                               validation_data=(x_test, y_test),
                                               )

        self.accuracies_per_epoch = historytemp.history['val_acc']


    def read_data(self):

        #read CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #normalize data
        xTrainMean = np.mean(x_train, axis=0)
        xTtrainStd = np.std(x_train, axis=0)
        x_train = (x_train - xTrainMean) / xTtrainStd
        x_test = (x_test - xTrainMean) / xTtrainStd

        return [x_train, x_test, y_train, y_test]

if __name__ == '__main__':

    # create and run a dense MLP model on CIFAR10
    model=MLP_CIFAR10()

    # save accuracies over for all training epochs
    # in "results" folder you can find the output of running this file
    np.savetxt("results/dense_mlp_srelu_sgd_cifar10_acc.txt", np.asarray(model.accuracies_per_epoch))




