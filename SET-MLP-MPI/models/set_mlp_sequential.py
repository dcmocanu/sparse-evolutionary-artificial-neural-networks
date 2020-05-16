# Authors: Decebal Constantin Mocanu et al.;
# Code associated with ECMLPKDD 2019 tutorial "Scalable Deep Learning: from theory to practice"; https://sites.google.com/view/sdl-ecmlpkdd-2019-tutorial
# This is a pre-alpha free software and was tested in Ubuntu 16.04 with Python 3.5.2, Numpy 1.14, SciPy 0.19.1, and (optionally) Cython 0.27.3;

# If you use parts of this code please cite the following article:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#If you have space please consider citing also these articles

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

#@article{Liu2019onemillion,
#  author =        {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
#  journal =       {arXiv:1901.09181},
#  title =         {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
#  year =          {2019},
#}

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from models.nn_functions import *
# the "sparseoperations" Cython library was tested in Ubuntu 16.04. Please note that you may encounter some "solvable" issues if you compile it in Windows.
import sparsebackpropagation
import datetime
import os
import sys
import concurrent.futures
import numpy as np
import threading
from numba import njit, jit, prange, jitclass
import functools
import itertools
import numba

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_Numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    keep_prob = 1 - rate
    noise = np.random.binomial(1, keep_prob, noise_shape) / np.float32(keep_prob)
    return x * noise, noise


def createSparseWeights(epsilon, noRows, noCols):
    limit = np.sqrt(6. / float(noRows + noCols))

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols), dtype='float32')
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float32(np.random.uniform(-limit, limit))
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype))  # boolean return


class SET_MLP:
    def __init__(self, dimensions, activations, **config):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes


        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------

        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        """
        self.n_layers = len(dimensions)
        self.learning_rate = config['lr']
        self.momentum = config['momentum']
        self.epochs = config['n_epochs']
        self.weight_decay = config['weight_decay']
        self.epsilon = config['epsilon']  # control the sparsity level as discussed in the paper
        self.zeta = config['zeta']  # the fraction of the weights removed
        self.dropout_rate = config['dropout_rate']  # dropout rate
        self.dimensions = dimensions
        self.batch_size = config['batch_size']

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}
        self.activations = {}

        for i in range(len(dimensions) - 2):
            self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i], dimensions[i + 1])  #create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
            self.activations[i + 2] = activations[i]

        limit = np.sqrt(6. / float(dimensions[-2] + dimensions[-1]))
        self.w[len(dimensions) - 1] = csr_matrix(np.random.uniform(-limit, limit,
                                                                   (dimensions[-2], dimensions[-1])), dtype='float32')
        self.b[len(dimensions) - 1] = np.zeros(dimensions[-1], dtype='float32')
        self.activations[len(dimensions)] = activations[-1]

        if config['loss'] == 'mse':
            self.loss = MSE(self.activations[self.n_layers])
        elif config['loss'] == 'cross_entropy':
            self.loss = CrossEntropy()
        else:
            raise NotImplementedError("The given loss function is  ot implemented")

    def parameters(self):
        """
                Retrieve the network parameters.
                :return: model parameters.
        """

        params = {
            'w': self.w,
            'b': self.b,
            'pdw': self.pdw,
            'pdd': self.pdd,
        }

        return params

    def set_parameters(self, params):
        self.w = params['w']
        self.b = params['b']
        self.pdw = params['pdw']
        self.pdd = params['pdd']

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = 1. - self.dropout_rate

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')
        # compute backpropagation updates
        #sparsebackpropagation.backpropagation_updates(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(),  np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):

            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            #sparsebackpropagation.backpropagation_updates(a[i - 1], delta, dw.row, dw.col, dw.data)
            # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(),  np.mean(delta, axis=0))
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

        # with ThreadPoolExecutor() as executor:
        #     k, v = update_params.items()
        #     results = executor.map(self._update_w_b, k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index]  # - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index]  # - self.weight_decay * self.b[index]

    def train_on_batch(self, x, y):
        z, a, masks = self._feed_forward(x, True)
        self._back_prop(z, a, masks, y)
        accuracy, activations = self.predict(x, y)
        return self.loss.loss(y, activations), accuracy

    def test_on_batch(self, x, y):
        accuracy, activations = self.predict(x, y)
        return self.loss.loss(y, activations), accuracy

    def fit(self, x, y_true, x_test, y_test, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # Initiate the loss object with the final activation function
        self.save_filename = save_filename
        # self.inputLayerConnections = []
        # self.inputLayerConnections.append(self.getCoreInputConnections())
        # np.savez_compressed(self.save_filename + "_input_connections.npz",
        #                     inputLayerConnections=self.inputLayerConnections)

        maximum_accuracy = 0
        metrics = np.zeros((self.epochs, 4))

        weights = []
        biases = []

        for i in range(self.epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            # training
            t1 = datetime.datetime.now()

            for j in range(x.shape[0] // self.batch_size):
                k = j * self.batch_size
                l = (j + 1) * self.batch_size
                z, a, masks = self._feed_forward(x_[k:l], True)

                self._back_prop(z, a, masks,  y_[k:l])

            t2 = datetime.datetime.now()

            print("\nSET-MLP Epoch ", i)
            print("Training time: ", t2 - t1)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test)
                accuracy_train, activations_train = self.predict(x, y_true)

                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test

                print(f"Testing time: {t4 - t3}\n; Loss test: {loss_test}; \n"
                                 f"Accuracy test: {accuracy_test}; \n"
                                 f"Maximum accuracy val: {maximum_accuracy}")

            weights.append(self.w)
            biases.append(self.b)

            t5 = datetime.datetime.now()
            if (i < self.epochs - 1):# do not change connectivity pattern after the last epoch

                # self.weightsEvolution_I() # this implementation is more didactic, but slow.
                self.weightsEvolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

            print(self.w[1].count_nonzero())
            print(self.w[2].count_nonzero())
            print(self.w[3].count_nonzero())
            print(self.w[4].count_nonzero())

            # save performance metrics values in a file
            if (self.save_filename != ""):
                np.savetxt(self.save_filename+".txt", metrics)

        np.savez_compressed(self.save_filename + "_weights.npz", *weights)
        np.savez_compressed(self.save_filename + "_biases.npz", *biases)

        return metrics

    def fit_generator(self, x, y_true, x_test, y_test, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        x = x.reshape(-1, 32, 32, 3)
        # data augmentation
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
        datagen.fit(x)

        # Initiate the loss object with the final activation function
        self.save_filename = save_filename
        self.inputLayerConnections = []
        self.inputLayerConnections.append(self.getCoreInputConnections())
        np.savez_compressed(self.save_filename + "_input_connections.npz",
                            inputLayerConnections=self.inputLayerConnections)
        maximum_accuracy = 0
        metrics = np.zeros((self.epochs, 4))
        weights = []
        biases = []
        for i in range(self.epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]
            # training
            t1 = datetime.datetime.now()
            output_generator = datagen.flow(x_, y_, batch_size=self.batch_size)

            for j in range(x.shape[0] // self.batch_size):
                x_b, y_b = next(output_generator)
                x_b = x_b.reshape(-1, 32 * 32 * 3)
                z, a, masks = self._feed_forward(x_b, True)
                self._back_prop(z, a, masks, y_b)
            t2 = datetime.datetime.now()
            print("\nSET-MLP Epoch ", i)
            print("Training time: ", t2 - t1)
            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test)
                accuracy_train, activations_train = self.predict(x.reshape(-1, 32 * 32 * 3), y_true)
                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test
                print(f"Testing time: {t4 - t3}\n; Loss test: {loss_test}; \n"
                      f"Accuracy test: {accuracy_test}; \n"
                      f"Maximum accuracy val: {maximum_accuracy}")
            weights.append(self.w)
            biases.append(self.b)
            t5 = datetime.datetime.now()
            if (i < self.epochs - 1):  # do not change connectivity pattern after the last epoch
                # self.weightsEvolution_I() # this implementation is more didactic, but slow.
                self.weightsEvolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)
            K.clear_session()
            print(self.w[1].count_nonzero())
            print(self.w[2].count_nonzero())
            print(self.w[3].count_nonzero())
            print(self.w[4].count_nonzero())
            # save performance metrics values in a file
            if (self.save_filename != ""):
                np.savetxt(self.save_filename + ".txt", metrics)
        np.savez_compressed(self.save_filename + "_weights.npz", *weights)
        np.savez_compressed(self.save_filename + "_biases.npz", *biases)
        return metrics

    def getCoreInputConnections(self):
        values = np.sort(self.w[1].data)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        wlil = self.w[1].tolil()
        wdok = dok_matrix((self.dimensions[0], self.dimensions[1]), dtype="float32")

        # remove the weights closest to zero
        keepConnections = 0
        for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
            for jk, val in zip(row, data):
                if ((val < largestNegative) or (val > smallestPositive)):
                    wdok[ik, jk] = val
                    keepConnections += 1
        return wdok.tocsr().getnnz(axis=1)

    def weightsEvolution_I(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        for i in range(1, self.n_layers - 1):

            values = np.sort(self.w[i].data)
            firstZeroPos = find_first_pos(values, 0)
            lastZeroPos = find_last_pos(values, 0)

            largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
            smallestPositive = values[
                int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

            wlil = self.w[i].tolil()
            pdwlil = self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")
            pdwdok = dok_matrix((self.dimensions[i - 1], self.dimensions[i]), dtype="float32")

            # remove the weights closest to zero
            keepConnections = 0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if ((val < largestNegative) or (val > smallestPositive)):
                        wdok[ik, jk] = val
                        pdwdok[ik, jk] = pdwlil[ik, jk]
                        keepConnections += 1
            limit = np.sqrt(6. / float(self.dimensions[i] + self.dimensions[i + 1]))
            # add new random connections
            for kk in range(self.w[i].data.shape[0] - keepConnections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik, jk] != 0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk] = np.random.uniform(-limit, limit)
                pdwdok[ik, jk] = 0

            self.pdw[i] = pdwdok.tocsr()
            self.w[i] = wdok.tocsr()

    def weightsEvolution_II(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # improved running time using numpy routines - Amarsagar Reddy Ramapuram Matavalam (amar@iastate.edu)
        for i in range(1, self.n_layers-1):
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            #if(self.w[i].count_nonzero()/(self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8):
                t_ev_1 = datetime.datetime.now()
                # converting to COO form - Added by Amar
                wcoo=self.w[i].tocoo()
                valsW=wcoo.data
                rowsW=wcoo.row
                colsW=wcoo.col

                pdcoo=self.pdw[i].tocoo()
                valsPD=pdcoo.data
                rowsPD=pdcoo.row
                colsPD=pdcoo.col
                #print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
                values=np.sort(self.w[i].data)
                firstZeroPos = find_first_pos(values, 0)
                lastZeroPos = find_last_pos(values, 0)

                largestNegative = values[int((1-self.zeta) * firstZeroPos)]
                smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

                #remove the weights (W) closest to zero and modify PD as well
                valsWNew=valsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                rowsWNew=rowsW[(valsW > smallestPositive) | (valsW < largestNegative)]
                colsWNew=colsW[(valsW > smallestPositive) | (valsW < largestNegative)]

                newWRowColIndex=np.stack((rowsWNew,colsWNew) , axis=-1)
                oldPDRowColIndex=np.stack((rowsPD,colsPD) , axis=-1)


                newPDRowColIndexFlag=array_intersect(oldPDRowColIndex,newWRowColIndex) # careful about order

                valsPDNew=valsPD[newPDRowColIndexFlag]
                rowsPDNew=rowsPD[newPDRowColIndexFlag]
                colsPDNew=colsPD[newPDRowColIndexFlag]

                self.pdw[i] = coo_matrix((valsPDNew, (rowsPDNew, colsPDNew)),(self.dimensions[i - 1], self.dimensions[i])).tocsr()

                # add new random connections
                keepConnections=np.size(rowsWNew)
                lengthRandom=valsW.shape[0]-keepConnections
                limit = np.sqrt(6. / float(self.dimensions[i - 1] + self.dimensions[i]))
                randomVals = np.random.uniform(-limit, limit, lengthRandom)
                zeroVals=0*randomVals # explicit zeros

                # adding  (wdok[ik,jk]!=0): condition
                while (lengthRandom>0):
                    ik = np.random.randint(0, self.dimensions[i - 1],size=lengthRandom,dtype='int32')
                    jk = np.random.randint(0, self.dimensions[i],size=lengthRandom,dtype='int32')

                    randomWRowColIndex=np.stack((ik,jk) , axis=-1)
                    randomWRowColIndex=np.unique(randomWRowColIndex,axis=0) # removing duplicates in new rows&cols
                    oldWRowColIndex=np.stack((rowsWNew,colsWNew) , axis=-1)

                    uniqueFlag=~array_intersect(randomWRowColIndex,oldWRowColIndex) # careful about order & tilda


                    ikNew=randomWRowColIndex[uniqueFlag][:,0]
                    jkNew=randomWRowColIndex[uniqueFlag][:,1]
                    # be careful - row size and col size needs to be verified
                    rowsWNew=np.append(rowsWNew, ikNew)
                    colsWNew=np.append(colsWNew, jkNew)

                    lengthRandom=valsW.shape[0]-np.size(rowsWNew) # this will constantly reduce lengthRandom

                # adding all the values along with corresponding row and column indices - Added by Amar
                valsWNew=np.append(valsWNew, randomVals) # be careful - we can add to an existing link ?
                #valsPDNew=np.append(valsPDNew, zeroVals) # be careful - adding explicit zeros - any reason??
                if (valsWNew.shape[0] != rowsWNew.shape[0]):
                    print("not good")
                self.w[i]=coo_matrix((valsWNew , (rowsWNew , colsWNew)),(self.dimensions[i-1],self.dimensions[i])).tocsr()

                #print("Number of non zeros in W and PD matrix after evolution in layer",i,[(self.w[i].data.shape[0]), (self.pdw[i].data.shape[0])])

                t_ev_2 = datetime.datetime.now()
                #print("Weights evolution time for layer",i,"is", t_ev_2 - t_ev_1)

        # self.pdw = {}
        # self.pdd = {}

    def predict(self, x_test, y_test, batch_size=1):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations
