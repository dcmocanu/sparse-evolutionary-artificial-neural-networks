# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation of Sparse Evolutionary Training (SET) of Multi Layer Perceptron (MLP) on lung dataset using Python, SciPy sparse data structures, and (optionally) Cython.
# This implementation can be used to create SET-MLP with hundred of thousands of neurons.
# If you would like to try out SET-MLP with various activation functions, optimization methods and so on (in the detriment of scalability) please use the Keras implementation from the folder "SET-MLP-Keras-Weights-Mask".

# This is a pre-alpha free software and was tested in Ubuntu 16.04 with Python 3.5.2, Numpy 1.14, SciPy 0.19.1, and (optionally) Cython 0.27.3;
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

# We thank to:
# Thomas Hagebols: for performing a thorough analyze on the performance of SciPy sparse matrix operations
# Ritchie Vink (https://www.ritchievink.com): for making available on Github a nice Python implementation of fully connected MLPs. This SET-MLP implementation was built on top of his MLP code:
#                                             https://github.com/ritchie46/vanilla-machine-learning/blob/master/vanilla_mlp.py

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
import scipy.io as sio
import sparseoperations
import datetime

def backpropagation_updates_Numpy(a, delta, rows, cols, out):
    for i in range (out.shape[0]):
        s=0
        for j in range(a.shape[0]):
            s+=a[j,rows[i]]*delta[j, cols[i]]
        out[i]=s/a.shape[0]

def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx

def createSparseWeights(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/10)
    print ("Create sparse matrix with ",weights.getnnz()," connections and ",(weights.getnnz()/(noRows * noCols))*100,"% sparsity level")
    weights=weights.tocsr()
    return weights



class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z

class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class MSE:
    def __init__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class NoActivation:
    """
    This is a plugin function for no activation.

    f(x) = x * 1
    """
    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(z):
        """
        The prime of z * 1 = 1
        :param z: (array)
        :return: z': (array)
        """
        return np.ones_like(z)


class SET_MLP:
    def __init__(self, dimensions, activations,epsilon=20):
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
        self.loss = None
        self.learning_rate = None
        self.momentum=None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.dimensions=dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw={}
        self.pdd={}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i], dimensions[i + 1])#create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

    def _feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = a[i]@self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

    def _back_prop(self, z, a, y_true):
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

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw=coo_matrix(self.w[self.n_layers-1])

        # compute backpropagation updates
        sparseoperations.backpropagation_updates_Cython(a[self.n_layers - 1],delta,dw.row,dw.col,dw.data)
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        #backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), delta)
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = (delta@self.w[i].transpose()) * self.activations[i].prime(z[i])
            dw = coo_matrix(self.w[i - 1])

            # compute backpropagation updates
            sparseoperations.backpropagation_updates_Cython(a[i - 1], delta, dw.row, dw.col, dw.data)
            # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            #backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(), delta)
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        #perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index]=-self.learning_rate * dw
            self.pdd[index] =  - self.learning_rate * np.mean(delta, 0)
        else:
            self.pdw[index]= self.momentum*self.pdw[index]-self.learning_rate * dw
            self.pdd[index] =  self.momentum * self.pdd[index] - self.learning_rate * np.mean(delta, 0)

        self.w[index] += self.pdw[index]-self.weight_decay*self.w[index]
        self.b[index] += self.pdd[index]-self.weight_decay*self.b[index]


    def fit(self, x, y_true, x_test,y_test,loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9, weight_decay=0.0002, zeta=0.3, testing=True, save_filename=""):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        :param momentum: (flt)
        :param weight_decay: (flt)
        :param zeta: (flt) #control the fraction of weights removed
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta

        maximum_accuracy=0

        metrics=np.zeros((epochs,3))

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_=x[seed]
            y_=y_true[seed]

            #training
            t1 = datetime.datetime.now()
            losstrain=0
            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l])
                losstrain+=self.loss.loss(y_[k:l], a[self.n_layers])
                self._back_prop(z, a, y_[k:l])
                #ToDo: adding dropout would improve the performance and decrease overfitting
            t2 = datetime.datetime.now()
            metrics[i, 0]=losstrain / (x.shape[0] // batch_size)
            print ("\nSET-MLP Epoch ",i)
            print ("Training time: ",t2-t1,"; Loss train: ",losstrain / (x.shape[0] // batch_size))

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy,activations=self.predict(x_test,y_test,batch_size)
                t4 = datetime.datetime.now()
                maximum_accuracy=max(maximum_accuracy,accuracy)
                losstest=self.loss.loss(y_test, activations)
                metrics[i, 1] = accuracy
                metrics[i, 2] = losstest
                print("Testing time: ", t4 - t3, "; Loss test: ", losstest,"; Accuracy: ", accuracy,"; Maximum accuracy: ", maximum_accuracy)

            t5 = datetime.datetime.now()
            if (i<epochs-1):# do not change connectivity pattern after the last epoch
                self.weightsEvolution()
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

            #save performance metrics values in a file
            if (save_filename!=""):
                np.savetxt(save_filename,metrics)

        return metrics



    def weightsEvolution(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        # TODO: this method could be seriously improved in terms of running time using Cython
        for i in range(1,self.n_layers):

            values=np.sort(self.w[i].data)
            firstZeroPos = find_first_pos(values, 0)
            lastZeroPos = find_last_pos(values, 0)

            largestNegative = values[int((1-self.zeta) * firstZeroPos)]
            smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

            wlil = self.w[i].tolil()
            pdwlil=self.pdw[i].tolil()
            wdok = dok_matrix((self.dimensions[i-1],self.dimensions[i]),dtype="float64")
            pdwdok= dok_matrix((self.dimensions[i-1],self.dimensions[i]),dtype="float64")

            #remove the weights closest to zero
            keepConnections=0
            for ik, (row, data) in enumerate(zip(wlil.rows, wlil.data)):
                for jk, val in zip(row, data):
                    if ((val < largestNegative) or (val > smallestPositive)):
                        wdok[ik,jk]=val
                        pdwdok[ik,jk]=pdwlil[ik,jk]
                        keepConnections+=1

            # add new random connections
            for kk in range(self.w[i].data.shape[0]-keepConnections):
                ik = np.random.randint(0, self.dimensions[i - 1])
                jk = np.random.randint(0, self.dimensions[i])
                while (wdok[ik,jk]!=0):
                    ik = np.random.randint(0, self.dimensions[i - 1])
                    jk = np.random.randint(0, self.dimensions[i])
                wdok[ik, jk]=np.random.randn() / 10
                pdwdok[ik, jk] = 0

            self.pdw[i]=pdwdok.tocsr()
            self.w[i]=wdok.tocsr()



    def predict(self, x_test,y_test,batch_size=1):
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
            _, a_test = self._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]
        correctClassification = 0
        for j in range(y_test.shape[0]):
            if (np.argmax(activations[j]) == np.argmax(y_test[j])):
                correctClassification += 1
        accuracy= correctClassification/y_test.shape[0]
        return accuracy, activations

if __name__ == "__main__":
    # Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
    np.random.seed(0)

    # load data
    mat = sio.loadmat('data/lung.mat')
    X = mat['X']
    # one hot encoding
    noClasses = np.max(mat['Y'])
    Y=np.zeros((mat['Y'].shape[0],noClasses))
    for i in range(Y.shape[0]):
        Y[i,mat['Y'][i]-1]=1

    #split data in training and testing
    indices=np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_train=X[indices[0:int(X.shape[0]*2/3)]]
    Y_train=Y[indices[0:int(X.shape[0]*2/3)]]
    X_test=X[indices[int(X.shape[0]*2/3):]]
    Y_test=Y[indices[int(X.shape[0]*2/3):]]

    #normalize data to have 0 mean and unit variance
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    xTrainMean = np.mean(X_train, axis=0)
    xTtrainStd = np.std(X_train, axis=0)
    X_train = (X_train - xTrainMean) / (xTtrainStd+0.0001)
    X_test = (X_test - xTrainMean) / (xTtrainStd+0.0001)

    # create SET-MLP
    set_mlp = SET_MLP((X_train.shape[1], 3000, 3000, 3000, Y_train.shape[1]), (Relu, Relu, Relu, Sigmoid), epsilon=20)

    # train SET-MLP
    set_mlp.fit(X_train, Y_train, X_test, Y_test, loss=MSE, epochs=100, batch_size=2, learning_rate=0.01, momentum=0.9, weight_decay=0.0002, zeta=0.3, testing=True, save_filename="Results/set_mlp.txt")

    # test SET-MLP
    accuracy,_=set_mlp.predict(X_test,Y_test,batch_size=1)

    print ("\nAccuracy of the last epoch on the testing data: ",accuracy)
