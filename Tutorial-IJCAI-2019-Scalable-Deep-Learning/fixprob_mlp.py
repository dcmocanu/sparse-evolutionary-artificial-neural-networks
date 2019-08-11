# Authors: Decebal Constantin Mocanu et al.;
# Code associated with IJCAI 2019 tutorial "Scalable Deep Learning: from theory to practice"; https://sites.google.com/view/scalable-deep-learning-ijcai19
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


import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
#the "sparseoperations" Cython library was tested in Ubuntu 16.04. Please note that you may encounter some "solvable" issues if you compile it in Windows.
import sparseoperations
import datetime


def backpropagation_updates_Numpy(a, delta, rows, cols, out):
    for i in range(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createSparseWeights(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float64(np.random.randn() / 10)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(A, B):
    # this are for array intersection
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)], 'formats': ncols * [A.dtype]}
    return np.in1d(A.view(dtype), B.view(dtype))  # boolean return


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
        return np.mean((y_pred - y_true) ** 2)

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


class FixProb_MLP:
    def __init__(self, dimensions, activations, epsilon=20):
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
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.droprate = 0  # dropout rate
        self.dimensions = dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = createSparseWeights(self.epsilon, dimensions[i],
                                                dimensions[i + 1])  # create sparse weight matrices
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

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

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            if (drop == False):
                if (i > 1):
                    z[i + 1] = z[i + 1] * (1 - self.droprate)
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if (drop):
                if (i < self.n_layers - 1):
                    dropMask = np.random.rand(a[i + 1].shape[0], a[i + 1].shape[1])
                    dropMask[dropMask >= self.droprate] = 1
                    dropMask[dropMask < self.droprate] = 0
                    a[i + 1] = dropMask * a[i + 1]

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
        dw = coo_matrix(self.w[self.n_layers - 1])

        # compute backpropagation updates
        sparseoperations.backpropagation_updates_Cython(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        # backpropagation_updates_Numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), delta)
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
            dw = coo_matrix(self.w[i - 1])

            # compute backpropagation updates
            sparseoperations.backpropagation_updates_Cython(a[i - 1], delta, dw.row, dw.col, dw.data)
            # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
            # backpropagation_updates_Numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

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

        # perform the update with momentum
        if (index not in self.pdw):
            self.pdw[index] = -self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * np.mean(delta, 0)
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * np.mean(delta, 0)

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, x_test, y_test, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002,  dropoutrate=0, testing=True, save_filename=""):
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
        :param droprate: (flt)
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.droprate = dropoutrate

        maximum_accuracy = 0

        metrics = np.zeros((epochs, 4))

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            # training
            t1 = datetime.datetime.now()

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l], True)


                self._back_prop(z, a, y_[k:l])

            t2 = datetime.datetime.now()

            print("\nFixProb-MLP Epoch ", i)
            print("Training time: ", t2 - t1)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.predict(x_test, y_test, batch_size)
                accuracy_train, activations_train = self.predict(x, y_true, batch_size)
                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.loss.loss(y_test, activations_test)
                loss_train = self.loss.loss(y_true, activations_train)
                metrics[i, 0] = loss_train
                metrics[i, 1] = loss_test
                metrics[i, 2] = accuracy_train
                metrics[i, 3] = accuracy_test
                print("Testing time: ", t4 - t3,"; Loss train: ", loss_train, "; Loss test: ", loss_test, "; Accuracy train: ", accuracy_train,"; Accuracy test: ", accuracy_test,
                      "; Maximum accuracy test: ", maximum_accuracy)

                      # save performance metrics values in a file
            if (save_filename != ""):
                np.savetxt(save_filename, metrics)

        return metrics


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
            _, a_test = self._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]
        correctClassification = 0
        for j in range(y_test.shape[0]):
            if (np.argmax(activations[j]) == np.argmax(y_test[j])):
                correctClassification += 1
        accuracy = correctClassification / y_test.shape[0]
        return accuracy, activations

def load_fashion_mnist_data(noTrainingSamples,noTestingSamples):
    np.random.seed(0)

    data=np.load("data/fashion_mnist.npz")

    indexTrain=np.arange(data["X_train"].shape[0])
    np.random.shuffle(indexTrain)

    indexTest=np.arange(data["X_test"].shape[0])
    np.random.shuffle(indexTest)

    X_train=data["X_train"][indexTrain[0:noTrainingSamples],:]
    Y_train=data["Y_train"][indexTrain[0:noTrainingSamples],:]
    X_test=data["X_test"][indexTest[0:noTestingSamples],:]
    Y_test=data["Y_test"][indexTest[0:noTestingSamples],:]

    #normalize in 0..1
    X_train = X_train.astype('float64') / 255.
    X_test = X_test.astype('float64') / 255.

    return X_train,Y_train,X_test,Y_test

if __name__ == "__main__":

    for i in range(1):
        #load data
        noTrainingSamples=2000 #max 60000 for Fashion MNIST
        noTestingSamples = 1000  # max 10000 for Fashion MNIST
        X_train, Y_train, X_test, Y_test = load_fashion_mnist_data(noTrainingSamples,noTestingSamples)

        #set model parameters
        noHiddenNeuronsLayer=1000
        epsilon=13 #set the sparsity level
        noTrainingEpochs=400
        batchSize=40
        dropoutRate=0.2
        learningRate=0.05
        momentum=0.9
        weightDecay=0.0002

        np.random.seed(i)

        # create FixProb-MLP (MLP with static sparse connectivity)
        fixprob_mlp = FixProb_MLP((X_train.shape[1], noHiddenNeuronsLayer, noHiddenNeuronsLayer,noHiddenNeuronsLayer, Y_train.shape[1]), (Relu, Relu,Relu, Sigmoid), epsilon=epsilon)

        # train FixProb-MLP
        fixprob_mlp.fit(X_train, Y_train, X_test, Y_test, loss=MSE, epochs=noTrainingEpochs, batch_size=batchSize, learning_rate=learningRate,
                    momentum=momentum, weight_decay=weightDecay, dropoutrate=dropoutRate, testing=True,
                    save_filename="Results/fixprob_mlp_"+str(noTrainingSamples)+"_training_samples_e"+str(epsilon)+"_rand"+str(i)+".txt")

        # test FixProb-MLP
        accuracy, _ = fixprob_mlp.predict(X_test, Y_test, batch_size=1)

        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
