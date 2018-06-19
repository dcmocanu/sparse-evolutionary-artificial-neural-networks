# Author: Decebal Constantin Mocanu et al.;
# Proof of concept implementation  of a Restricted Boltzmann Machine (RBM) with a fix sparsity pattern (FixProb) on COIL20 dataset using Python, SciPy sparse data structures, and (optionally) Cython.
# This implementation can be used to create RBM-FixProb with hundred of thousands of neurons.

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

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix
import sparseoperations
import datetime
import scipy.io as sio
import matplotlib.pyplot as plt

def contrastive_divergence_updates_Numpy(wDecay, lr, DV, DH, MV, MH, rows, cols, out):
    for i in range (out.shape[0]):
        s1=0
        s2=0
        for j in range(DV.shape[0]):
            s1+=DV[j,rows[i]]*DH[j, cols[i]]
            s2+=MV[j,rows[i]]*MH[j, cols[i]]
        out[i]+=lr*(s1/DV.shape[0]-s2/DV.shape[0])-wDecay*out[i]
    #return out

def createSparseWeights(epsilon,noRows,noCols):
    # generate an Erdos Renyi sparse weights mask
    weights=lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0,noRows),np.random.randint(0,noCols)]=np.float64(np.random.randn()/20)
    print ("Create sparse matrix with ",weights.getnnz()," connections and ",(weights.getnnz()/(noRows * noCols))*100,"% sparsity level")
    weights=weights.tocsr()
    return weights

class Sigmoid:
    @staticmethod
    def activation(z):

        return 1 / (1 + np.exp(-z))

    def activationStochastic(z):
        z=Sigmoid.activation(z)
        za=z.copy()
        prob=np.random.uniform(0,1,(z.shape[0],z.shape[1]))
        za[za>prob]=1
        za[za<=prob]=0
        return za


class RBM_FixProb:
    def __init__(self, noVisible, noHiddens,epsilon=10):
        self.noVisible = noVisible #number of visible neurons
        self.noHiddens=noHiddens # number of hidden neurons
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper

        self.learning_rate = None
        self.weight_decay = None

        self.W=createSparseWeights(self.epsilon,self.noVisible,self.noHiddens) # create weights sparse matrix
        self.bV=np.zeros(self.noVisible) #biases of the visible neurons
        self.bH = np.zeros(self.noHiddens) #biases of the hidden neurons

    def fit(self, X_train, X_test, batch_size,epochs,lengthMarkovChain=2,weight_decay=0.0000002,learning_rate=0.1, testing=True, save_filename=""):

        # set learning parameters
        self.lengthMarkovChain=lengthMarkovChain
        self.weight_decay=weight_decay
        self.learning_rate=learning_rate


        minimum_reconstructin_error=100000
        metrics=np.zeros((epochs,2))
        reconstruction_error_train=0

        for i in range (epochs):
            # Shuffle the data
            seed = np.arange(X_train.shape[0])
            np.random.shuffle(seed)
            x_ = X_train[seed]

            # training
            t1 = datetime.datetime.now()
            for j in range(x_.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                reconstruction_error_train+=self.learn(x_[k:l])
            t2 = datetime.datetime.now()

            reconstruction_error_train=reconstruction_error_train/(x_.shape[0] // batch_size)
            metrics[i, 0] = reconstruction_error_train
            print ("\nRBM-FixProb Epoch ",i)
            print ("Training time: ",t2-t1,"; Reconstruction error train: ",reconstruction_error_train)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if (testing):
                t3 = datetime.datetime.now()
                reconstruction_error_test=self.reconstruct(X_test)
                t4 = datetime.datetime.now()
                metrics[i, 1] = reconstruction_error_test
                minimum_reconstructin_error = min(minimum_reconstructin_error, reconstruction_error_test)
                print("Testing time: ", t4 - t3, "; Reconstruction error test: ", reconstruction_error_test,"; Minimum reconstruction error: ", reconstruction_error_test)

            #save performance metrics values in a file
            if (save_filename!=""):
                np.savetxt(save_filename,metrics)

    def runMarkovChain(self,x):
        self.DV=x
        self.DH=self.DV@self.W  + self.bH
        self.DH=Sigmoid.activationStochastic(self.DH)

        for i in range(1,self.lengthMarkovChain):
            if (i==1):
                self.MV = self.DH @ self.W.transpose() + self.bV
            else:
                self.MV = self.MH @ self.W.transpose() + self.bV
            self.MV = Sigmoid.activation(self.MV)
            self.MH=self.MV@self.W  + self.bH
            self.MH = Sigmoid.activationStochastic(self.MH)

    def reconstruct(self,x):
        self.runMarkovChain(x)
        return (np.mean((self.DV-self.MV)*(self.DV-self.MV)))

    def learn(self,x):
        self.runMarkovChain(x)
        self.update()
        return (np.mean((self.DV - self.MV) * (self.DV - self.MV)))

    def getRecontructedVisibleNeurons(self,x):
        #return recontructions of the visible neurons
        self.reconstruct(x)
        return self.MV

    def getHiddenNeurons(self,x):
        # return hidden neuron values
        self.reconstruct(x)
        return self.MH


    def update(self):
        #computer Contrastive Divergence updates
        self.W=self.W.tocoo()
        sparseoperations.contrastive_divergence_updates_Cython(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)
        # If you have problems with Cython please use the backpropagation_updates_Numpy method by uncommenting the line below and commenting the one above. Please note that the running time will be much higher
        #contrastive_divergence_updates_Numpy(self.weight_decay, self.learning_rate, self.DV, self.DH, self.MV, self.MH, self.W.row, self.W.col, self.W.data)

        # perform the weights update
        # TODO: adding momentum would make learning faster
        self.W=self.W.tocsr()
        self.bV=self.bV+self.learning_rate*(np.mean(self.DV,axis=0)-np.mean(self.MV,axis=0))-self.weight_decay*self.bV
        self.bH = self.bH + self.learning_rate * (np.mean(self.DH, axis=0) - np.mean(self.MH, axis=0)) - self.weight_decay * self.bH

if __name__ == "__main__":
    # Comment this if you would like to use the full power of randomization. I use it to have repeatable results.
    np.random.seed(0)

    # load data
    mat = sio.loadmat('data/COIL20.mat')
    X = mat['X']
    Y=mat['Y']  # the labels are, in fact, not used in this demo

    #split data in training and testing
    indices=np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_train=X[indices[0:int(X.shape[0]*2/3)]]
    Y_train=Y[indices[0:int(X.shape[0]*2/3)]]
    X_test=X[indices[int(X.shape[0]*2/3):]]
    Y_test=Y[indices[int(X.shape[0]*2/3):]]

    #these data are already normalized in the [0,1] interval. If you use other data you would have to normalize them
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')

    # create RBM-FixProb
    rbm_fixprob=RBM_FixProb(X_train.shape[1],noHiddens=200,epsilon=10)

    # train RBM-FixProb
    rbm_fixprob.fit(X_train, X_test, batch_size=10,epochs=1000,lengthMarkovChain = 2, weight_decay = 0.0000002, learning_rate = 0.1, testing = True, save_filename = "Results/rbm_fixprob.txt")

    # get reconstructed data
    reconstructions=rbm_fixprob.getRecontructedVisibleNeurons(X_test)
    print ("\nReconstruction error of the last epoch on the testing data: ",np.mean((reconstructions-X_test)*(reconstructions-X_test)))

    # get hidden neurons values to be used, for instance, with a classifier
    hiddens=rbm_fixprob.getHiddenNeurons(X_test)
