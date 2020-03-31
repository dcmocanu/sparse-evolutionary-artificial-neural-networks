import sys, os
import numpy as np
import argparse
import json
import re
import logging
from set_mlp import *
from keras.datasets import cifar10

from mpi4py import MPI
from time import time, sleep
from manager import MPIManager
from train.algo import Algo
from train.data import Data
from train.model import MPIModel
from utils import import_keras
from train.trace import Trace
from logger import initialize_logger

# Debugging
# size = MPI.COMM_WORLD.Get_size()
# rank = MPI.COMM_WORLD.Get_rank()
# import pydevd_pycharm
# port_mapping = [61263, 61264, 61268]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help='display metrics for each training batch', action='store_true')
    parser.add_argument('--profile', help='profile theano code', action='store_true')
    parser.add_argument('--monitor', help='Monitor cpu and gpu utilization', action='store_true')
    parser.add_argument('--trace', help='Record timeline of activity', action='store_true')
    parser.add_argument('--tf', help='use tensorflow backend', action='store_true')
    parser.add_argument('--torch', help='use pytorch', action='store_true')
    parser.add_argument('--thread_validation', help='run a single process', action='store_true')

    # model arguments
    parser.add_argument('--trial-name', help='descriptive name for trial',
                        default='train', dest='trial_name')

    # training data arguments
    parser.add_argument('--features-name', help='name of HDF5 dataset with input features',
                        default='features', dest='features_name')
    parser.add_argument('--labels-name', help='name of HDF5 dataset with output labels',
                        default='labels', dest='labels_name')
    parser.add_argument('--batch', help='batch size', default=100, type=int)
    parser.add_argument('--preload-data', help='Preload files as we read them', default=0, type=int,
                        dest='data_preload')
    parser.add_argument('--cache-data', help='Cache the input files to a provided directory', default='',
                        dest='caching_dir')

    # configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--max-gpus', dest='max_gpus', help='max GPUs to use',
                        type=int, default=-1)
    parser.add_argument('--master-gpu', help='master process should get a gpu',
                        action='store_true', dest='master_gpu')
    parser.add_argument('--synchronous', help='run in synchronous mode', action='store_true')

    # configuration of training process
    parser.add_argument('--optimizer', help='optimizer for master to use', default='sgd')
    parser.add_argument('--loss', help='loss function', default='binary_crossentropy')
    parser.add_argument('--early-stopping', default=None,
                        dest='early_stopping', help='patience for early stopping')
    parser.add_argument('--target-metric', default=None,
                        dest='target_metric', help='Passing configuration for a target metric')
    parser.add_argument('--worker-optimizer', help='optimizer for workers to use',
                        dest='worker_optimizer', default='sgd')
    parser.add_argument('--worker-optimizer-params',
                        help='worker optimizer parameters (string representation of a dict)',
                        dest='worker_optimizer_params', default='{}')
    parser.add_argument('--sync-every', help='how often to sync weights with master',
                        default=1, type=int, dest='sync_every')
    parser.add_argument('--mode', help='Mode of operation.'
                                       'One of "sgd" (Stohastic Gradient Descent), "easgd" (Elastic Averaging SGD) or "gem" (Gradient Energy Matching)',
                        default='sgd')
    parser.add_argument('--elastic-force', help='beta parameter for EASGD', type=float, default=0.9)
    parser.add_argument('--elastic-lr', help='worker SGD learning rate for EASGD',
                        type=float, default=1.0, dest='elastic_lr')
    parser.add_argument('--elastic-momentum', help='worker SGD momentum for EASGD',
                        type=float, default=0, dest='elastic_momentum')
    parser.add_argument('--gem-lr', help='learning rate for GEM', type=float, default=0.01, dest='gem_lr')
    parser.add_argument('--gem-momentum', help='momentum for GEM', type=float, default=0.9, dest='gem_momentum')
    parser.add_argument('--gem-kappa', help='Proxy amplification parameter for GEM', type=float, default=2.0,
                        dest='gem_kappa')
    parser.add_argument('--checkpoint',
                        help='Base name of the checkpointing file. If omitted no checkpointing will be done',
                        default=None)
    parser.add_argument('--checkpoint-interval', help='Number of epochs between checkpoints', default=5, type=int,
                        dest='checkpoint_interval')

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, metavar='LRD',
                        help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, metavar='D',
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.0002, metavar='W',
                        help='Dropout rate')
    parser.add_argument('--epsilon', type=int, default=20, metavar='E',
                        help='Sparsity level')
    parser.add_argument('--zeta', type=float, default=0.3, metavar='Z',
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--n-neurons', type=int, default=3000, metavar='H',
                        help='Number of neurons in the hidden layer')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=100000, metavar='N',
                        help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, metavar='N',
                        help='Number of testing samples')
    parser.add_argument('--n-processes', type=int, default=6, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')

    args = parser.parse_args()

    initialize_logger(filename="log.txt", file_level=args.log_level, stream_level=args.log_level)

    # Set parameters
    n_hidden_neurons = args.n_neurons
    epsilon = args.epsilon
    zeta = args.zeta
    n_epochs = args.epochs
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    n_processes = args.n_processes
    n_training_samples = args.n_training_samples
    n_testing_samples = args.n_testing_samples
    learning_rate_decay = args.lr_rate_decay

    config = {
        'n_processes': n_processes,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'dropout_rate': dropout_rate,
        'seed': 1,
        'lr': learning_rate,
        'lr_decay': learning_rate_decay,
        'zeta': zeta,
        'epsilon': epsilon,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'n_hidden_neurons': n_hidden_neurons,
        'n_training_samples': n_training_samples,
        'n_testing_samples': n_testing_samples,
    }

    X_train, Y_train, X_test, Y_test = load_cifar10_data(2000, 1000)

    comm = MPI.COMM_WORLD.Dup()

    model_weights = None

    data = Data(batch_size=batch_size, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test)
    # We initialize the Data object with the training data list
    # so that we can use it to count the number of training examples
    validate_every = int(data.x_train.shape[0] / batch_size)

    # Some input arguments may be ignored depending on chosen algorithm
    algo = Algo(optimizer='sgd', loss='binary_crossentropy', validate_every=validate_every,
                sync_every=1, worker_optimizer='sgd',
                worker_optimizer_params={})

    dimensions = (3072, 4000, 1000, 4000, 10)
    model = MPIModel(model=SET_MLP(dimensions, (Relu, Relu, Relu, Sigmoid), **config))

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager(comm=comm, data=data, algo=algo, model=model,
                         num_epochs=args.epochs, x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test,
                         num_masters=1, num_processes=4,
                         synchronous=True,
                         verbose=True, monitor=False,
                         early_stopping=False,
                         target_metric=None,
                         thread_validation=False
    )

    # Process 0 launches the training procedure
    if comm.Get_rank() == 0:
        logging.debug('Training configuration: %s', algo.get_config())

        t_0 = time()
        histories = manager.process.train()
        delta_t = time() - t_0
        manager.free_comms()
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))

    comm.barrier()
    logging.info("Terminating")