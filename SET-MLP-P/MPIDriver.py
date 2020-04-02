import argparse
import logging
from set_mlp import *
from keras.datasets import cifar10

from mpi4py import MPI
from time import time
from manager import MPIManager
from train.algo import Algo
from train.data import Data
from train.model import MPIModel
from logger import initialize_logger

# Debugging with size > 1
# size = MPI.COMM_WORLD.Get_size()
# rank = MPI.COMM_WORLD.Get_rank()
# import pydevd_pycharm
# port_mapping = [56131, 56135]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)


def shared_partitions(n, num_workers, batch_size):
    # remove last data point
    dinds = list(range(n))
    num_batches = n // batch_size
    worker_size = num_batches // num_workers

    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        data[w] = dinds[w * batch_size * worker_size: (w+1) * batch_size * worker_size]

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configuration of network topology
    parser.add_argument('--masters', help='number of master processes', default=1, type=int)
    parser.add_argument('--processes', help='number of processes per worker', default=1, type=int)
    parser.add_argument('--synchronous', help='run in synchronous mode', action='store_true')

    # Configuration of training process
    parser.add_argument('--optimizer', help='optimizer for master to use', default='sgd')
    parser.add_argument('--loss', help='loss function', default='mse')
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
                        'One of "sgd" (Stohastic Gradient Descent), "easgd" (Elastic Averaging SGD) or '
                        '"gem" (Gradient Energy Matching)',
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

    # logging configuration
    parser.add_argument('--log-file', default=None, dest='log_file',
                        help='log file to write, in additon to output stream')
    parser.add_argument('--log-level', default='info', dest='log_level', help='log level (debug, info, warn, error)')

    # Model configuration
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20,  help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate (default: 0.01)')
    parser.add_argument('--lr-rate-decay', type=float, default=0.0, help='learning rate decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=0.0002, help='Dropout rate')
    parser.add_argument('--epsilon', type=int, default=20, help='Sparsity level')
    parser.add_argument('--zeta', type=float, default=0.3,
                        help='It gives the percentage of unimportant connections which are removed and replaced with '
                             'random ones after every epoch(in [0..1])')
    parser.add_argument('--n-neurons', type=int, default=3000, help='Number of neurons in the hidden layer')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-training-samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--n-testing-samples', type=int, default=10000, help='Number of testing samples')

    args = parser.parse_args()

    # Initialize logger
    initialize_logger(filename=args.log_file, file_level=args.log_level, stream_level=args.log_level)

    # SET parameters
    model_config = {
        'n_processes': args.processes,
        'n_epochs': args.epsilon,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'seed': 1,
        'lr': args.lr,
        'lr_decay': args.lr_rate_decay,
        'zeta': args.zeta,
        'epsilon': args.epsilon,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'n_hidden_neurons': args.n_neurons,
        'n_training_samples': args.n_training_samples,
        'n_testing_samples': args.n_testing_samples,
        'loss': args.loss
    }

    # Load dataset
    X_train, Y_train, X_test, Y_test = load_cifar10_data(args.n_training_samples, args.n_testing_samples)

    comm = MPI.COMM_WORLD.Dup()
    partition = shared_partitions(args.n_training_samples, comm.Get_size()-1, args.batch_size)

    model_weights = None
    rank = comm.Get_rank()
    if rank != 0:
        data = Data(batch_size=args.batch_size, x_train=X_train, y_train=Y_train,
                    x_test=X_test, y_test=Y_test)
    else:
        data = Data(batch_size=args.batch_size, x_train=None,
                    y_train=None,
                    x_test=X_test, y_test=Y_test)

    validate_every = int(X_train.shape[0] / args.batch_size)
    del X_train, Y_train, X_test, Y_test

    # Some input arguments may be ignored depending on chosen algorithm
    algo = Algo(optimizer='sgd', loss='binary_crossentropy', validate_every=validate_every,
                sync_every=1, worker_optimizer='sgd',
                worker_optimizer_params={})

    dimensions = (3072, 4000, 1000, 4000, 10)
    model = MPIModel(model=SET_MLP(dimensions, (Relu, Relu, Relu, Sigmoid), **model_config))

    # Creating the MPIManager object causes all needed worker and master nodes to be created
    manager = MPIManager(comm=comm, data=data, algo=algo, model=model,
                         num_epochs=args.epochs, num_masters=1,
                         num_processes=args.processes,
                         synchronous=True,
                         verbose=True, monitor=False,
                         early_stopping=False,
                         target_metric=None,
                         thread_validation=False
    )

    # Process 0 launches the training procedure
    if rank == 0:
        logging.debug('Training configuration: %s', algo.get_config())

        t_0 = time()
        histories = manager.process.train()
        delta_t = time() - t_0
        manager.free_comms()
        logging.info("Training finished in {0:.3f} seconds".format(delta_t))

    comm.barrier()
    logging.info("Terminating")