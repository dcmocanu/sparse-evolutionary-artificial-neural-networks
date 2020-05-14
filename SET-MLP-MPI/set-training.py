from models.set_mlp_sequential import *
import time
import argparse
import os
from utils.load_data import *

# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training settings
parser = argparse.ArgumentParser(description='SET Parallel Training ')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=3000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-rate-decay', type=float, default=0.0, metavar='LRD',
                    help='learning rate decay (default: 0)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
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
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--n-training-samples', type=int, default=50000, metavar='N',
                    help='Number of training samples')
parser.add_argument('--n-testing-samples', type=int, default=10000, metavar='N',
                    help='Number of testing samples')
parser.add_argument('--n-validation-samples', type=int, default=10000, help='Number of validation samples')
parser.add_argument('--n-processes', type=int, default=6, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


if __name__ == "__main__":
    args = parser.parse_args()

    for i in range(1):
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

        # Prepare config object for the parameter server
        config = {
            'n_processes': n_processes,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'dropout_rate': dropout_rate,
            'seed': i,
            'lr': learning_rate,
            'lr_decay': learning_rate_decay,
            'zeta': zeta,
            'epsilon': epsilon,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'n_hidden_neurons': n_hidden_neurons,
            'n_training_samples': n_training_samples,
            'n_testing_samples': n_testing_samples,
            'loss': 'cross_entropy'
        }


        np.random.seed(5)

        # Load augmented dataset
        start_time = time.time()
        X_train, Y_train, X_test, Y_test = load_cifar10_data(args.n_training_samples, args.n_testing_samples)
        step_time = time.time() - start_time
        print("Loading augmented dataset time: ", step_time)

        # Load basic cifar10 dataset
        # X_train, Y_train, X_test, Y_test = load_cifar10_data(n_training_samples, n_testing_samples)

        # Create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        print("Number of neurons per layer:", X_train.shape[1], n_hidden_neurons, n_hidden_neurons,
        n_hidden_neurons, Y_train.shape[1])

        set_mlp = SET_MLP((X_train.shape[1], 4000, 1000, 4000,
                           Y_train.shape[1]), (Relu, Relu, Relu, Softmax), **config)
        start_time = time.time()
        set_mlp.fit(X_train, Y_train, X_test, Y_test, testing=True,
                    save_filename=r"Results2/fast2_set_mlp_sequential_cifar10_" +
                                  str(n_training_samples) + "_training_samples_e" + str(
                        epsilon) + "_rand" + str(i))
        step_time = time.time() - start_time
        print("\nTotal training time: ", step_time)

        # Test SET-MLP
        accuracy, _ = set_mlp.predict(X_test, Y_test, batch_size=1)
        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
