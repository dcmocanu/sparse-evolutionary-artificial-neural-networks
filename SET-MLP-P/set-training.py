from set_mlp import *
from parameter_server import *
import time
import argparse
from parameter_server import *
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# **** change the warning level ****
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training settings
parser = argparse.ArgumentParser(description='SET Parallel Training ')
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
parser.add_argument('--dropout-rate', type=float, default=0.2, metavar='D',
                    help='Dropout rate')
parser.add_argument('--weight-decay', type=float, default=0.0002, metavar='W',
                    help='Dropout rate')
parser.add_argument('--epsilon', type=int, default=13, metavar='E',
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
parser.add_argument('--n-processes', type=int, default=15, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


cur_dir = os.path.dirname(os.path.abspath(__file__))
path_to_data = ['cifar10']
images_dirs = os.path.join(cur_dir, *path_to_data)


def load_images(curr_dir, label):
    print(f"Loading class {label} images ...")
    class_dir = os.path.join(images_dirs, curr_dir)

    x_train = []
    y_train = []

    # iterate through the names of contents of the folder
    for image_path in os.listdir(class_dir):
        # create the full input path and read the file
        input_path = os.path.join(class_dir, image_path)
        image = Image.open(input_path)
        x_train.append(np.asarray(image))
        y_train.append(label)

    x_train = np.asarray(x_train).reshape((-1, 32, 32, 3))
    y_train = np.asarray(y_train).flatten()

    print(f"Finished loading for class {label} images ...")
    return x_train, y_train


def load_augmented_cifar10_parallel(n_train_samples, n_test_samples):
    class_dirs = os.listdir(images_dirs)

    x_train = np.array([]).reshape((-1, 32, 32, 3))
    y_train = np.array([])

    # Loop through the data folders with training data
    with ProcessPoolExecutor() as executor:
        results = executor.map(load_images, class_dirs, range(10))
        for i, res in enumerate(results):
            x_train = np.concatenate((x_train, res[0]), axis=0)
            y_train = np.concatenate((y_train, res[1]))

    (_, _), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x_train.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x_train[index_train[0:n_train_samples], :]
    y_train = y_train[index_train[0:n_train_samples], :]
    x_test = x_test[index_test[0:n_test_samples], :]
    y_test = y_test[index_test[0:n_test_samples], :]

    # normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float64')
    x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float64')
    return x_train, y_train, x_test, y_test


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
            'delay': 1,
            'delay_type': 'const',  # const or random
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
        }

        np.random.seed(i)

        # Load augmented dataset
        start_time = time.time()
        X_train, Y_train, X_test, Y_test = load_augmented_cifar10_parallel(n_training_samples, n_testing_samples)
        step_time = time.time() - start_time
        print("Loading augmented dataset time: ", step_time)

        # Load basic cifar10 dataset
        # X_train, Y_train, X_test, Y_test = load_cifar10_data(n_training_samples, n_testing_samples)

        # create SET-MLP (MLP with adaptive sparse connectivity trained with Sparse Evolutionary Training)
        # print("Number of neurons per layer:", X_train.shape[1], n_hidden_neurons, n_hidden_neurons,
        # n_hidden_neurons, Y_train.shape[1])
        # Train SET
        # set_mlp = SET_MLP((X_train.shape[1], n_hidden_neurons, n_hidden_neurons, n_hidden_neurons,
        #                    Y_train.shape[1]), (Relu, Relu, Relu, Sigmoid), **config)
        # train SET-MLP
        # set_mlp.fit(X_train, Y_train, X_test, Y_test, batch_size, testing=True,
        #             save_filename="Results/set_mlp_" + str(n_training_samples) + "_training_samples_e" + str(
        #                 epsilon) + "_rand" + str(i))

        ps = ParameterServer(**config)

        # Initialize workers on the server
        ps.initiate_workers()

        start_time = time.time()
        ps.train()
        step_time = time.time() - start_time
        print("\nTotal training time: ", step_time)

        # test SET-MLP
        accuracy, _ = ps.predict(X_test, Y_test, batch_size=1)
        print("\nAccuracy of the last epoch on the testing data: ", accuracy)
