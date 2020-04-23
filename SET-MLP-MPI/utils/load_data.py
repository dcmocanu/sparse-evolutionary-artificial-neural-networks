### Utilities for mpi_learn module
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle
import h5py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from keras.datasets import cifar10
from keras.utils import np_utils
from PIL import Image

# Augmented dataset path
cur_dir = os.path.dirname(os.path.abspath(__file__))
path_to_data = ['..', '..', '..', 'cifar10_augmented_1M']
images_dirs = os.path.join(cur_dir, *path_to_data)


class Error(Exception):
    pass


def load_fashion_mnist_data(n_training_samples, n_testing_samples):
    np.random.seed(0)

    data = np.load("../Tutorial-IJCAI-2019-Scalable-Deep-Learning/data/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:n_training_samples], :]
    y_train = data["Y_train"][index_train[0:n_training_samples], :]
    x_test = data["X_test"][index_test[0:n_testing_samples], :]
    y_test = data["Y_test"][index_test[0:n_testing_samples], :]

    # Normalize in 0..1
    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    return x_train, y_train, x_test, y_test


def load_cifar10_1M_data():
    np.random.seed(0)

    data = np.load("../data/CIFAR10/cifar10_augmented_1M.npz", mmap_mode='r')

    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']


def load_cifar10_500K_data():
    np.random.seed(0)

    data = np.load("../data/CIFAR10/cifar10_augmented_500K.npz", mmap_mode='r')

    return data['X_train'], data['Y_train'], data['X_test'], data['Y_test']


def load_cifar10_data(n_training_samples, n_testing_samples):
    np.random.seed(0)

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_test = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float64')
    x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float64')

    return x_train, y_train, x_test, y_test


def load_cifar10_data_not_flattened(n_training_samples, n_testing_samples):
    np.random.seed(0)

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = np_utils.to_categorical(y, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    return x_train, y_train, x_test, y_test

def load_images(curr_dir, label):
    print(f"Loading class {label} images ...")
    class_dir = os.path.join(images_dirs, curr_dir)

    x_train = []
    y_train = []

    # Iterate through the images in the given the folder
    for image_path in os.listdir(class_dir):
        # Create the full input path and read the file
        input_path = os.path.join(class_dir, image_path)
        image = Image.open(input_path)
        x_train.append(np.asarray(image))
        y_train.append(label)

    x_train = np.asarray(x_train).reshape((-1, 32, 32, 3))
    y_train = np.asarray(y_train).flatten()

    print(f"Finished loading for class {label} images ...")
    return x_train, y_train


def load_augmented_cifar10_parallel():
    class_dirs = os.listdir(images_dirs)

    x_train = np.array([], dtype='float32').reshape((-1, 32, 32, 3))
    y_train = np.array([])

    # Loop through the data folders with training data
    with ProcessPoolExecutor(max_workers=12) as executor:
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

    x_train = x_train[index_train[0:1000000], :]
    y_train = y_train[index_train[0:1000000], :]
    x_test = x_test[index_test[0:10000], :]
    y_test = y_test[index_test[0:10000], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float64')
    x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float64')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_augmented_cifar10_parallel()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    np.savez_compressed('../../data/CIFAR10/cifar10_augmented_1M.npz', X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test)

    # hf = h5py.File('../../data/CIFAR10/cifar_500K.h5', 'w')
    # hf.create_dataset('x_train', data=x_train, compression='gzip')
    # hf.create_dataset('y_train', data=y_train, compression='gzip')
    # hf.create_dataset('x_test', data=x_test, compression='gzip')
    # hf.create_dataset('y_test', data=y_test, compression='gzip')
    # hf.close()

    # joblib.dump(x_train, '../../data/CIFAR10/cifar_500K_x_train.joblib', compress=3)

    # x_train.dump('../../data/CIFAR10/cifar_500K_x_train.pkl')
