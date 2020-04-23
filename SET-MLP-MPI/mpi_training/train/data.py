### Data class and associated helper methods

import numpy as np
import logging


class Data(object):
    """Class providing an interface to the input training and testing data.
        Attributes:
          x_train: array of data points to use for training
          y_train: array of labels to use for training
          x_test: array of data points to use for testing
          y_test: array of labels to use for testing
          batch_size: size of training batches
    """

    def __init__(self, x_train, y_train, x_test, y_test, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size

    def inf_generate_data(self):
        while True:
            try:
                for B in self.generate_data():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")

    def generate_data(self):
        """Yields batches of training data until none are left."""
        for j in range(self.x_train.shape[0] // self.batch_size):
            start_pos = j * self.batch_size
            end_pos = (j + 1) * self.batch_size

            yield self.x_train[start_pos:end_pos], self.y_train[start_pos:end_pos]

    def generate_test_data(self):
        """Yields batches of training data until none are left."""
        for j in range(self.x_test.shape[0] // self.batch_size):
            start_pos = j * self.batch_size
            end_pos = (j + 1) * self.batch_size

            yield self.x_test[start_pos:end_pos], self.y_test[start_pos:end_pos]

    def count_data(self):
        return self.x_train.shape[0]

    def shuffle(self):
        seed = np.arange(self.x_train.shape[0])
        np.random.shuffle(seed)
        self.x_train = self.x_train[seed]
        self.y_train = self.y_train[seed]

    def is_numpy_array(self, data):
        return isinstance(data, np.ndarray)

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])
