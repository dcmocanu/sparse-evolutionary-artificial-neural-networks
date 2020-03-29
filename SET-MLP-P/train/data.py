### Data class and associated helper methods

import numpy as np
import h5py
import os
import time
from threading import Thread
import itertools
import logging


class FilePreloader(Thread):
    def __init__(self, files_list, file_open, n_ahead=2):
        Thread.__init__(self)
        self.deamon = True
        self.n_concurrent = n_ahead
        self.files_list = files_list
        self.file_open = file_open
        self.loaded = {}  ## a dict of the loaded objects
        self.should_stop = False

    def getFile(self, name):
        ## locks until the file is loaded, then return the handle
        return self.loaded.setdefault(name, self.file_open(name))

    def closeFile(self, name):
        ## close the file and
        if name in self.loaded:
            self.loaded.pop(name).close()

    def run(self):
        while not self.files_list:
            time.sleep(1)
        for name in itertools.cycle(self.files_list):
            if self.should_stop:
                break
            n_there = len(self.loaded.keys())
            if n_there < self.n_concurrent:
                logging.debug("preloading %s with %d", name, n_there)
                self.getFile(name)
            else:
                time.sleep(5)

    def stop(self):
        logging.debug("Stopping FilePreloader")
        self.should_stop = True


def data_class_getter(name):
    """Returns the specified Data class"""
    data_dict = {
        "H5Data": H5Data,
    }
    try:
        return data_dict[name]
    except KeyError:
        logging.warning("{0:s} is not a known Data class. Returning None...".format(name))
        return None


class Data(object):
    """Class providing an interface to the input training data.
        Derived classes should implement the load_data function.
        Attributes:
          file_names: list of data files to use for training
          batch_size: size of training batches
    """

    def __init__(self, batch_size, cache=None, s3=None):
        """Stores the batch size and the names of the data files to be read.
            Params:
              batch_size: batch size for training
        """
        self.batch_size = batch_size
        self.caching_directory = cache if cache else os.environ.get('GANINMEM', '')
        self.use_s3 = s3 if s3 else os.environ.get('USES3', '')
        self.fpl = None

    def set_caching_directory(self, cache):
        self.caching_directory = cache

    def set_file_names(self, file_names):
        ## hook to copy data in /dev/shm
        relocated = []
        if self.caching_directory:
            goes_to = self.caching_directory
            goes_to += str(os.getpid())
            os.system('mkdir %s ' % goes_to)
            os.system('rm %s/* -f' % goes_to)  ## clean first if anything
            for fn in file_names:
                relocate = goes_to + '/' + fn.split('/')[-1]
                if not os.path.isfile(relocate):
                    logging.info("copying %s to %s", fn, relocate)
                    if (self.use_s3):
                        if os.system('s3cmd get s3://gan-bucket/%s %s' % (fn, relocate)) == 0:
                            relocated.append(relocate)
                        else:
                            logging.info("was unable to copy the file s3://ganbucket/%s to %s", fn, relocate)
                            relocated.append(fn)  ## use the initial one
                    else:
                        if os.system('cp %s %s' % (fn, relocate)) == 0:
                            relocated.append(relocate)
                        else:
                            logging.info("was enable to copy the file %s to %s", fn, relocate)
                            relocated.append(fn)  ## use the initial one
                else:
                    relocated.append(relocate)

            self.file_names = relocated
        else:
            self.file_names = file_names

        if self.fpl:
            self.fpl.files_list = self.file_names

    def inf_generate_data(self):
        while True:
            try:
                for B in self.generate_data():
                    yield B
            except StopIteration:
                logging.warning("start over generator loop")

    def generate_data(self):
        """Yields batches of training data until none are left."""
        leftovers = None
        for cur_file_name in self.file_names:
            cur_file_features, cur_file_labels = self.load_data()
            # concatenate any leftover data from the previous file
            if leftovers is not None:
                cur_file_features = self.concat_data(leftovers[0], cur_file_features)
                cur_file_labels = self.concat_data(leftovers[1], cur_file_labels)
                leftovers = None
            num_in_file = self.get_num_samples(cur_file_features)

            for cur_pos in range(0, num_in_file, self.batch_size):
                next_pos = cur_pos + self.batch_size
                if next_pos <= num_in_file:
                    yield (self.get_batch(cur_file_features, cur_pos, next_pos),
                           self.get_batch(cur_file_labels, cur_pos, next_pos))
                else:
                    leftovers = (self.get_batch(cur_file_features, cur_pos, num_in_file),
                                 self.get_batch(cur_file_labels, cur_pos, num_in_file))

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
            cur_file_features, cur_file_labels = self.load_data()
            num_data += self.get_num_samples(cur_file_features)
        return num_data

    def is_numpy_array(self, data):
        return isinstance(data, np.ndarray)

    def get_batch(self, data, start_pos, end_pos):
        """Input: a numpy array or list of numpy arrays.
            Gets elements between start_pos and end_pos in each array"""
        if self.is_numpy_array(data):
            return data[start_pos:end_pos]
        else:
            return [arr[start_pos:end_pos] for arr in data]

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
           Returns: numpy array or list of arrays, in which each array in data1 has been
             concatenated with the corresponding array in data2"""
        if self.is_numpy_array(data1):
            return np.concatenate((data1, data2))
        else:
            return [self.concat_data(d1, d2) for d1, d2 in zip(data1, data2)]

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])

    def load_data(self):
        """Input: name of file from which the data should be loaded
            Returns: tuple (X,Y) where X and Y are numpy arrays containing features
                and labels, respectively, for all data in the file
            Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError


class H5Data(Data):
    """Loads data stored in hdf5 files
        Attributes:
          features_name, labels_name: names of the datasets containing the features
          and labels, respectively
    """

    def __init__(self, batch_size,
                 X, Y,
                 features_name='features', labels_name='labels'):
        """Initializes and stores names of feature and label datasets"""
        super(H5Data, self).__init__(batch_size)
        self.features_name = features_name
        self.labels_name = labels_name
        self.X = X
        self.Y = Y

    def load_data(self):
        return self.X, self.Y

    def load_hdf5_data(self, data):
        """Returns a numpy array or (possibly nested) list of numpy arrays
            corresponding to the group structure of the input HDF5 data.
            If a group has more than one key, we give its datasets alphabetically by key"""
        if hasattr(data, 'keys'):
            out = [self.load_hdf5_data(data[key]) for key in sorted(data.keys())]
        else:
            out = data[:]
        return out

    def count_data(self):
        """This is faster than using the parent count_data
            because the datasets do not have to be loaded
            as numpy arrays"""
        num_data = 0
        for in_file_name in self.file_names:
            h5_file = h5py.File(in_file_name, 'r')
            X = h5_file[self.features_name]
            if hasattr(X, 'keys'):
                num_data += len(X[X.keys()[0]])
            else:
                num_data += len(X)
            h5_file.close()
        return num_data

    def finalize(self):
        if self.fpl:
            self.fpl.stop()