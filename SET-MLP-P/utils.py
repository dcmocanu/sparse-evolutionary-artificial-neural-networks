### Utilities for mpi_learn module
import os
import sys
import numpy as np
import logging

class Error(Exception):
    pass

def weights_from_shapes(weights_shapes):
    """Returns a list of numpy arrays representing the NN architecture"""
    return [ np.zeros( shape, dtype=np.float32 ) for shape in weights_shapes ]

def shapes_from_weights(weights):
    """Returns a list of tuples indicating the array shape of each layer of the NN"""
    return [ w.shape for w in weights ]

def get_num_gpus():
    """Returns the number of GPUs available"""
    logging.debug("Determining number of GPUs...")
    from pycuda import driver
    driver.init()
    num_gpus = driver.Device.count()
    logging.debug("Number of GPUs: {}".format(num_gpus))
    return num_gpus

def get_device_name(dev_type, dev_num, backend='theano'):
    """Returns cpu/gpu device name formatted for
    theano or keras, as specified by the backend argument"""
    if backend == 'tensorflow':
        return "/%s:%d" % (dev_type, dev_num)
    else:
        if dev_type == 'cpu':
            return 'cpu'
        else:
            return dev_type+str(dev_num)

def import_keras(tries=10):
    """There is an issue when multiple processes import Keras simultaneously --
        the file .keras/keras.json is sometimes not read correctly.
        as a workaround, just try several times to import keras."""
    for try_num in range(tries):
        try:
            stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            import keras
            sys.stderr = stderr
            return
        except ValueError:
            logging.warning("Unable to import keras. Trying again: {0:d}".format(try_num))
            from time import sleep
            sleep(0.1)
    logging.error("Failed to import keras!")

def load_model(filename=None, model=None, weights_file=None, custom_objects={}):
    """Loads model architecture from JSON and instantiates the model.
        filename: path to JSON file specifying model architecture
        model:    (or) a Keras model to be cloned
        weights_file: path to HDF5 file containing model weights
	custom_objects: A Dictionary of custom classes used in the model keyed by name"""
    import_keras()
    from keras.models import model_from_json, clone_model
    if filename is not None:
        with open( filename ) as arch_f:
            json_str = arch_f.readline()
            new_model = model_from_json( json_str, custom_objects=custom_objects)
    if model is not None:
        new_model = clone_model(model)
    if weights_file is not None:
        new_model.load_weights( weights_file )
    return new_model