### ModelBuilder class and associated helper methods

from utils import load_model, get_device_name
from .optimizer import OptimizerBuilder
import numpy as np
import copy
from set_mlp import *
import sys
import six
import logging


class MPIModel(object):
    """Class that abstract all details of the model
    """

    def __init__(self, model=None, models=None):
        self.model = model
        self.models = models
        self.histories = {}
        if model and models:
            raise Exception("Cannot specify single and multiple models")

    def print_metrics(self, metrics):
        if self.model:
            names = ['loss', 'accuracy']
            for name, metric in zip(names, metrics):
                logging.info("{0}: {1:.3f}".format(name, metric))
        else:
            for im, m in enumerate(self.models):
                names = m.metrics_names
                ametric = metrics[im, ...]
                logging.info('model {0} {1}'.format(im, m.name))
                for name, metric in zip(names, ametric):
                    logging.info("{0}: {1:.3f}".format(name, metric))

    def get_logs(self, metrics, val=False):
        if self.model:
            if val:
                return {'val_' + name: np.asscalar(metric) for name, metric in
                        zip(self.model.metrics_names, metrics)}
            else:
                return {name: np.asscalar(metric) for name, metric in
                        zip(self.model.metrics_names, metrics)}
        else:
            logs = []
            for im, m in enumerate(self.models):
                ametrics = metrics[im, ...]
                if val:
                    logs.append({'val_' + name: np.asscalar(metric) for name, metric in
                                 zip(m.metrics_names, ametrics)})
                else:
                    logs.append({name: np.asscalar(metric) for name, metric in
                                 zip(m.metrics_names, ametrics)})
            return logs

    def update_history(self, items, arg_hist):
        if self.model:
            for m, v in items.items():
                arg_hist.setdefault(m, []).append(v)
        else:
            for im, (m, it) in enumerate(zip(self.models, items)):
                m_name = "model%s" % im
                try:
                    m_name = m.name
                except:
                    logging.warning("no name attr")
                for m, v in it.items():
                    arg_hist.setdefault(m_name, {}).setdefault(m, []).append(v)
        self.histories = arg_hist

    def format_update(self):
        if not self.model.parameters()['w']:
            return {'w': {}, 'b': {}}

    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, w):
        self.model.set_parameters(w)

    def train_on_batch(self, **args):
        return np.asarray(self.model.train_on_batch(**args))

    def test_on_batch(self, **args):
        return np.asarray(self.model.test_on_batch(**args))

    def predict(self, x, y):
        return self.model.predict(x, y, self.model.batch_size)

    def compute_loss(self, y, activations):
        self.model.loss.loss(y, activations)

    def figure_of_merit(self, **args):
        ## runs like predict trace, and provides a non differentiable figure of merit for hyper-opt
        ## can of course be the validation loss
        if self.model:
            ## return a default value from the validation history
            return (1. - self.histories['val_acc'][-1])
            # return self.histories['val_loss'][-1]
        else:
            return 0.


class ModelBuilder(object):
    """Class containing instructions for building neural net models.
        Derived classes should implement the build_model and get_backend_name
        functions.
        Attributes:
            comm: MPI communicator containing all running MPI processes
    """

    def __init__(self, comm):
        """Arguments:
            comm: MPI communicator
        """
        self.comm = comm

    def get_device_name(self, device):
        """Should return a device name under a desired convention"""
        return device

    def build_model(self):
        """Should return an uncompiled Keras model."""
        raise NotImplementedError

    def get_backend_name(self):
        """Should return the name of backend framework."""
        raise NotImplementedError


class SETModel(ModelBuilder):
    """ModelBuilder class that builds from model architecture specified
        in a JSON file.
        Attributes:
            filename: path to JSON file specifying model architecture
    """

    def __init__(self, comm, **config):
        super(SETModel, self).__init__(comm)
        self.config = config

    def build_model(self, local_session=True):
        self.dimensions = (3072, 4000, 1000, 4000, 10)
        return MPIModel(model=SET_MLP(self.dimensions, (Relu, Relu, Relu, Sigmoid), **self.config))

    def get_backend_name(self):
        return 'keras'