import numpy as np
import copy
from models.set_mlp import *
import logging


class MPIModel(object):
    """Class that abstract all details of the model
    """

    def __init__(self, model=None, models=None):
        self.model = model
        self.batch_size = self.model.batch_size
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

    def format_update(self):
        if not self.model.parameters()['pdw']:
            return {'pdw': {}, 'pdd': {}}

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
        return self.model.loss.loss(y, activations)

    def weight_evolution(self):
        self.model.weightsEvolution_II()
