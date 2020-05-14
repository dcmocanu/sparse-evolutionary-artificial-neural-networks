### Optimizers used to update master process weights

import numpy as np
import logging
import scipy.sparse as sparse

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError


class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr):
        super(VanillaSGD, self).__init__()
        self.learning_rate = lr

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            # try:
            #     dw = retain_valid_updates(weights['w'][index], dw)
            # except:
            #     return weights

            weights['pdw'][index] = - self.learning_rate * dw
            weights['pdd'][index] = - self.learning_rate * delta

            weights['w'][index] += weights['pdw'][index]
            weights['b'][index] += weights['pdd'][index]

        return weights


class MomentumSGD(Optimizer):
    """Stochastic gradient descent with momentum and weight decay
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, lr, weight_decay, momentum):
        super(MomentumSGD, self).__init__()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            # try:
            #     dw = retain_valid_updates(weights['w'][index], dw)
            # except:
            #     return weights

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = - self.learning_rate * dw
                weights['pdd'][index] = - self.learning_rate * delta
            else:
                weights['pdw'][index] = self.momentum * weights['pdw'][index] - self.learning_rate * dw
                weights['pdd'][index] = self.momentum * weights['pdd'][index] - self.learning_rate * delta

            weights['w'][index] += weights['pdw'][index] #- self.weight_decay * weights['w'][index]
            weights['b'][index] += weights['pdd'][index] #- self.weight_decay * weights['b'][index]

        return weights


class GEM(Optimizer):
    """GEM optimizer
        learning_rate: base learning rate, kept constant
        momentum: momentum term, constant
        kappa: Proxy amplification. Experimental results show 2 is a good value.
        """

    def __init__(self, learning_rate=0.01, momentum=0.9, kappa=1.0):
        super(GEM, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.kappa = kappa
        self.epsilon = 1e-16

        self.central_variable_moment = {'w': {}, 'b': {}}
        self.stale = {'w': {}, 'b': {}}
        self.moment = {'pdw': {}, 'pdd': {}}

    def begin_compute_update(self, gradient):

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            if index not in self.moment['pdw']:
                self.moment['pdw'][index] = - self.learning_rate * dw
                self.moment['pdd'][index] = - self.learning_rate * delta
            else:
                self.moment['pdw'][index] = self.momentum * self.moment['pdw'][index] - self.learning_rate * dw
                self.moment['pdd'][index] = self.momentum * self.moment['pdd'][index] - self.learning_rate * delta

    def gradient_energy_matching(self, gradient):
        update_gem = {}

        for idx, v in gradient.items():
            dw = - self.learning_rate * v[0]
            delta = - self.learning_rate * v[1]

            proxy = self.kappa * np.abs(self.moment['pdw'][idx])

            central_variable = np.abs(self.central_variable_moment['w'][idx])

            update = np.abs(dw)
            pi_w = sparse_divide_nonzero(proxy - central_variable, update)
            pi_w.data[np.isnan(pi_w.data)] = self.epsilon
            np.clip(pi_w.data, 0., 5., out=pi_w.data)  # For numerical stability.

            proxy = self.kappa * np.abs(self.moment['pdd'][idx])
            central_variable = np.abs(self.central_variable_moment['b'][idx])
            update = np.abs(delta)
            pi_b = (proxy - central_variable) / (update + self.epsilon)
            np.clip(pi_b, 0., 5., out=pi_b)  # For numerical stability.

            update_gem[idx] = pi_w.multiply(dw), pi_b * delta

        return update_gem

    def compute_update(self, weights, gradient):

        for idx, b in weights['b'].items():
            if idx in self.stale['b']:
                self.central_variable_moment['b'][idx] = (b - self.stale['b'][idx])
            else:
                self.central_variable_moment['b'][idx] = np.zeros_like(b)
            self.stale['b'][idx] = np.copy(b)

        for idx, w in weights['w'].items():
            if idx in self.stale['w']:
                self.central_variable_moment['w'][idx] = (w - self.stale['w'][idx])
            else:
                self.central_variable_moment['w'][idx] = sparse.csr_matrix(w.shape, dtype='float64')
            self.stale['w'][idx] = w.copy()

        update = self.gradient_energy_matching(gradient)

        return update

    def apply_update(self, weights, gradient):
        """Add the update to the weights."""

        for index, v in gradient.items():
            dw = v[0]
            delta = v[1]

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = dw
                weights['pdd'][index] = delta
            else:
                weights['pdw'][index] = self.momentum * weights['pdw'][index] + dw
                weights['pdd'][index] = self.momentum * weights['pdd'][index] + delta

            weights['w'][index] += weights['pdw'][index]  # - self.weight_decay * weights['w'][index]
            weights['b'][index] += weights['pdd'][index]  # - self.weight_decay * weights['b'][index]

        return weights


def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / (inv_b.data + 1e-16)
    return a.multiply(inv_b)


def get_optimizer(name):
    """Get optimizer class by string identifier"""
    lookup = {
            # Native optimizers
            'sgd':           VanillaSGD,
            'sgdm':          MomentumSGD,
            'gem':           GEM,
            }
    return lookup[name]


def retain_valid_updates(weights, gradient):
    cols = gradient.shape[1]
    Ia, Ja, Va = sparse.find(weights)
    Ib, Jb, Vb = sparse.find(gradient)
    Ka = np.array(Ia * cols + Ja)
    Kb = np.array(Ib * cols + Jb)

    indices = np.setdiff1d(Kb, Ka, assume_unique=True)
    if len(indices) != 0:
        raise AssertionError()
        rows, cols = np.unravel_index(indices, gradient.shape)
        gradient[rows, cols] = 0
        gradient.eliminate_zeros()

    return gradient


def retain_valid_weights(correct_weights, new_weights):
    cols = new_weights.shape[1]
    Ia, Ja, Va = sparse.find(correct_weights)
    Ib, Jb, Vb = sparse.find(new_weights)
    Ka = Ia * cols + Ja
    Kb = Ib * cols + Jb

    indices = np.setdiff1d(Kb, Ka, assume_unique=True)
    if len(indices) != 0:
        rows, cols = np.unravel_index(indices, new_weights.shape)
        correct_weights = correct_weights.tolil()
        correct_weights[rows, cols] = new_weights[rows, cols]

    return correct_weights.tocsr()


class OptimizerBuilder(object):
    """Builds a  optimizer"""

    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        if self.name == 'sgd' and 'lr' not in self.config:
            logging.warning("Learning rate for SGD not set, using 0.1")
            self.config['lr'] = 0.1

    def build(self):
        from keras.optimizers import deserialize
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        return opt