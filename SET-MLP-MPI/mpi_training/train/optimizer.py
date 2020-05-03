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

            dw = retain_valid_updates(weights['w'][index], dw)
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

            # perform the update with momentum
            if index not in weights['pdw']:
                weights['pdw'][index] = - self.learning_rate * dw
                weights['pdd'][index] = - self.learning_rate * delta
            else:
                dw = retain_valid_updates(weights['w'][index], dw)

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

    def __init__(self, learning_rate=0.05, momentum=0.9, kappa=1.0):
        super(GEM, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.kappa = kappa
        self.epsilon = 10e-13

        self.central_variable_moment = None
        self.stale = None
        self.moment = None

        self.tensors_initialized = False

    def init_tensors(self, weights):
        if not self.tensors_initialized:
            self.central_variable_moment = [ np.zeros_like(w) for w in weights ]
            self.stale = [ np.zeros_like(w) for w in weights ]
            self.moment = [ np.zeros_like(w) for w in weights ]

            self.tensors_initialized = True

    def begin_compute_update(self, cur_weights, new_weights):
        self.init_tensors(cur_weights)

        update = []

        for idx, (cur_w, new_w) in enumerate(zip(cur_weights, new_weights)):
            update.append( np.subtract( cur_w, new_w ))
            update[idx] *= -self.learning_rate
            # Update the states with the current gradient.
            self.moment[idx] *= self.momentum
            self.moment[idx] += update[idx]

        return update

    def gradient_energy_matching(self, gradient):
        Pi = [] # Pi tensors for all parameters.

        for idx, g in enumerate(gradient):
            proxy = self.kappa * np.abs(self.moment[idx])
            central_variable = np.abs(self.central_variable_moment[idx])
            update = np.abs(g + self.epsilon)
            pi = (proxy - central_variable) / update
            np.clip(pi, 0., 5., out=pi) # For numerical stability.
            Pi.append(pi)

        return Pi

    def compute_update(self, weights, gradient):
        for idx, w in enumerate(weights):
            self.central_variable_moment[idx] = (w - self.stale[idx])
            self.stale[idx] = np.copy(w)

        update = []

        pi = self.gradient_energy_matching(gradient)
        # Apply the scalars to the update.
        for idx, g in enumerate(gradient):
            update.append(np.multiply(g, pi[idx]))

        return update

    def apply_update(self, weights, update):
        """Add the update to the weights."""
        new_weights = []
        for w, u in zip(weights, update):
            new_weights.append(np.add(w, u))
        return new_weights


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

    # indices = list(set(Kb).intersection(set(Ka)))
    if Ka != Kb:
        # rows, cols = np.unravel_index(indices, new_weights.shape)
        # correct_weights = correct_weights.tolil()
        # correct_weights[rows, cols] = new_weights[rows, cols]
        raise AssertionError()

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