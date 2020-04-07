### Optimizers used to update master process weights

import numpy as np
import copy
import logging


class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError


class MultiOptimizer(Optimizer):
    def __init__(self, opt, s):
        self.opts = [copy.deepcopy(opt) for i in range(s)]

    def reset(self):
        for o in self.opts:
            o.reset()

    def apply_update(self, weights, gradient):
        r = []
        for o, w, g in zip(self.opts, weights, gradient):
            r.append( o.apply_update(w, g) )
        return r


class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, learning_rate=0.5):
        super(VanillaSGD, self).__init__()
        self.learning_rate = 0.5

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""
        for k, dw in gradient['pdw'].items():
            weights['pdw'][k] = self.learning_rate * dw
            weights['w'][k] += weights['pdw'][k]

        for k, delta in gradient['pdd'].items():
            weights['pdd'][k] = self.learning_rate * delta
            weights['b'][k] += weights['pdd'][k]

        return weights


class RunningAverageOptimizer(Optimizer):
    """Base class for AdaDelta, Adam, and RMSProp optimizers.
        rho (tunable parameter): decay constant used to compute running parameter averages
        epsilon (tunable parameter): small constant used to prevent division by zero
        running_g2: running average of the squared gradient, where squaring is done componentwise"""

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(RunningAverageOptimizer, self).__init__()
        self.init_rho = rho
        self.init_epsilon = epsilon
        RunningAverageOptimizer.reset(self)

    def reset(self):
        super(RunningAverageOptimizer, self).reset()
        self.epsilon = self.init_epsilon
        self.rho = self.init_rho
        self.running_g2 = None

    def running_average_square_np(self, previous, update):
        """Computes and returns the running average of the square of a numpy array.
            previous (numpy array): value of the running average in the previous step
            update (numpy array): amount of the update"""
        try:
            new_contribution = (1-self.rho) * np.square(update)
            old_contribution = self.rho * previous
            return new_contribution + old_contribution
        except Exception as e:
            logging.error("FAILED TO COMPUTE THE RUNNING AVG SQUARE due to %s",str(e))
            logging.debug("rho %d",self.rho)
            logging.debug("min update %d",np.min(update))
            logging.debug("max update %d",np.max(update))
            logging.debug("min previous %d",np.min(previous))
            logging.debug("max previous %d",np.max(previous))
            return previous

    def running_average_square(self, previous, update):
        """Returns the running average of the square of a quantity.
            previous (list of numpy arrays): value of the running average in the previous step
            update (list of numpy arrays): amount of the update"""
        if previous == 0:
            previous = [ np.zeros_like(u) for u in update ]
        result = []
        for prev, up in zip(previous, update):
            result.append( self.running_average_square_np( prev, up ) )
        return result

    def sqrt_plus_epsilon(self, value):
        """Computes running RMS from the running average of squares.
            value: numpy array containing the running average of squares"""
        return np.sqrt( np.add(value, self.epsilon) )


class Adam(RunningAverageOptimizer):
    """Adam optimizer.
        Note that the beta_2 parameter is stored internally as 'rho'
        and "v" in the algorithm is called "running_g2"
        for consistency with the other running-average optimizers
        Attributes:
          learning_rate: base learning rate
          beta_1: decay rate for the first moment estimate
          m: running average of the first moment of the gradient
          t: time step
        """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
            epsilon=1e-8):
        super(Adam, self).__init__(rho=beta_2, epsilon=epsilon)
        self.init_learning_rate = learning_rate
        self.init_beta_1 = beta_1
        Adam.reset(self)

    def reset(self):
        super(Adam, self).reset()
        self.beta_1 = self.init_beta_1
        self.learning_rate = self.init_learning_rate
        self.t = 0
        self.m = None

    def running_average_np(self, previous, update):
        """Computes and returns the running average of a numpy array.
            Parameters are the same as those for running_average_square_np"""
        try:
            new_contribution = (1-self.beta_1) * update
            old_contribution = self.beta_1 * previous
            return new_contribution + old_contribution
        except Exception as e:
            logging.error("FAILED TO UPDATE THE RUNNING AVERAGE due to %s",str(e))
            logging.debug("beta_1 %d",self.beta_1)
            logging.debug("min update %d",np.min(update))
            logging.debug("max update %d",np.max(update))
            logging.debug("min previous %d",np.min(previous))
            logging.debug("max previous %d",np.max(previous))
            return previous

    def running_average(self, previous, update):
        """Returns the running average of the square of a quantity.
            Parameters are the same as those for running_average_square_np"""
        result = []
        for prev, up in zip(previous, update):
            result.append( self.running_average_np( prev, up ) )
        return result

    def apply_update(self, weights, gradient):
        """Update the running averages of the first and second moments
            of the gradient, and compute the update for this time step"""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.m is None:
            self.m = [ np.zeros_like(g) for g in gradient ]

        self.t += 1
        self.m = self.running_average( self.m, gradient )
        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        alpha_t = self.learning_rate * (1 - self.rho**self.t)**(0.5) / (1 - self.beta_1**self.t)
        new_weights = []
        for w, g, g2 in zip(weights, self.m, self.running_g2):
            try:
                update = alpha_t * g / ( np.sqrt(g2) + self.epsilon )
            except Exception as e:
                logging.error("FAILED TO MAKE A WEIGHT UPDATE due to %s",str(e))
                logging.debug("alpha_t %d",alpha_t)
                logging.debug("beta_1 %d",self.beta_1)
                logging.debug("t %d",self.t)
                logging.debug("learning rate %d",self.learning_rate)
                logging.debug("rho %d",self.rho)
                logging.debug("epsilon %d",self.epsilon)
                logging.debug("min gradient %d",np.min( g ))
                logging.debug("max gradient %d",np.max( g ))
                logging.debug("min gradient 2 %d",np.min( g2 ))
                logging.debug("max gradient 2 %d",np.max( g2 ))
                try:
                    update = alpha_t * g / ( np.sqrt(g2) + self.epsilon )
                    try:
                        new_weights.append( w - update )
                    except:
                        logging.debug("no sub")
                except:
                    try:
                        update = g / ( np.sqrt(g2) + self.epsilon )
                        logging.debug("min b %d",np.min( update ))
                        logging.debug("max b %d",np.max( update ))
                        logging.debug("min |b| %d",np.min(np.fabs( update)))
                        #update *= alpha_t
                    except:
                        try:
                            update = 1./ ( np.sqrt(g2) + self.epsilon )
                        except:
                            try:
                                update = 1./ ( g2 + self.epsilon )
                            except:
                                pass
                update = 0
            new_weights.append( w - update )
        return new_weights


class AdaDelta(RunningAverageOptimizer):
    """ADADELTA adaptive learning rate method.
        running_dx2: running average of squared parameter updates
        """

    def __init__(self, rho=0.95, epsilon=1e-8):
        super(AdaDelta, self).__init__(rho, epsilon)
        AdaDelta.reset(self)

    def reset(self):
        super(AdaDelta, self).reset()
        self.running_dx2 = None

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients and weight updates,
            and compute the Adadelta update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]
        if self.running_dx2 is None:
            self.running_dx2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        updates = []
        for w, g, g2, dx2 in zip(weights, gradient, self.running_g2, self.running_dx2):
            update = np.multiply( np.divide( self.sqrt_plus_epsilon(dx2), self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
            updates.append(update)
        self.running_dx2 = self.running_average_square( self.running_dx2, updates )
        return new_weights


class RMSProp(RunningAverageOptimizer):
    """RMSProp adaptive learning rate method.
        learning_rate: base learning rate, kept constant
        """

    def __init__(self, rho=0.9, epsilon=1e-8, learning_rate=0.001):
        super(RMSProp, self).__init__(rho, epsilon)
        self.init_learning_rate = learning_rate
        self.reset()

    def reset(self):
        super(RMSProp, self).reset()
        self.learning_rate = self.init_learning_rate

    def apply_update(self, weights, gradient):
        """Update the running averages of gradients,
            and compute the update for this step."""
        if self.running_g2 is None:
            self.running_g2 = [ np.zeros_like(g) for g in gradient ]

        self.running_g2 = self.running_average_square( self.running_g2, gradient )
        new_weights = []
        for w, g, g2 in zip(weights, gradient, self.running_g2):
            update = np.multiply( np.divide( self.learning_rate, self.sqrt_plus_epsilon(g2) ), g )
            new_weights.append( np.subtract( w, update ) )
        return new_weights


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
            'adadelta':      AdaDelta,
            'rmsprop':       RMSProp,
            'adam':          Adam,
            'gem':           GEM,
            }
    return lookup[name]


class OptimizerBuilder(object):
    """Builds a  optimizer"""

    def __init__(self, name, config=None):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        if self.name == 'sgd' and 'lr' not in self.config:
            logging.warning("Learning rate for SGD not set, using 1.0.")
            self.config['lr'] = 0.5

    def build(self):
        from keras.optimizers import deserialize
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        return opt