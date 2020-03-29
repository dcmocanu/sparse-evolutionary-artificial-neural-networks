### Optimizers used to update master process weights

import numpy as np
import copy
import pickle
import os
import re
import logging

from utils import weights_from_shapes

class Optimizer(object):
    """Base class for optimization algorithms.
        Currently doesn't do anything."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def apply_update(self, weights, gradient):
        raise NotImplementedError

    def save(self, fn = None):
        if fn is None:
            fn = 'master-opt-{}.algo'.format( os.getpid())
        d= open(fn,'wb')
        pickle.dump(self, d)
        d.close()
        logging.info("Saved state to %s", fn)

    def load(self, fn = 'algo_.pkl'):
        if not fn.endswith('.algo'):
            fn = fn + '.algo'
        try:
            d = open(fn, 'rb')
            new_self = pickle.load( d )
            d.close()
        except:
            new_self = None
        return new_self

class MultiOptimizer(Optimizer):
    def __init__(self, opt, s):
        self.opts = [copy.deepcopy(opt) for i in range(s)]

    def reset(self):
        for o in self.opts:
            o.reset()

    def apply_update(self, weights, gradient):
        r = []
        for o,w,g in zip(self.opts, weights, gradient):
            r.append( o.apply_update(w,g) )
        return r

class VanillaSGD(Optimizer):
    """Stochastic gradient descent with no extra frills.
          learning_rate: learning rate parameter for SGD"""

    def __init__(self, learning_rate=0.01):
        super(VanillaSGD, self).__init__()
        self.learning_rate = learning_rate

    def apply_update(self, weights, gradient):
        """Move weights in the direction of the gradient, by the amount of the
            learning rate."""
        new_weights = []
        for w, g in zip(weights, gradient):
            if type(w) == list:
                new_weights.append( [] )
                for ww, gg in zip(w,g):
                    new_weights[-1].append( np.subtract( ww, self.learning_rate*gg) )
            else:
                new_weights.append(np.subtract(w, self.learning_rate*g))
        return new_weights

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

class TFOptimizer(Optimizer):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.sess = None
        self.saver = None
        self.load_fn = None
        self.tf_optimizer = None

        self.reset()

    def reset(self):
        import tensorflow as tf

        if self.sess:
            self.sess.close() #free resources from previous execution

        self.sess = tf.Session()
        self.do_reset = True

    def save(self, fn='train_history'):
        fn = re.sub(r'\.algo$', '', fn)
        if not fn.startswith('./'):
            fn = './' + fn
        path = self.saver.save(self.sess, fn)
        logging.info("Saved state to %s", path)

    def setup_update(self, weights):
        import tensorflow as tf

        """Setup the tf computational graph. Should be run once for each model
            Receives the weights in order to know the shapes to use
        """
        self.gradient = [ tf.placeholder(dtype=tf.float32, shape=w.shape, name="gradient") for w in weights ]
        self.weights = [ tf.Variable(w, dtype=tf.float32, name="weights_{}".format(i)) for i, w in enumerate(weights) ]

        var_list = zip(self.gradient, self.weights)

        self.tf_time = tf.Variable(1, dtype=tf.float32, name="time")

        self.optimizer_op = self.tf_optimizer.apply_gradients(
            grads_and_vars=var_list,
            global_step=self.tf_time,
            name='optimizer_op' # We may need to create uniqie name
        )

        self.saver = tf.train.Saver(max_to_keep=None)

        if self.load_fn:
            self.saver.restore(self.sess, self.load_fn)
        else:
            self.sess.run(tf.global_variables_initializer())

    def load(self, fn='train_history'):
        load_fn = re.sub(r'\.algo$', '', fn)
        if os.path.isfile(load_fn + '.meta'):
            self.load_fn = load_fn
            self.do_reset = True
            return self
        else:
            return None

    def apply_update(self, weights, gradient):
        if self.do_reset:
            self.setup_update(weights)
            self.do_reset = False

        gradient_dict = {placeholder : value for placeholder, value in zip(self.gradient, gradient)}

        #Trace.begin("tf_optimizer")
        self.sess.run(self.optimizer_op, feed_dict=gradient_dict)
        #Trace.end("tf_optimizer")

        #Trace.begin("tf_get_weights")
        res = self.sess.run(self.weights)
        #Trace.end("tf_get_weights")

        return res

class GradientDescentTF(TFOptimizer):
    def __init__(self, learning_rate=0.01):
        super(GradientDescentTF, self).__init__(learning_rate=learning_rate)

    def setup_update(self, weights):
        import tensorflow as tf

        self.tf_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
            use_locking=False,
            name='SGDMaster' # We may need to create uniqie name
        )

        super(GradientDescentTF, self).setup_update(weights)

class AdaDeltaTF(TFOptimizer):
    def __init__(self, learning_rate=0.001, rho=0.95, epsilon=1e-8):
        super(AdaDeltaTF, self).__init__(learning_rate=learning_rate, rho=rho, epsilon=epsilon)

    def setup_update(self, weights):
        import tensorflow as tf

        self.tf_optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            use_locking=False,
            name='AdaDeltaMaster' # We may need to create uniqie name
        )

        super(AdaDeltaTF, self).setup_update(weights)

class RMSPropTF(TFOptimizer):
    def __init__(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10):
        super(RMSPropTF, self).__init__(learning_rate=learning_rate, decay=decay,
            momentum=momentum, epsilon=epsilon)

    def setup_update(self, weights):
        import tensorflow as tf

        self.tf_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self.decay,
            momentum=self.momentum,
            epsilon=self.epsilon,
            use_locking=False,
            name='RMSPropMaster' # We may need to create uniqie name
        )

        super(RMSPropTF, self).setup_update(weights)

class AdamTF(TFOptimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(AdamTF, self).__init__(learning_rate=learning_rate,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    def setup_update(self, weights):
        import tensorflow as tf

        self.tf_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta_1,
            beta2=self.beta_2,
            epsilon=self.epsilon,
            use_locking=False,
            name='AdamMaster' # We may need to create uniqie name
        )

        super(AdamTF, self).setup_update(weights)

class TorchOptimizer(Optimizer):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.parameters = None
        self.torch_optimizer = None
        self.state = None

        self.reset()

    def reset(self):
        self.do_reset = True

    def apply_update(self, weights, gradient):
        import torch
        if self.do_reset:
            self.setup_update(weights)
            if self.state is not None:
                self.torch_optimizer.load_state_dict(self.state)
            self.do_reset = False

        for p, w, g in zip (self.parameters, weights, gradient):
            p.data.copy_(torch.from_numpy(w))
            p.grad.data.copy_(torch.from_numpy(g))

        self.torch_optimizer.step()
        return [i.data.cpu().numpy() for i in list(self.parameters)]

    def setup_update(self, weights):
        import torch
        if self.parameters is not None:
            for p in self.parameters:
                del p
        else:
            self.parameters = [None] * len(weights)
        for i, w in enumerate(weights):
            p = torch.from_numpy(w).cuda()
            g = torch.from_numpy(w).cuda()
            var = torch.autograd.Variable(p, requires_grad=True)
            var.grad = torch.autograd.Variable(g)
            self.parameters[i] = var

    def save(self, fn=None):
        if fn is None:
            fn = 'master-opt-{}.algo'.format( os.getpid())

        state = self.torch_optimizer.state_dict()
        with open(fn, 'wb') as out_file:
            pickle.dump(state, out_file)
        logging.info("Saved state to %s", fn)

    def load(self, fn = 'algo_.pkl'):
        if not fn.endswith('.algo'):
            fn = fn + '.algo'
        with open(fn, 'rb') as in_file:
            self.state = pickle.load(in_file)
        return self

class SGDTorch(TorchOptimizer):
    def __init__(self, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGDTorch, self).__init__(lr=lr, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    def setup_update(self, weights):
        super(SGDTorch, self).setup_update(weights)
        import torch.optim as topt
        self.torch_optimizer = topt.SGD(self.parameters, self.lr, self.momentum, self.dampening,
            self.weight_decay, self.nesterov)

class AdaDeltaTorch(TorchOptimizer):
    def __init__(self, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
        super(AdaDeltaTorch, self).__init__(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

    def setup_update(self, weights):
        super(AdaDeltaTorch, self).setup_update(weights)
        import torch.optim as topt
        self.torch_optimizer = topt.Adadelta(self.parameters, self.lr, self.rho, self.eps, self.weight_decay)

class RMSPropTorch(TorchOptimizer):
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        super(RMSPropTorch, self).__init__(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
            momentum=momentum, centered=centered)

    def setup_update(self, weights):
        super(RMSPropTorch, self).setup_update(weights)
        import torch.optim as topt
        self.torch_optimizer = topt.RMSprop(self.parameters, self.lr, self.alpha, self.eps,
            self.weight_decay, self.momentum, self.centered)

class AdamTorch(TorchOptimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        super(AdamTorch, self).__init__(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def setup_update(self, weights):
        super(AdamTorch, self).setup_update(weights)
        import torch.optim as topt
        self.torch_optimizer = topt.Adam(self.parameters, self.lr, self.betas, self.eps,
            self.weight_decay)

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
            # Wrappers around TF's optimizers
            'sgdtf':         GradientDescentTF,
            'adadeltatf':    AdaDeltaTF,
            'rmsproptf':     RMSPropTF,
            'adamtf':        AdamTF,
            # Wrappers arount Torch's optimizers
            'sgdtorch':      SGDTorch,
            'adadeltatorch': AdaDeltaTorch,
            'rmsproptorch':  RMSPropTorch,
            'adamtorch':     AdamTorch,
            }
    return lookup[name]

class OptimizerBuilder(object):
    """Builds a new Keras or Torch optimizer and optionally wraps it in horovod DistributedOptimizer."""

    def __init__(self, name, config=None, horovod_wrapper=False):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        if self.name == 'sgd' and 'lr' not in self.config:
            logging.warning("Learning rate for SGD not set, using 1.0.")
            self.config['lr'] = 1.
        self.horovod_wrapper = horovod_wrapper

    def build(self):
        from keras.optimizers import deserialize
        opt_config = {'class_name': self.name, 'config': self.config}
        opt = deserialize(opt_config)
        if self.horovod_wrapper:
            import horovod.keras as hvd
            if hasattr(opt, 'lr'):
                opt.lr *= hvd.size()
            opt = hvd.DistributedOptimizer(opt)
        return opt

    def build_torch(self, model):
        import torch
        lookup = {
            'sgd':      torch.optim.SGD,
            'adadelta': torch.optim.Adadelta,
            'rmsprop':  torch.optim.RMSprop,
            'adam':     torch.optim.Adam
            }
        if self.name not in lookup:
            logging.warning("No optimizer '{}' found, using SGD instead".format(self.name))
            self.name = 'sgd'
        opt = lookup[self.name](model.parameters(), **self.config)
        if self.horovod_wrapper:
            import horovod.torch as hvd
            opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())
        return opt