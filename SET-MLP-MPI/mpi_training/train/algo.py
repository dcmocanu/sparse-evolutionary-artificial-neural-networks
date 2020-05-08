### Algo class

from .optimizer import get_optimizer, retain_valid_updates, retain_valid_weights


class Algo(object):
    """The Algo class contains all information about the training algorithm.
        Attributes:
          optimizer: instance of the Optimizer class used to compute training updates
          optimizer_name: name of the optimizer
          staleness: difference in time step between master and most recent worker's update
          worker_update_type: whether to send weights or gradients to parent process
          send_before_apply: whether to send weights before applying update
          step_counter: counts time steps to determine when to sync
          (used for Elastic Averaging SGD)
        See __init__ for list of other supported attributes
          """

    # available options and their default values
    supported_opts = {
                      'validate_every': 1000,
                      'sync_every': 1,
                      'mode': 'sgdm',
                      'optimizer_params': '{}',
                      'elastic_force': None,
                      'elastic_momentum': 0
    }

    def __init__(self, optimizer, **kwargs):
        """optimizer: string naming an optimization algorithm as defined in Optimizer.get_optimizer()
            Configuration options should be provided as keyword arguments.
            Available arguments are:
               loss: string naming the loss function to be used for training
               validate_every: number of time steps to wait between validations
               sync_every: number of time steps to wait before getting weights from parent
               mode: 'sgd', 'easgd' or 'gem' are supported
               elastic_force: alpha constant in the Elastic Averaging SGD algorithm
               elastic_lr: EASGD learning rate for worker
               elastic_momentum: EASGD momentum for worker
            Optimizer configuration options should be provided as additional
            named arguments (check your chosen optimizer class for details)."""
        for opt in self.supported_opts:
            if opt in kwargs:
                setattr(self, opt, kwargs[opt])
            else:
                setattr(self, opt, self.supported_opts[opt])

        self.optimizer_name = optimizer
        if optimizer is not None:
            optimizer_args = {arg: val for arg, val in kwargs.items()
                              if arg not in self.supported_opts}

            self.optimizer = get_optimizer(optimizer)(**optimizer_args)
        else:
            self.optimizer = None
            if self.mode == 'easgd':
                self.optimizer = get_optimizer('sgdm')(lr=0.05, momentum=0.9, weight_decay=0.0002)

        """ Workers are only responsible for computing the gradient and 
            sending it to the master, so we use ordinary SGD with learning rate 1 and 
            compute the gradient as (old weights - new weights) after each batch."""

        self.step_counter = 0
        if self.mode == 'gem':
            self.worker_update_type = 'gem'
        elif self.mode == 'easgd':
            self.worker_update_type = 'weights'
            self.send_before_apply = True
        else:
            self.worker_update_type = 'update'
            self.send_before_apply = False

        # Keep track if internal state was restored
        self.restore = False

    def get_config(self):
        config = {}
        config['optimizer'] = str(self.optimizer_name)
        for opt in self.supported_opts:
            config[opt] = str(getattr(self, opt))
        return config

    def __str__(self):
        strs = ["optimizer: " + str(self.optimizer_name)]
        strs += [opt + ": " + str(getattr(self, opt)) for opt in self.supported_opts]
        return '\n'.join(strs)

    ### For Worker ###
    def compute_update(self, cur_weights, new_weights):
        """Computes the update to be sent to the parent process"""
        if self.worker_update_type == 'gem':
            return self.optimizer.begin_compute_update(cur_weights, new_weights)
        elif self.worker_update_type == 'weights':
            return new_weights
        else:
            update = {'pdw': {}, 'pdd': {}}
            update['pdw'] = new_weights['pdw']
            update['pdd'] = new_weights['pdd']
            return update

    ### For Worker ###
    def compute_update_gem(self, update):
        return self.optimizer.begin_compute_update(update)

    def compute_update_worker(self, weights, update):
        """Compute the update on worker (for GEM)"""
        if self.mode == 'gem':  # Only possible in GEM mode
            return self.optimizer.compute_update(weights, update)

    def set_worker_model_weights(self, model, weights, gradients=None):
        """Apply a new set of weights to the worker's copy of the model"""
        if self.mode == 'easgd':
            new_weights = self.get_elastic_update(model.get_weights(), weights)
            self.optimizer.apply_update(new_weights, gradients)
            model.set_weights(new_weights)
        else:
            model.set_weights(weights)

    def get_elastic_update(self, cur_weights, other_weights):
        """EASGD weights update"""

        # for index, weights in other_weights['w'].items():
        #     if index in cur_weights['pdw']:
        #         cur_weights['pdw'][index] = retain_valid_updates(other_weights['w'][index], cur_weights['pdw'][index])

        for index, v in other_weights['w'].items():
            #cur_weights['w'][index] = retain_valid_weights(other_weights['w'][index], cur_weights['w'][index])
            cur_weights['w'][index] = cur_weights['w'][index] - self.elastic_force * (cur_weights['w'][index] - other_weights['w'][index])
        for index, v in other_weights['b'].items():
            cur_weights['b'][index] = cur_weights['b'][index] - self.elastic_force * (cur_weights['b'][index] - other_weights['b'][index])

        return cur_weights

    def should_sync(self):
        """Determine whether to pull weights from the master"""
        self.step_counter += 1
        return self.step_counter % self.sync_every == 0

    ### For Master ###

    def apply_update(self, weights, update):
        """Calls the optimizer to apply an update
            and returns the resulting weights"""
        if self.mode == 'easgd':
            return self.get_elastic_update_master(weights, update)
        else:
            return self.optimizer.apply_update(weights, update)

    def get_elastic_update_master(self, cur_weights, other_weights):
        """EASGD weights update"""

        for index, v in other_weights['w'].items():
            #other_weights['w'][index] = retain_valid_weights(cur_weights['w'][index], other_weights['w'][index])
            cur_weights['w'][index] = cur_weights['w'][index] + self.elastic_force * (other_weights['w'][index] - cur_weights['w'][index])
        for index, v in other_weights['b'].items():
            cur_weights['b'][index] = cur_weights['b'][index] + self.elastic_force * (other_weights['b'][index] - cur_weights['b'][index])

        return cur_weights