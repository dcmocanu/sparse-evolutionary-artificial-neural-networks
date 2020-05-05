### MPIWorker and MPIMaster classes

import json
import numpy as np
import time
import logging
import datetime

from mpi4py import MPI
from mpi_training.train.monitor import Monitor
from utils.load_data import Error
from mpi_training.logger import get_logger, set_logging_prefix


### Classes ###

class MPIProcess(object):
    """Base class for processes that communicate with one another via MPI.
       Attributes:
           parent_comm: MPI intracommunicator used to communicate with this process's parent
           parent_rank (integer): rank of this node's parent in parent_comm
           rank (integer): rank of this node in parent_comm
           model: SET model to train
           algo: Algo object defining how to optimize model weights
           weights: list of numpy arrays storing the last weights received from the parent
           update: latest update obtained from training
           data: Data object used to generate training or validation data
           time_step: for keeping track of time
           num_epochs: integer giving the number of epochs to train for
           stop_training: becomes true when it is time to stop training
           idle_time: kep track of process' idle time
    """

    def __init__(self, parent_comm, process_comm, parent_rank=None, num_epochs=1, data=None, algo=None,
                 model=None, monitor=False, save_filename=None,):
        """If the rank of the parent is given, initialize this process and immediately start
            training. If no parent is indicated, training should be launched with train().

            Parameters:
              parent_comm: MPI intracommunicator used to communicate with parent
              parent_rank (integer): rank of this node's parent in parent_comm
              num_epochs: number of training epochs
              data: Data object used to generate training or validation data
              algo: Algo object used to configure the training process
              model: SET model to train
              monitor: whether to monitor CPU/GPU usage
              save_filename: file name to log metrics and weights
        """
        self.parent_comm = parent_comm
        self.process_comm = process_comm
        self.parent_rank = parent_rank
        self.num_epochs = num_epochs
        self.data = data
        self.algo = algo
        self.model = model
        self.save_filename = save_filename
        self.idle_time = 0.0

        self.gradients = None
        self.update = None
        self.stop_training = False
        self.time_step = 0
        self._is_shadow = (self.process_comm is not None and self.process_comm.Get_rank() != 0)

        self.monitor = Monitor(save_filename=save_filename) if monitor else None

        process_type = self.__class__.__name__.replace('MPI', '')[0]
        set_logging_prefix(
            MPI.COMM_WORLD.Get_rank(),
            self.parent_comm.Get_rank() if self.parent_comm is not None else '-',
            self.process_comm.Get_rank() if self.process_comm is not None else '-',
            process_type
        )
        self.logger = get_logger()

        if self.process_comm is not None and self.process_comm.Get_size() > 1:
            self.process_comm.Barrier()
            self.process_comm.Barrier()

        self.rank = parent_comm.Get_rank() if parent_comm else 0
        self.ranks = "{0}:{1}:{2}".format(
            MPI.COMM_WORLD.Get_rank(),
            self.parent_comm.Get_rank() if self.parent_comm is not None else '-',
            self.process_comm.Get_rank() if self.process_comm is not None else '-')

        self.epoch = 0

        self.build_model()
        if (self.parent_rank is not None and self.parent_comm is not None):
            self.bcast_weights(self.parent_comm)
        if (self.parent_rank is not None and self.parent_comm is not None) or (self.process_comm):
            self.train()

    def is_shadow(self, sync=False):
        """signals that the process is a sub-process and should not act normally"""
        if self.process_comm and sync:
            import inspect
            self.logger.debug("syncing on the process communicator from", inspect.stack()[1][3])
            self.process_comm.Barrier()
        return self._is_shadow

    def build_model(self):
        self.weights = self.model.get_weights()
        self.update = self.model.format_update()

    def check_sanity(self):
        """Throws an exception if any model attribute has not been set yet."""
        for par in ['model',
                    'weights']:
            if not hasattr(self, par) or getattr(self, par) is None:
                raise Error("%s not found!  Process %s does not seem to be set up correctly." % (par, self.ranks))

    def train(self, testing=False):
        """To be implemented in derived classes"""
        raise NotImplementedError

    def print_metrics(self, metrics):
        """Display metrics computed during training or validation"""
        self.model.print_metrics(metrics)

    def get_logs(self, metrics, val=False):
        """Get dictionary of logs computed during training.
            If val is True, appends 'val' to the beginning of each metric name"""
        MPI.COMM_WORLD.Abort()
        if val:
            return {'val_' + name: np.asscalar(metric) for name, metric in
                    zip(self.model.metrics_names(), metrics)}
        else:
            return {name: np.asscalar(metric) for name, metric in
                    zip(self.model.metrics_names(), metrics)}

    def do_send_sequence(self):
        """Actions to take when sending an update to parent:
            -Send the update (if the parent accepts it)
            -Sync time and model weights with parent"""
        if self.is_shadow():
            return

        self.send_update(check_permission=True)
        t1 = datetime.datetime.now()
        self.time_step = self.recv_time_step()
        self.weights = self.recv_weights()
        t2 = datetime.datetime.now()
        self.idle_time += (t2 - t1).total_seconds()

        self.algo.set_worker_model_weights(self.model, self.weights, self.gradients)

    def apply_update(self):
        """Updates weights according to update received from worker process"""
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            self.weights = self.algo.apply_update(self.weights, self.update)
            self.model.set_weights(self.weights)

    ### MPI-related functions below ###

    # This dict associates message strings with integers to be passed as MPI tags.
    tag_lookup = {
        'any': MPI.ANY_TAG,
        'train': 0,
        'exit': 1,
        'begin_weights': 2,
        'begin_update': 3,
        'time': 4,
        'bool': 5,
        'weights': 12,
        'update': 12,
        'begin_gem': 13,
        'update_gem': 14,
    }
    # This dict is for reverse tag lookups.
    inv_tag_lookup = {value: key for key, value in tag_lookup.items()}

    def lookup_mpi_tag(self, name, inv=False):
        """Searches for the indicated name in the tag lookup table and returns it if found.
            Params:
              name: item to be looked up
              inv: boolean that is True if an inverse lookup should be performed (int --> string)"""
        if inv:
            lookup = self.inv_tag_lookup
        else:
            lookup = self.tag_lookup
        try:
            return lookup[name]
        except KeyError:
            self.logger.error("Not found in tag dictionary: {0} -- returning None".format(name))
            return None

    def recv(self, obj=None, tag=MPI.ANY_TAG, source=None, status=None, comm=None):
        """ Wrapper around MPI.recv/Recv. Returns the received object.
            Params:
              obj: variable into which the received object should be placed
              tag: string indicating which MPI tag should be received
              source: integer rank of the message source.  Defaults to self.parent_rank
              status: MPI status object that is filled with received status information
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if source is None:
            if self.parent_rank is None:
                raise Error("Attempting to receive %s from parent, but parent rank is None" % tag)
            source = self.parent_rank
        tag_num = self.lookup_mpi_tag(tag)
        # if tag in ['bool','time']:
        #    comm.Recv(obj, source=source, tag=tag_num, status=status )
        #    return obj

        obj = comm.recv(source=source, tag=tag_num, status=status)
        return obj

    def send(self, obj, tag, dest=None, comm=None):
        """ Wrapper around MPI.send/Send.  Params:
             obj: object to send
             tag: string indicating which MPI tag to send
             dest: integer rank of the message destination.  Defaults to self.parent_rank
             comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        if dest is None:
            if self.parent_rank is None:
                raise Error("Attempting to send %s to parent, but parent rank is None" % tag)
            dest = self.parent_rank
        tag_num = self.lookup_mpi_tag(tag)
        # if tag in ['time']:
        #    comm.Send( obj, dest=dest, tag=tag_num )
        #    return
        comm.send(obj, dest=dest, tag=tag_num)

    def bcast(self, obj, root=0, comm=None):
        """Wrapper around MPI.bcast/Bcast.  Returns the broadcasted object.
            Params:
              obj: object to broadcast
              root: rank of the node to broadcast from
              comm: MPI communicator to use.  Defaults to self.parent_comm"""
        if comm is None:
            comm = self.parent_comm
        obj = comm.bcast(obj, root=root)
        return obj

    def send_exit_to_parent(self):
        if self.is_shadow(sync=True): return
        """Send exit tag to parent process, if parent process exists"""
        if self.parent_rank is not None:
            self.send(None, 'exit')

    def send_arrays(self, obj, expect_tag, tag, comm=None, dest=None, check_permission=False):
        self.send(None, expect_tag, comm=comm, dest=dest)
        if check_permission:
            # To check permission we send the update's time stamp to the master.
            # Then we wait to receive the decision yes/no.
            self.send_time_step(comm=comm, dest=dest)
            decision = self.recv_bool(comm=comm, source=dest)
            if not decision:
                return
        self.send(obj, tag, comm=comm, dest=dest)

    def send_weights(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow(): return
        """Send NN weights to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the weights we first send the tag 'begin_weights'."""
        self.send_arrays(self.weights, expect_tag='begin_weights', tag='weights',
                         comm=comm, dest=dest, check_permission=check_permission)

    def send_update(self, comm=None, dest=None, check_permission=False):
        if self.is_shadow(): return
        """Send update to the process specified by comm (MPI communicator) and dest (rank).
            Before sending the update we first send the tag 'begin_update'"""
        self.send_arrays(self.update, expect_tag='begin_update', tag='update',
                         comm=comm, dest=dest, check_permission=check_permission)

    def send_time_step(self, comm=None, dest=None):
        if self.is_shadow(): return
        """Send the current time step"""
        self.send(obj=self.time_step, tag='time', dest=dest, comm=comm)

    def send_bool(self, obj, comm=None, dest=None):
        if self.is_shadow(): return
        self.send(obj=obj, tag='bool', dest=dest, comm=comm)

    def recv_arrays(self, obj, tag, comm=None, source=None):
        return self.recv(obj, tag, comm=comm, source=source)

    def recv_weights(self, comm=None, source=None, add_to_existing=False):
        """Receive NN weights layer by layer from the process specified by comm and source"""
        if self.is_shadow(): return
        return self.recv_arrays(self.weights, tag='weights', comm=comm, source=source)

    def recv_update(self, comm=None, source=None, add_to_existing=False):
        """Receive an update layer by layer from the process specified by comm and source.
            Add it to the current update if add_to_existing is True,
            otherwise overwrite the current update"""
        if self.is_shadow(): return
        if add_to_existing:
            tmp = {}
            tmp = self.recv_arrays(tmp, tag='update', comm=comm, source=source)

            for index, v in tmp.items():
                dw = v[0]
                delta = v[1]

                # perform the update with momentum
                if index not in self.update:
                    self.update[index] = (dw, delta)
                else:
                    self.update[index] = (self.update[index][0] + dw, self.update[index][1] + delta)

        else:
            self.update = self.recv_arrays(self.update, tag='update', comm=comm, source=source)

    def recv_time_step(self, comm=None, source=None):
        """Receive the current time step"""
        if self.is_shadow(): return
        return self.recv(tag='time', comm=comm, source=source)

    def recv_bool(self, comm=None, source=None):
        if self.is_shadow(): return
        return self.recv(tag='bool', comm=comm, source=source)

    def recv_exit_from_parent(self):
        ir = None
        if not self.is_shadow():
            ir = self.parent_comm.irecv(source=0, tag=self.lookup_mpi_tag('exit'))
        elif self.process_comm:
            ir = self.process_comm.irecv(source=0, tag=self.lookup_mpi_tag('exit'))
        return ir

    def bcast_weights(self, comm, root=0):
        """Broadcast weights shape and weights (layer by layer)
            on communicator comm from the indicated root rank"""
        self.bcast(self.weights, comm=comm, root=root)


class MPIWorker(MPIProcess):
    """This class trains its NN model and exchanges weight updates with its parent."""

    def __init__(self, data, algo, model, process_comm,parent_comm, parent_rank=None,
                 num_epochs=1, monitor=False, save_filename=None):
        """Raises an exception if no parent rank is provided. Sets the number of epochs
            using the argument provided, then calls the parent constructor"""
        info = "Creating MPIWorker with rank {0} and parent rank {1} on a communicator of size {2}"
        tell_comm = parent_comm if parent_comm is not None else process_comm
        if tell_comm: logging.info(info.format(tell_comm.Get_rank(),
                                               parent_rank,
                                               tell_comm.Get_size()))

        super(MPIWorker, self).__init__(parent_comm, process_comm, parent_rank,
                                        num_epochs=num_epochs, data=data, algo=algo, model=model,
                                        monitor=monitor, save_filename=save_filename)

    def build_model(self):
        super(MPIWorker, self).build_model()

    def sync_with_parent(self):

        if self.algo.mode == 'easgd':
            self.compute_update()

        if self.algo.mode == 'gem':
            self.compute_update()
            self.do_gem_sequence()
        else:
            self.do_send_sequence()

    def notify_parent(self):
        """Notify parent that worker is ready to begin GEM sequence."""
        self.send(None, 'begin_gem')

    def do_gem_sequence(self):
        """GEM sequence on worker:
         -Notify parent we are ready
         -Receive central variable (weights) from master
         -Compute the update with GEM
         -Send the update to master and apply it to own weights
        """
        self.notify_parent()
        self.recv_weights()
        self.update = self.algo.compute_update_worker(self.weights, self.update)
        self.send_update()
        self.apply_update()
        self.algo.set_worker_model_weights(self.model, self.weights)

    def train(self, testing=False):
        """  Wait for the signal to train. Then train for num_epochs epochs.
            In each step, train on one batch of input data, then send the update to the master
            and wait to receive a new set of weights.  When done, send 'exit' signal to parent.
        """
        self.check_sanity()
        self.await_signal_from_parent()

        maximum_accuracy = 0
        metrics = np.zeros((self.num_epochs, 6))

        self.model.set_weights(self.weights)

        for epoch in range(1, self.num_epochs + 1):
            self.logger.info("Beginning epoch {:d}".format(self.epoch + epoch))

            if self.monitor:
                self.monitor.start_monitor()

            self.data.shuffle()

            for j in range(self.data.x_train.shape[0] // self.data.batch_size):
                k = j * self.data.batch_size
                l = (j + 1) * self.data.batch_size

                if self.process_comm:
                    # broadcast the weights to all processes
                    self.bcast_weights(comm=self.process_comm)
                    if self.process_comm.Get_rank() != 0:
                        self.model.set_weights(self.weights)

                self.update = self.model.train_on_batch(x=self.data.x_train[k:l], y=self.data.y_train[k:l])

                if self.algo.should_sync():
                    self.sync_with_parent()

            if self.monitor:
                self.monitor.stop_monitor()

            if testing:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.model.predict(self.data.x_test, self.data.y_test)
                accuracy_train, activations_train = self.model.predict(self.data.x_train, self.data.y_train)
                t4 = datetime.datetime.now()
                maximum_accuracy = max(maximum_accuracy, accuracy_test)
                loss_test = self.model.compute_loss(self.data.y_test, activations_test)
                loss_train = self.model.compute_loss(self.data.y_train, activations_train)
                metrics[epoch - 1, 0] = loss_train
                metrics[epoch - 1, 1] = loss_test
                metrics[epoch - 1, 2] = accuracy_train
                metrics[epoch - 1, 3] = accuracy_test

                self.logger.debug("Validation metrics:")
                self.logger.debug(f"Testing time: {t4 - t3}\n; Loss test: {loss_test}; \n"
                                  f"Accuracy test: {accuracy_test}; \n"
                                  f"Maximum accuracy val: {maximum_accuracy}")
                # save performance metrics values in a file
                if (self.save_filename != ""):
                    np.savetxt(self.save_filename + ".txt", metrics)

            if self.stop_training:
                break

        self.logger.debug("Signing off")
        self.logger.info(f"Worker idle time: {self.idle_time}")

        self.send_exit_to_parent()

    def compute_update(self):
        """Compute the update from the new and old sets of model weights"""
        self.gradients = self.update
        self.update = self.algo.compute_update(self.weights, self.model.get_weights())

    def await_signal_from_parent(self):
        """Wait for 'train' signal from parent process"""
        if not self.is_shadow():
            tag = self.recv(tag='train')
        if self.process_comm:
            if self.process_comm.Get_rank() == 0:
                for r in range(1, self.process_comm.Get_size()):
                    self.send(None, tag='train', comm=self.process_comm, dest=r)
            else:
                self.recv(tag='train', comm=self.process_comm)


class MPIMaster(MPIProcess):
    """This class sends model information to its worker processes and updates its model weights
        according to updates or weights received from the workers.

        Attributes:
          child_comm: MPI intracommunicator used to communicate with child processes
          has_parent: boolean indicating if this process has a parent process
          num_workers: integer giving the number of workers that work for this master
          best_val_loss: best validation loss computed so far during training
          running_workers: list of workers not yet done training
          waiting_workers_list: list of workers that sent updates and are now waiting
          num_sync_workers: number of worker updates to receive before performing an update
          update_tag: MPI tag to expect when workers send updates
          epoch: current epoch number
    """

    def __init__(self, parent_comm, parent_rank=None, child_comm=None,
                 num_epochs=1, data=None, algo=None, model=None,
                 num_sync_workers=1,monitor=False, save_filename=None,):
        """Parameters:
              child_comm: MPI communicator used to contact children"""
        if child_comm is None:
            raise Error("MPIMaster initialized without child communicator")
        self.child_comm = child_comm
        self.has_parent = False
        if parent_rank is not None:
            self.has_parent = True
        self.best_val_loss = 0.
        self.best_val_acc = 0.
        self.weights_to_save = []
        self.biases_to_save = []

        self.num_workers = child_comm.Get_size() - 1  # all processes but one are workers
        self.metrics = np.zeros((num_epochs + 1, 4))

        self.num_sync_workers = num_sync_workers
        info = ("Creating MPIMaster with rank {0} and parent rank {1}. "
                "(Communicator size {2}, Child communicator size {3})")
        logging.info(info.format(parent_comm.Get_rank(), parent_rank, parent_comm.Get_size(),
                                 child_comm.Get_size()))
        if self.num_sync_workers > 1:
            logging.info("Will wait for updates from {0:d} workers before synchronizing".format(self.num_sync_workers))

        super(MPIMaster, self).__init__(parent_comm, process_comm=None, parent_rank=parent_rank, data=data,
                                        algo=algo, model=model, num_epochs=num_epochs, save_filename=save_filename,
                                        monitor=monitor)

    def decide_whether_to_sync(self):
        """Check whether enough workers have sent updates"""
        return (len(self.waiting_workers_list) >= self.num_sync_workers)

    def is_synchronous(self):
        return self.num_sync_workers > 1

    def accept_update(self):
        """Returns true if the master should accept the latest worker's update, false otherwise"""
        return (not self.is_synchronous()) or self.algo.staleness == 0

    def sync_children(self):
        """Update model weights and signal all waiting workers to work again.
            Send our update to our parent, if we have one"""
        while self.waiting_workers_list:
            child = self.waiting_workers_list.pop()
            self.sync_child(child)

    def sync_child(self, child):
        self.send_time_step(dest=child, comm=self.child_comm)
        self.send_weights(dest=child, comm=self.child_comm)

    def sync_parent(self):
        if self.has_parent:
            self.do_send_sequence()
        else:
            self.time_step += 1

    def do_update_sequence(self, source):
        """Update procedure:
         -Compute the staleness of the update and decide whether to accept it.
         -If we accept, we signal the worker and wait to receive the update.
         -After receiving the update, we determine whether to sync with the workers.
         -Finally we run validation if we have completed one epoch's worth of updates."""
        child_time = self.recv_time_step(source=source, comm=self.child_comm)
        self.algo.staleness = self.time_step - child_time
        accepted = self.accept_update()
        self.send_bool(accepted, dest=source, comm=self.child_comm)
        if accepted:
            t1 = datetime.datetime.now()
            self.recv_update(source=source, comm=self.child_comm,
                             add_to_existing=self.is_synchronous())
            self.waiting_workers_list.append(source)
            t2 = datetime.datetime.now()
            self.idle_time += (t2 - t1).total_seconds()

            if self.decide_whether_to_sync():
                if self.algo.send_before_apply:

                    self.sync_parent()
                    self.sync_children()
                    self.apply_update()

                    if self.algo.validate_every > 0 and self.time_step > 0:
                        if self.time_step % self.algo.validate_every == 0:
                            self.weights_to_save.append(self.weights['w'])
                            self.biases_to_save.append(self.weights['b'])

                            self.validate(self.weights)
                            if self.epoch < self.num_epochs // self.num_workers - 1:
                                t5 = datetime.datetime.now()
                                self.logger.info(self.weights['w'][1].count_nonzero())
                                self.logger.info(self.weights['w'][2].count_nonzero())
                                self.logger.info(self.weights['w'][3].count_nonzero())
                                self.logger.info(self.weights['w'][4].count_nonzero())
                                self.model.model.weightsEvolution_III()
                                t6 = datetime.datetime.now()
                                self.logger.info(f"Weights evolution time  {t6 - t5}")

                            self.weights = self.model.get_weights()
                            self.logger.info(f"Master epoch {self.epoch + 1}")

                else:

                    self.apply_update()
                    if self.is_synchronous():
                        self.update = {}

                    if self.algo.validate_every > 0 and self.time_step > 0:
                        if self.time_step % self.algo.validate_every == 0:
                            self.weights_to_save.append(self.weights['w'])
                            self.biases_to_save.append(self.weights['b'])

                            self.validate(self.weights)
                            if self.epoch < self.num_epochs//(self.num_workers) - 1:
                                t5 = datetime.datetime.now()

                                self.model.weight_evolution()
                                t6 = datetime.datetime.now()
                                self.logger.info(f"Weights evolution time  {t6 - t5}")
                                self.weights = self.model.get_weights()

                            self.logger.info(f"Master epoch {self.epoch+1}")

                    self.sync_parent()
                    self.sync_children()

            if self.algo.validate_every > 0 and self.time_step > 0:
                if self.time_step % self.algo.validate_every == 0:
                    self.epoch += 1
        else:
            self.sync_child(source)

    def do_gem_sequence(self, source):
        """Gradient enery matching procedure:
         -Send the current central variable (weights) to worker.
         -Wait to receive the update and apply it.
         -Finally we run validation if we have completed one epoch's worth of updates."""
        self.send_weights(dest=source, comm=self.child_comm)

    def do_gem_update_sequence(self, source):
        self.recv_update(source=source, comm=self.child_comm)
        self.apply_update()
        # Update model weights and signal all waiting workers to work again.
        self.time_step += 1
        if self.algo.validate_every > 0 and self.time_step > 0:
            if self.time_step % self.algo.validate_every == 0:
                self.validate(self.weights)
                self.epoch += 1

    def do_worker_finish_sequence(self, worker_id):
        """Actions to take when a worker finishes training"""
        self.running_workers.remove(worker_id)
        self.num_sync_workers -= 1

    def process_message(self, status):
        """Extracts message source and tag from the MPI status object and processes the message.
            Returns the tag of the message received.
            Possible messages are:
            -begin_update: worker is ready to send a new update
            -begin_gem: Worker needs central variable to start GEM
            -exit: worker is done training and will shut down
        """
        source = status.Get_source()
        tag = self.lookup_mpi_tag(status.Get_tag(), inv=True)

        if tag == 'begin_update':
            if self.algo.mode == 'gem':
                self.do_gem_update_sequence(source)
            else:
                self.do_update_sequence(source)
        elif tag == 'begin_gem':
            self.do_gem_sequence(source)
        elif tag == 'exit':
            self.do_worker_finish_sequence(source)
        else:
            raise ValueError("Tag %s not recognized" % tag)
        return tag

    def shut_down_workers(self):
        """Signal all running workers to shut down"""
        for worker_id in self.running_workers:
            self.logger.info("Signaling worker {0:d} to shut down".format(worker_id))
            self.send_exit_to_child(worker_id)

    def train(self, testing=False):
        """ Broadcasts model information to children and signals them to start training.
            Receive messages from workers and processes each message until training is done.
            When finished, signal the parent process that training is complete.
        """
        self.start_time = time.time()

        self.check_sanity()
        self.bcast_weights(comm=self.child_comm)
        self.signal_children()

        status = MPI.Status()
        self.running_workers = list(range(1, self.num_workers + 1))
        self.waiting_workers_list = []

        self.logger.info("Master initialize training")

        if self.monitor:
            self.monitor.start_monitor()

        while self.running_workers:
            t1 = datetime.datetime.now()
            self.recv_any_from_child(status)
            t2 = datetime.datetime.now()
            self.idle_time += (t2 - t1).total_seconds()
            self.process_message(status)

            if self.stop_training:
                self.shut_down_workers()

        if self.monitor:
            self.monitor.stop_monitor()

        np.savez_compressed(self.save_filename + "_weights.npz", *self.weights_to_save)
        np.savez_compressed(self.save_filename + "_biases.npz", *self.biases_to_save)

        self.logger.info("Done training")
        self.logger.info(f"Master idle time: {self.idle_time}")
        # If we did not finish the last epoch, validate one more time.
        # (this happens if the batch size does not divide the dataset size)
        if self.epoch < self.num_epochs:
            self.epoch += 1
            self.validate(self.weights)

        if (self.monitor and self.save_filename != ""):
            with open(self.save_filename + "_monitor.json", 'w') as file:
                file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

        self.send_exit_to_parent()
        self.stop_time = time.time()

    def validate(self, weights):
        return self.validate_aux(weights, self.model)

    def test(self, weights):
        return self.test_aux(weights, self.model)

    def validate_aux(self, weights, model):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics."""
        if self.has_parent:
            return {}
        model.set_weights(weights)

        self.logger.debug("Starting validation")
        t3 = datetime.datetime.now()
        accuracy_test, activations_test = self.model.predict(self.data.x_test, self.data.y_test)
        accuracy_train, activations_train = self.model.predict(self.data.x_train, self.data.y_train)
        t4 = datetime.datetime.now()
        self.best_val_acc = max(self.best_val_acc , accuracy_test)
        loss_test = self.model.compute_loss(self.data.y_test, activations_test)
        loss_train = self.model.compute_loss(self.data.y_train, activations_train)
        self.metrics[self.epoch-1, 0] = loss_train
        self.metrics[self.epoch-1, 1] = loss_test
        self.metrics[self.epoch-1, 2] = accuracy_train
        self.metrics[self.epoch-1, 3] = accuracy_test

        self.logger.info("Validation metrics:")
        self.logger.info(f"Testing time: {t4 - t3}\n; Loss test: {loss_test}; \n"
                          f" Accuracy test: {accuracy_test}; \n"
                          f"Maximum accuracy val: {self.best_val_acc}")

        # save performance metrics values in a file
        if (self.save_filename != ""):
            np.savetxt(self.save_filename + ".txt", self.metrics)

        self.logger.debug("Ending validation")
        return None

    def test_aux(self, weights, model):
        """Compute the loss on the validation data.
            Return a dictionary of validation metrics."""
        if self.has_parent:
            return {}
        model.set_weights(weights)

        self.logger.debug("Starting validation")
        val_metrics = np.zeros((1,))
        i_batch = 0
        for i_batch, batch in enumerate(self.data.generate_test_data()):
            new_val_metrics = model.test_on_batch(x=batch[0], y=batch[1])

            if val_metrics.shape != new_val_metrics.shape:
                val_metrics = np.zeros(new_val_metrics.shape)
            val_metrics += new_val_metrics

        val_metrics = val_metrics / float(i_batch + 1)

        self.logger.info("Validation metrics:")
        self.print_metrics(val_metrics)
        self.logger.debug("Ending validation")
        return None

    ### MPI-related functions below

    def signal_children(self):
        """Sends each child a message telling them to start training"""
        for child in range(1, self.child_comm.Get_size()):
            self.send(obj=None, tag='train', dest=child, comm=self.child_comm)

    def recv_any_from_child(self, status):
        """Receives any message from any child.  Returns the provided status object,
            populated with information about received message"""
        self.recv(tag='any', source=MPI.ANY_SOURCE, status=status, comm=self.child_comm)
        return status

    def send_exit_to_child(self, child, comm=None):
        if comm is None:
            comm = self.child_comm
        return comm.isend(None, dest=child, tag=self.lookup_mpi_tag('exit'))

    def build_model(self):
        """Builds the Keras model and updates model-related attributes"""
        super(MPIMaster, self).build_model()