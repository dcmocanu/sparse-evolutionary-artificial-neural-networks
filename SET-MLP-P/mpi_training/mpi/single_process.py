import numpy as np
import logging
from mpi_training.mpi.process import MPIWorker, MPIMaster


class MPISingleWorker(MPIWorker):
    """This class trains its model with no communication to other processes"""
    def __init__(self, num_epochs, data, algo, model, verbose, monitor):

        self.has_parent = False

        self.best_val_loss = None

        super(MPISingleWorker, self).__init__(data, algo, model, process_comm=None, parent_comm=None, parent_rank=None,
            num_epochs=num_epochs, verbose=verbose, monitor=monitor)

    def train(self, testing=False):
        self.check_sanity()

        for epoch in range(1, self.num_epochs + 1):
            logging.info("beginning epoch {:d}".format(self.epoch + epoch))
            if self.monitor:
                self.monitor.start_monitor()
            epoch_metrics = np.zeros((1,))
            i_batch = 0

            for i_batch, batch in enumerate(self.data.generate_data()):
                train_metrics = self.model.train_on_batch(x=batch[0], y=batch[1])

                if epoch_metrics.shape != train_metrics.shape:
                    epoch_metrics = np.zeros(train_metrics.shape)
                epoch_metrics += train_metrics

                ######
                # self.update = self.algo.compute_update(self.weights, self.model.get_weights())
                # self.weights = self.algo.apply_update(self.weights, self.update)
                # self.algo.set_worker_model_weights(self.model, self.weights)
                ######

                self.weights = self.model.get_weights()

            if self.monitor:
                self.monitor.stop_monitor()
            epoch_metrics = epoch_metrics / float(i_batch+1)

            if testing:
                self.logger.info("Epoch metrics:")
                self.print_metrics(epoch_metrics)

            if self.stop_training:
                break

            self.validate()

            if epoch < self.num_epochs:  # do not change connectivity pattern after the last epoch
                self.model.weight_evolution()
                self.weights = self.model.get_weights()

        logging.info("Signing off")
        self.model.set_weights(self.weights)

        if self.monitor:
            self.update_monitor(self.monitor.get_stats())

    def validate(self):
        return MPIMaster.validate_aux(self, self.weights, self.model)