import numpy as np
import logging
from mpi_training.mpi.process import MPIWorker, MPIMaster
import datetime
import json


class MPISingleWorker(MPIWorker):
    """This class trains its model with no communication to other processes"""
    def __init__(self, num_epochs, data, algo, model,monitor, save_filename):

        self.has_parent = False
        self.best_val_loss = None

        super(MPISingleWorker, self).__init__(data, algo, model, process_comm=None, parent_comm=None,
                                              parent_rank=None, num_epochs=num_epochs, monitor=monitor,
                                              save_filename=save_filename)

    def train(self, testing=True):
        self.check_sanity()

        weights = []
        biases = []

        maximum_accuracy = 0
        metrics = np.zeros((self.num_epochs, 4))
        for epoch in range(1, self.num_epochs + 1):
            logging.info("beginning epoch {:d}".format(self.epoch + epoch))
            if self.monitor:
                self.monitor.start_monitor()

            for i_batch, batch in enumerate(self.data.generate_data()):
                self.update = self.model.train_on_batch(x=batch[0], y=batch[1])

                self.weights = self.algo.apply_update(self.weights, self.update)
                self.algo.set_worker_model_weights(self.model, self.weights)
                self.weights = self.model.get_weights()

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
                metrics[epoch-1, 0] = loss_train
                metrics[epoch-1, 1] = loss_test
                metrics[epoch-1, 2] = accuracy_train
                metrics[epoch-1, 3] = accuracy_test
                self.logger.info(f"Testing time: {t4 - t3}\n; Loss train: {loss_train}; Loss test: {loss_test}; \n"
                                 f"Accuracy train: {accuracy_train}; Accuracy test: {accuracy_test}; \n"
                                 f"Maximum accuracy test: {maximum_accuracy}")
                # save performance metrics values in a file
                if (self.save_filename != ""):
                    np.savetxt(self.save_filename + ".txt", metrics)

            if self.stop_training:
                break

            weights.append(self.weights['w'])
            biases.append(self.weights['b'])
            if epoch < self.num_epochs - 1:  # do not change connectivity pattern after the last epoch
                self.model.weight_evolution()
                self.weights = self.model.get_weights()

        logging.info("Signing off")
        self.model.set_weights(self.weights)

        np.savez_compressed(self.save_filename + "_weights.npz", *weights)
        np.savez_compressed(self.save_filename + "_biases.npz", *biases)

        if (self.save_filename != "" and self.monitor):
            with open(self.save_filename + "_monitor.json", 'w') as file:
                file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

    def test(self, weights):
        return MPIMaster.test_aux(self, weights, self.model)

    def validate(self, weights):
        return MPIMaster.validate_aux(self, weights, self.model)