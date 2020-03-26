from set_mlp  import *
import types
from random import Random, shuffle
import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp


def shared_randomness_partitions(n, num_workers):
    # remove last data point
    dinds = list(range(n))
    #shuffle(dinds)
    worker_size = (n) // num_workers

    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        data[w] = dinds[w * worker_size: (w+1) * worker_size]

    return data


class Queue:
    def __init__(self, max_len):
        self.queue = list()
        self.maxlen = max_len
        self.len = 0

    def push(self, grad):

        if self.len < self.maxlen:
            self.queue.append(grad)
            self.len += 1
        else:
            ret = self.queue.pop(0)
            self.queue.append(grad)

    def sample(self, delay):

        if delay >= self.len:
            return self.queue[0]
        # print(delay)
        # i-th element in the queue is the i step delayed version of the param
        return self.queue[self.len - delay - 1]


class Worker:
    def __init__(self, model, id, batch_size, parent, data, labels, delay=1):

        self.parent = parent
        self.id = id
        self.delay = delay
        self.model = model
        self.batch_size = batch_size
        self.loss_function = model.loss
        self.epoch = 0

        self.data = data
        self.labels = labels

    def get_next_mini_batch(self):

        data, labels = self.data, self.labels

        return data, labels

    def get_server_weights(self):

        params = self.parent.queue.sample(self.delay)

        self.model.set_parameters(copy.deepcopy(params))

        del params

    def assign_weights(self, model):
        """
        Takes in a model and assigns the weights of the model to self.model.
        :param model:
        :return:
        """

        self.model.set_parameters(copy.deepcopy(model.parameters()))

    def compute_gradients(self, data=None, label=None):

        #time.sleep(self.id)
        #self.get_server_weights()
        self.assign_weights(self.parent.model)

        batchdata, batchlabels = self.get_next_mini_batch()     # passes to device already

        z, a  = self.model._feed_forward(batchdata)

        self.model._back_prop(z, a, batchlabels)

        accuracy, activations = self.model.predict(batchdata, batchlabels, batchlabels.shape[0])

        loss = self.model.loss.loss(batchlabels, activations)

        pdw = self.model.parameters()['pdw']
        pdd = self.model.parameters()['pdd']

        self.parent.model.set_parameters(copy.deepcopy(self.model.parameters()))

        # compute the gradients and return the list of gradients
        return pdw, pdd, loss, batchlabels.shape[0]


class ParameterServer:
    def __init__(self, **config):

        #  Training related
        self.momentum = config['momentum']
        self.weight_decay = config['weight_decay']
        self.epsilon = config['epsilon']  # control the sparsity level as discussed in the paper
        self.zeta = config['zeta']   # the fraction of the weights removed
        self.dropout_rate = config['dropout_rate']   # dropout rate
        self.init_lr = config['lr']
        self.lr_schedule = 'decay'
        self.decay = config['lr_decay']
        self.batch_size = config['batch_size']
        self.epochs = config['n_epochs']
        self.no_neurons = config['n_hidden_neurons']
        self.n_training_samples = config['n_training_samples']
        self.n_testing_samples = config['n_testing_samples']
        self.num_workers = config['n_processes']

        self.lr = self.init_lr
        self.epoch = 0

        # server worker related parameters
        # location, foldername added new as compared to ParameterServer
        self.max_delay = 1
        self.queue = Queue(self.max_delay + 1)
        self.workers = []
        self.partitions = {}
        self.delaytype = config['delay_type']

        # data loading
        self.x_train, self.y_train, self.x_test, self.y_test = load_cifar10_data(self.n_training_samples, self.n_testing_samples)

        # set model dimensions
        self.dimensions = (self.x_train.shape[1], self.no_neurons, self.no_neurons, self.no_neurons,
                              self.y_train.shape[1])

        print("Number of neurons per layer:", self.x_train.shape[1], self.no_neurons, self.no_neurons, self.no_neurons,
              self.y_train.shape[1])

        # Instantiate master model
        self.model = SET_MLP(self.dimensions, (Relu, Relu, Relu, Sigmoid), **config)
        self.n_layers = len(self.dimensions)
        self.config = config

        self.num_workers = self.x_train.shape[0] // self.batch_size

        self.update_queue()

    def initiate_workers(self):
        self.partitions = shared_randomness_partitions(len(self.x_train), self.num_workers)
        # initialize workers on the server
        self.workers = []
        for id_ in range(self.num_workers):
            self.workers.append(Worker(parent=self, id=id_, data=self.x_train[self.partitions[id_]],
                                     labels=self.y_train[self.partitions[id_]],
                                     batch_size=self.batch_size, model=copy.deepcopy(self.model)))

    def update_queue(self, reset=False):
        if reset:
            self.queue.queue = []
            self.queue.len = 0
            self.queue.push(copy.deepcopy(self.model.parameters()))
        else:
            self.queue.push(copy.deepcopy(self.model.parameters()))

    def lr_update(self, itr, epoch):
        if self.lr_schedule == 'const':
            self.lr = self.init_lr
        elif self.lr_schedule == 'decay':
            # do not use this currently. lr_update is being called after each step, so it leads to lr->0.
            self.lr = self.init_lr/(1 + self.decay * self.epoch)
        elif self.lr_schedule == 't':
            # lr = c/t
            self.lr = self.init_lr / (1 + self.epoch)

        return

    def compute_norm(self, parameters):
        total_norm = 0

        for p in parameters:
            param_norm = parameters[p].norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def clip_(self, parameters, max_norm, inplace=True):
        """
        If inplace, parameters gets changed. Else, a deepcopy of the parameters is made and which is updated and returned
        Either ways, parameters or cp is returned.
        # Note: Do not pass a generator object for parameters. Always pass a list.
        :param parameters:
        :param max_norm:
        :param inplace:
        :return:
        """

        parameters = [parameters]

        total_norm = self.compute_norm(parameters)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            if inplace:
                for p in parameters:
                    parameters[p] * clip_coef
                return total_norm, parameters
            else:
                if isinstance(parameters, types.GeneratorType):
                    parameters = list(parameters)

                cp = copy.deepcopy(parameters)
                for p in cp:
                    parameters[p] * clip_coef
                return total_norm, cp
        return total_norm, parameters

    def train(self, testing=False):
        best_acc = 0
        metrics = np.zeros((self.epochs, 4))
        #num_iter_per_epoch = len(self.partitions[0])//self.batch_size + 1
        running_itr = 0

        for epoch in range(self.epochs):

            self.epoch = epoch
            print("\nSET-MLP Epoch ", epoch)
            start_time = time.time()

            # Shuffle the data
            seed = np.arange(self.x_train.shape[0])
            np.random.shuffle(seed)
            self.x_train = self.x_train[seed]
            self.y_train = self.y_train[seed]

            # 1. Parallel training (WIP)
            #self.step()

            # 2.1 Sequential training with workers (OK)
            # pdd = {}
            # pdw = {}
            # for j in range(self.x_train.shape[0] // self.batch_size):
            #    k = j * self.batch_size
            #    l = (j + 1) * self.batch_size
            #    self.workers[j].data = self.x_train[k:l]
            #    self.workers[j].labels = self.y_train[k:l]
            #
            #    pdw[self.workers[j].id], pdd[self.workers[j].id], worker_loss, batch_size_ = self.workers[j].compute_gradients()
            #
            #    self.model.set_parameters(copy.deepcopy(self.workers[j].model.parameters()))

            # 2.2 Sequential training with workers (OK)
            # partitions = shared_randomness_partitions(len(self.x_train), self.num_workers)
            # self.partitions = partitions
            # for worker_ in self.workers:
            #     worker_.data = self.x_train[self.partitions[worker_.id]]
            #     worker_.labels = self.y_train[self.partitions[worker_.id]]
            #
            #     pdw[worker_.id], pdd[worker_.id], worker_loss, batch_size_ = worker_.compute_gradients()
            #
            #     self.model.set_parameters(copy.deepcopy(worker_.model.parameters()))

            # 3. Classic training (OK)
            for j in range(self.x_train.shape[0] // self.batch_size):
                k = j * self.batch_size
                l = (j + 1) * self.batch_size
                z, a = self.model._feed_forward(self.x_train[k:l], False)

                self.model._back_prop(z, a, self.y_train[k:l])

            step_time = time.time() - start_time
            print("Training time: ", step_time)

            running_itr += 1

            # Update learning rate
            # self.lr_update(running_itr, epoch)

            # test model performance on the test data at each epoch
            # this part is useful to understand model performance and can be commented for production settings
            if True:
                print("epoch test loss")
                start_time = time.time()
                accuracy_test, activations_test = self.model.predict(self.x_test, self.y_test, self.batch_size)
                test_time = time.time() - start_time
                print("test time", test_time)

                print("epoch train loss")
                start_time = time.time()
                accuracy_train, activations_train = self.model.predict(self.x_train, self.y_train, self.batch_size)
                train_time = time.time() - start_time
                print("train time", train_time)

                # is_best = (accuracy_test > best_acc)
                best_acc = max(accuracy_test, best_acc)

                loss_test = self.model.loss.loss(self.y_test, activations_test)
                loss_train = self.model.loss.loss(self.y_train, activations_train)
                metrics[epoch, 0] = loss_train
                metrics[epoch, 1] = loss_test
                metrics[epoch, 2] = accuracy_train
                metrics[epoch, 3] = accuracy_test
                print("Testing time: ", test_time,"; Loss train: ", loss_train, "; Loss test: ", loss_test, "; Accuracy train: ", accuracy_train,
                      "; Accuracy test: ", accuracy_test, "; Maximum accuracy test: ", best_acc)

            t5 = datetime.datetime.now()
            if epoch < self.epochs - 1:  # do not change connectivity pattern after the last epoch

                # self.weightsEvolution_I() #this implementation is more didactic, but slow.
                self.model.weightsEvolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
            t6 = datetime.datetime.now()
            print("Weights evolution time ", t6 - t5)

        print("training completed")

    def step(self):

        pdw = {}
        pdd = {}

        # self.initiate_workers()
        # for j in range(self.x_train.shape[0] // self.batch_size):
        #     k = j * self.batch_size
        #     l = (j + 1) * self.batch_size
        #     with ThreadPoolExecutor() as executor:
        #         results = executor.map(Worker.compute_gradients, self.workers,
        #                                self.x_train[k:l],
        #                                self.y_train[k:l])
        #         for i, res in enumerate(results):
        #             pdw[i] = res[0]
        #             pdd[i] = res[1]
        #             losses.append(res[2])

        with ThreadPoolExecutor() as executor:
            results = executor.map(Worker.compute_gradients, self.workers)
            for i, res in enumerate(results):
                pdw[i] = res[0]
                pdd[i] = res[1]
        # self.aggregate_gradients(pdw, pdd)

    def aggregate_gradients(self, pdw, pdd):

        # average gradients across all workers (includes cached gradients)

        for id_ in range(1, len(pdw)):
            for key, param in pdw[id_].items():
                pdw[0][key] += param

        for _, param in pdw[0].items():
            param /= len(pdw)

        for id_ in range(1, len(pdd)):
            for key, param in pdd[id_].items():
                pdd[0][key] += param

        for _, param in pdd[0].items():
            param /= len(pdd)

        # norm_before_clip = nn.utils.clip_grad_norm_(grads[0], 1)
        # norm_before_clip = self.compute_norm(grads[0])
        #norm_before_clip, _ = self.clip_(grads[0], max_norm=1, inplace=True)
        #norm_after_clip = self.compute_norm(grads[0])

        # Assign grad data to model grad data. Update parameters of the model
        for (id1, param1), (id2, param2) in zip(pdw[0].items(), pdd[0].items()):
            self.update_parameters(id1, param1, param2)

        #self.update_queue()

    def update_parameters(self, index, pdw, pdd):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.model.pdw[index] = pdw
        self.model.pdd[index] = pdd

        self.model.w[index] += self.model.pdw[index] - self.weight_decay * self.model.w[index]
        self.model.b[index] += self.model.pdd[index] - self.weight_decay * self.model.b[index]

    def predict(self, x_test, y_test, batch_size=1):

        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test = self.model._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]

        correct_classification = 0

        for j in range(y_test.shape[0]):
            if np.argmax(activations[j]) == np.argmax(y_test[j]):
                correct_classification += 1
        accuracy = correct_classification / y_test.shape[0]

        return accuracy, activations
