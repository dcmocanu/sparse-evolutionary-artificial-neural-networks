from set_mlp  import *
import torch
import types
import copy

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
    def __init__(self, model, id , parent):

        self.parent = parent
        self.id = id
        self.delay = 1

        self.model = model

        self.batch_size = model.batch_size

        self.loss_function = MSE
        self.epoch = 0

    def get_next_mini_batch(self):

        pass

    def get_server_weights(self):

        params = self.parent.queue.sample(self.delay)
        for param_1, param_2 in zip(self.model.parameters(), params):
            param_1.data = param_2.clone().detach().requires_grad_().data
        del params

    def assign_weights(self, model):
        """
        Takes in a model and assigns the weights of the model to self.model.
        Skipping the check for model and self.model belonging to the same nn.Module type.
        :param model:
        :return:
        """
        for param_1, param_2 in zip(self.model.parameters(), model.parameters()):
            param_1.data = param_2.data

    def compute_gradients(self):

        start_time = time.time()
        self.get_server_weights()
        start_time = time.time()

        batchdata, batchlabels = self.get_next_mini_batch()     # passes to device already
        start_time = time.time()

        output = self.model.forward(batchdata)

        loss = self.loss_function(output, batchlabels)
        self.model.zero_grad()
        loss.backward()

        # compute the gradients and return the list of gradients

        self.nn_time_meter.update(time.time() - start_time, 1)

        return [param.grad.data.cpu() for param in self.model.parameters()], loss.data, batchlabels.size(0)

class ParameterServer:
    def __init__(self, no_neurons, batch_size, epochs=10, epsilon=13):

        #  Training related
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = 0.3  # the fraction of the weights removed
        self.dropout_rate = 0.2  # dropout rate
        self.epoch = 0
        self.init_lr = 0.05
        self.lr_schedule = 'decay'
        self.lr = self.init_lr
        self.decay = 0.0
        self.batch_size = batch_size
        self.epochs = epochs
        self.no_neurons = no_neurons

        # server worker related parameters
        # location, foldername added new as compared to ParameterServer
        self.max_delay = 1
        self.queue = Queue(self.max_delay + 1)
        self.num_workers = 12
        self.workers = []
        #self.inf_batch_size = 10000
        self.delaytype = 'const'

        # data loading
        self.x_train, self.y_train, self.x_test, self.y_test = load_cifar10_data(2000, 1000)

        #self.train_loader = DataLoader(self.train_data, batch_size=10000, num_workers=8)
        #self.test_loader = DataLoader(self.test_data, batch_size=10000, num_workers=8)

        # choosing model and loss function
        self.dimensions = (self.x_train.shape[1], no_neurons, no_neurons, no_neurons,
                              self.y_train.shape[1])
        self.model = SET_MLP(self.dimensions, (Relu, Relu, Relu, Sigmoid), MSE, epsilon=epsilon)
        self.n_layers = len(self.dimensions)

        # Functions to be called at init
        #self.update_queue()

    def update_queue(self, reset=False):
        if reset:
            self.queue.queue = []
            self.queue.len = 0
            self.queue.push([param.clone().detach().requires_grad_().cpu() for param in self.model.parameters()])
        else:
            self.queue.push([param.clone().detach().requires_grad_().cpu() for param in self.model.parameters()])

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
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for p in parameters:
            param_norm = p.data.norm(2)
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

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        total_norm = self.compute_norm(parameters)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            if inplace:
                for p in parameters:
                    p.data.mul_(clip_coef)
                return total_norm, parameters
            else:
                if isinstance(parameters, types.GeneratorType):
                    parameters = list(parameters)

                cp = copy.deepcopy(parameters)
                for p in cp:
                    p.data.mul_(clip_coef)
                return total_norm, cp
        return total_norm, parameters

    def train(self, testing=True):
        # Initiate the loss object with the final activation function
        # self.save_filename = "Results/set_mlp_"+str(self.x_train[0])+"_training_samples_e"+str(self.epsilon)+"_rand"+str(1)
        # self.inputLayerConnections = []
        # self.inputLayerConnections.append(self.model.getCoreInputConnections())
        # np.savez_compressed(self.save_filename + "_input_connections.npz",
        #                     inputLayerConnections=self.inputLayerConnections)

        best_acc = 0
        metrics = np.zeros((self.epochs, 4))
        #num_iter_per_epoch = len(self.partitions[0])//self.batch_size + 1
        running_itr = 0

        for epoch in range(self.epochs):

            self.epoch = epoch
            start_time = time.time()
            # Shuffle the data
            seed = np.arange(self.x_train.shape[0])
            np.random.shuffle(seed)
            x_ = self.x_train[seed]
            y_ = self.y_train[seed]

            # try:
            #     self.step()
            # except Exception as e:
            #     # propagating exception
            #     print('exception in step ', e)
            #     raise e

            for j in range(self.x_train.shape[0] // self.batch_size):
                k = j * self.batch_size
                l = (j + 1) * self.batch_size
                z, a = self.model._feed_forward(x_[k:l], True)

                self.model._back_prop(z, a, y_[k:l])

            step_time = time.time() - start_time
            print("\nSET-MLP Epoch ", epoch)
            print("Training time: ", step_time)

            running_itr += 1
            #self.lr_update(running_itr, epoch)

            if testing:
                print("epoch test loss")
                start_time = time.time()
                accuracy_test, activations_test = self.predict(self.x_test, self.y_test, self.batch_size)
                test_time = time.time() - start_time
                print("test time", test_time)

                print("epoch train loss")
                start_time = time.time()
                accuracy_train, activations_train = self.predict(self.x_train, self.y_train, self.batch_size)
                train_time = time.time() - start_time
                print("train time", train_time)

                is_best = (accuracy_test > best_acc)
                best_acc = max(accuracy_test, best_acc)

                loss_test = self.model.loss.loss(self.y_test, activations_test)
                loss_train = self.model.loss.loss(self.y_train, activations_train)
                metrics[epoch, 0] = loss_train
                metrics[epoch, 1] = loss_test
                metrics[epoch, 2] = accuracy_train
                metrics[epoch, 3] = accuracy_test
                print("Testing time: ", test_time,"; Loss train: ", loss_train, "; Loss test: ", loss_test, "; Accuracy train: ", accuracy_train,
                      "; Accuracy test: ", accuracy_test, "; Maximum accuracy test: ", best_acc)

        print("training completed")

    def step(self):

        grads = {}
        loss = 0
        batch_size = 0

        delays = []
        grad_norms = []
        losses = []
        start_time = time.time()

        for worker_ in self.workers:

            # worker_.get_server_weights()

            grads[worker_.id], worker_loss, batch_size_ = worker_.compute_gradients()

            # Check for nan values and exit the program. Also sends an email
            for wg in grads[worker_.id]:
                if torch.isnan(wg).any() or torch.isinf(wg).any():
                    # self.nan_handler(msg=str(worker_.id) + 'grad')
                    raise Exception('found Nan/Inf values')

            if torch.isnan(worker_loss) or torch.isinf(worker_loss).any():
                # self.nan_handler(msg=str(worker_.id) + 'loss')
                raise Exception('found Nan/Inf values')

            batch_size += batch_size_
            loss += worker_loss * batch_size_

            losses.append(worker_loss * batch_size)
            grad_norms.append(self.compute_norm(grads[worker_.id]))
            delays.append(worker_.delay)

        self.aggregate_gradients(grads)
        loss /= batch_size

    def aggregate_gradients(self, grads):

        # average gradients across all workers (includes cached gradients)

        for id_ in range(1, len(grads)):
            for param1, param2 in zip(grads[0], grads[id_]):
                param1.data += param2.data

        for param in grads[0]:
            param.data /= len(grads)

        # norm_before_clip = nn.utils.clip_grad_norm_(grads[0], 1)
        # norm_before_clip = self.compute_norm(grads[0])
        norm_before_clip, _ = self.clip_(grads[0], max_norm=1, inplace=True)
        norm_after_clip = self.compute_norm(grads[0])

        # Assign grad data to model grad data. Update parameters of the model
        for param1, param2 in zip(self.model.parameters(), grads[0]):
            param1.data -= self.lr * param2.data

        self.update_queue()

    def predict(self, x_test, y_test, batch_size=1):

        """
               :param x_test: (array) Test input
               :param y_test: (array) Correct test output
               :param batch_size:
               :return: (flt) Classification accuracy
               :return: (array) A 2D array of shape (n_cases, n_classes).
               """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test = self.model._feed_forward(x_test[k:l])
            activations[k:l] = a_test[self.n_layers]
        correctClassification = 0
        for j in range(y_test.shape[0]):
            if (np.argmax(activations[j]) == np.argmax(y_test[j])):
                correctClassification += 1
        accuracy = correctClassification / y_test.shape[0]
        return accuracy, activations
