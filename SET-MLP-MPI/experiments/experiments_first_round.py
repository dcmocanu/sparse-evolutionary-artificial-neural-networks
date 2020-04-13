""""
*** First round of experiments to benchmark the baseline for Parallel SET Training Algorithm ***
An initial set of experiments need to be set in order to assess the performance of the Parallel SET
Training, which is the first (basic) version of the algorithm. By systematically analyzing the algorithm
against the baseline (sequential training), many improvement points can be found to further
refine the methodology and achieve the final goals of this research. After the first experimental
stage, the first parallel version and the sequential training will remain the stable baselines to benchmark
our new solutions.

Metrics to consider during the evaluation:
- Speed performance, overall training time under equivalent settings
- Accuracy performance on the test, train and validation data set
- Generalization performance (loss), check robustness to over-fitting behaviour

Requirements:
- Save the degree distribution of the neurons (weight matrices and  biases, directly before each topology change).
- Track Master and Workers idle time
- Log desired metrics
- Average metrics over three runs (at least)
- Make an experiment where we gradually increase the number of workers (from 2 up to 6)
- Make sure to ave consistent settings for all models under analysis

Design choices:
- SGD with momentum and weight decay
- ReLu activation function, last layer with Sigmoid and MSE loss
- Fix learning rate with dropout

Algorithms versions to compare:
- MLP (Fully connected)
- SET sequential version
- FixProb sequential version
- SET with mask over weights (Keras implementation)
- Parallel SET Training (Sync version)
- Parallel SET Training (Async version)
- Parallel FixProb Training (Sync version)
- Parallel FixProb Training (Async version)

Dataset (for now only one to keep things easy):
- CIFAR10 (training_samples=50000, validation_samples=10000, testing_samples=10000)
- Normalized using zero mean and one standard deviation (training dataset)
- Consider to use augmented version
- For the parallel version the train data is partitioned among all the workers

Parameters:
- Architecture dimensions = (3072, 4000, 1000, 4000, 10)
- Activations = (Relu, Relu, Relu, Sigmoid)
- ε = 20, and ζ = 0.3
- dropout =0.3 (or 0.2)
- 1000 training epochs (or equivalent)
- momentum = 0.9, and weight_decay = 0.0002
- learning_rate = 0.05
- batch_size = 128
- validate_every = (epochs//batch_size) * num_workers , and sync_every = 1
"""

import argparse
import logging
import numpy as np
from models.set_mlp_mpi import *
from utils.load_data import *
from mpi4py import MPI
from time import time
from mpi_training.mpi.manager import MPIManager
from mpi_training.train.algo import Algo
from mpi_training.train.data import Data
from mpi_training.train.model import MPIModel
from mpi_training.logger import initialize_logger

if __name__ == '__main__':
    pass