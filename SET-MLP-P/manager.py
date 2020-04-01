### MPIManager class and associated functions

from __future__ import division
import math
from mpi4py import MPI
import numpy as np
import time
import json
import logging
from train.data import Data
from utils import get_num_gpus


def get_groups(comm, num_masters=1, num_processes=1):
    masters = list(range(0, num_masters))  # index 0 is the uber master, the other one asre sub-masters
    n_active_master = max(num_masters - 1, 1)
    groups = [set() for _ in range(n_active_master)]
    instances_ranks = list(range(num_masters, comm.Get_size(), num_processes))
    processes = [None] * len(instances_ranks)
    for ir, i_rank in enumerate(instances_ranks):
        _, ig = divmod(ir, n_active_master)
        groups[ig].add(i_rank)
        groups[ig].add(masters[-1 - ig])
        processes[ir] = list(range(i_rank, min(i_rank + num_processes, comm.Get_size())))

    groups = [sorted(g) for g in groups]
    return masters, groups, processes


class MPIManager(object):
    """The MPIManager class defines the topology of the MPI process network
        and creates master and worker objects for each process accordingly.
        Two configurations are available:
          1) one master supervising other masters, each controlling some workers
          2) one master and N-1 workers, where N is the number of MPI processes
        Attributes:
          process: the MPI worker or master object running on this process
          data: Data object containing information used for training/validation
          algo: Algo object containing training algorithm configuration options
          model_builder: ModelBuilder object
          num_masters: integer indicating the number of master processes.
            If num_masters > 1, an additional master will be created to supervise all masters.
          num_workers: integer indicating the number of worker processes
          worker_id: ID of worker node, used for indexing training data files
          num_epochs (integer): number of times to iterate over the training data
          comm_block: MPI intracommunicator used for message passing between master and workers.
            Process 0 is the master and the other processes are workers.
          comm_masters: MPI intracommunicator used for message passing between masters.
            (It will be None if there is only one master.)
          train_list: list of training data file names
          val_list: list of validation data file names
          is_master: boolean determining if this process is a master
          should_validate: boolean determining if this process should run training validation
          synchronous: whether or not to syncronize workers after each update
          verbose: whether to make MPIProcess objects verbose
    """

    def __init__(self, comm, data, algo, model, num_epochs, num_masters=1, num_processes=6, synchronous=False,
                 verbose=False, custom_objects={}, early_stopping=None, target_metric=None,
                 monitor=False, thread_validation=False, checkpoint=None, checkpoint_interval=5):
        """Create MPI communicator(s) needed for training, and create worker
            or master object as appropriate.
            Params:
            comm: MPI intracommunicator containing all processes
            data: Data object containing information used for training/validation
            algo: Algo object containing training algorithm configuration options
            model_builder: ModelBuilder object
            num_masters: number of master processes
            num_processes: number of processes that make up a worker/master (gpu allreduce)
            num_epochs: number of times to iterate over the training data
            train_list: list of training data files
            val_list: list of validation data files
            synchronous: true if masters should operate in synchronous mode
            verbose: whether to make MPIProcess objects verbose
            monitor: whether to monitor per-process resource (CPU/GPU) usage
        """
        self.data = data
        self.algo = algo
        self.model = model
        self.num_masters = num_masters
        self.num_processes = num_processes
        if comm.Get_size() != 1:
            n_instances, remainder = divmod(comm.Get_size() - self.num_masters, self.num_processes)
        else:
            n_instances, remainder = 1, 0
        if remainder:
            logging.info("The accounting is not correct, %d nodes are left behind", remainder)
        self.num_workers = n_instances  # - self.num_masters
        self.worker_id = -1

        self.num_epochs = num_epochs

        self.synchronous = synchronous
        self.verbose = verbose
        self.monitor = monitor
        self.comm_block = None
        self.comm_masters = None
        self.comm_instance = None
        self.is_master = None
        self.should_validate = None
        self.custom_objects = custom_objects
        self.early_stopping = early_stopping
        self.target_metric = target_metric
        self.thread_validation = thread_validation
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.make_comms(comm)

    def make_comms(self, comm):
        """Define the network topology by creating communicators linking masters with their slaves.
            Set comm_block to contain one master and all of its workers.
            Set comm_masters to contain all masters, including the "super-master" supervising them.
            Set comm_instance to contain the nodes contributing to the woker/master instance
            Define a master or worker object as appropriate for each process.
            If a worker is created, it is assigned some data files to train on.
        """
        # For masters we let child_comm be the communicator used to message the node's
        # children, and parent_comm be that used to message the node's parents.

        self.parent_rank = 0
        self.is_master = False
        rank = comm.Get_rank()
        masters, groups, processes = get_groups(comm, self.num_masters, self.num_processes)

        logging.debug("masters %s", str(masters))
        if len(masters) > 1:
            self.comm_masters = comm.Create(comm.Get_group().Incl(masters))
        if rank in masters:
            ## make the communicator for masters
            self.is_master = True
        if rank == 0:
            self.should_validate = True

        logging.debug("groups %s", str(groups))
        for igr, gr in enumerate(groups):
            if rank in gr:
                self.comm_block = comm.Split(igr)
                break

        if self.comm_block is None:
            _ = comm.Split(len(groups))

        if not self.is_master and self.comm_block:
            self.worker_id = self.comm_block.Get_rank()

        logging.debug("processes %s", str(processes))
        for ipr, pr in enumerate(processes):
            if rank in pr and len(pr) > 1:
                ## make the communicator for that process group
                self.comm_instance = comm.Split(ipr)
                break

        if not self.comm_instance:
            _ = comm.Split(len(processes))

        if self.comm_instance:
            ids = self.comm_instance.allgather(self.worker_id)
            self.worker_id = list(filter(lambda i: i != -1, ids))[0]

        logging.debug("master comm", self.comm_masters.Get_size() if self.comm_masters else "N/A")
        logging.debug("block comm", self.comm_block.Get_size() if self.comm_block else "N/A")
        logging.debug("instance comm", self.comm_instance.Get_size() if self.comm_instance else "N/A")

        # Case (1)
        if self.num_masters > 1:
            if self.is_master:
                parent_comm = self.comm_masters
                # if rank==0:
                if self.comm_masters.Get_rank() == 0:
                    child_comm = self.comm_masters
                    self.parent_rank = None
                else:
                    child_comm = self.comm_block
        else:
            if self.is_master:
                parent_comm = self.comm_block
                child_comm = self.comm_block
                self.parent_rank = None

        # Process initialization
        if comm.Get_size() != 1:
            from process import MPIWorker, MPIMaster
            if self.is_master:

                num_sync_workers = self.get_num_sync_workers(child_comm)
                self.process = MPIMaster(parent_comm, parent_rank=self.parent_rank,
                                         data=self.data, algo=self.algo, model=self.model,
                                         child_comm=child_comm, num_epochs=self.num_epochs,
                                         num_sync_workers=num_sync_workers,
                                         verbose=self.verbose, custom_objects=self.custom_objects,
                                         early_stopping=self.early_stopping, target_metric=self.target_metric,
                                         threaded_validation=self.thread_validation,
                                         checkpoint=self.checkpoint, checkpoint_interval=self.checkpoint_interval
                                         )
            else:

                self.process = MPIWorker(data=self.data, algo=self.algo,
                                         model=self.model,
                                         process_comm=self.comm_instance,
                                         parent_comm=self.comm_block,
                                         parent_rank=self.parent_rank,
                                         num_epochs=self.num_epochs,
                                         verbose=self.verbose,
                                         monitor=self.monitor,
                                         custom_objects=self.custom_objects,
                                         checkpoint=self.checkpoint, checkpoint_interval=self.checkpoint_interval
                                         )
        else:  # Single Process mode
            from single_process import MPISingleWorker
            self.process = MPISingleWorker(data=self.data, algo=self.algo,
                                           model=self.model,
                                           num_epochs=self.num_epochs,
                                           verbose=self.verbose,
                                           monitor=self.monitor,
                                           custom_objects=self.custom_objects,
                                           early_stopping=self.early_stopping,
                                           target_metric=self.target_metric,
                                           checkpoint=self.checkpoint, checkpoint_interval=self.checkpoint_interval)

    def figure_of_merit(self):
        ##if (self.comm_masters and self.comm_masters.Get_rank() == 0) or (self.comm_block.Get_rank() == 0):
        if self.parent_rank is None:
            ## only the uber-master returns a valid fom
            return self.process.model.figure_of_merit()
        else:
            return None

    def train(self):
        if self.parent_rank is None:
            return self.process.train()
        else:
            return None

    def get_num_sync_workers(self, comm):
        """Returns the number of workers the master should wait for
            at each training time step.  Currently set to 95% of the
            number of workers (or 1 if running asynchronously).
            comm should be the master's child communicator."""
        if self.synchronous:
            return int(math.ceil(0.95 * (comm.Get_size() - 1)))
        return 1

    def free_comms(self):
        """Free active MPI communicators"""
        if self.process.process_comm is not None:
            logging.debug("holding on %d", self.process.process_comm.Get_size())
            self.process.process_comm.Barrier()
        if self.comm_block is not None:
            self.comm_block.Free()
        if self.comm_masters is not None:
            self.comm_masters.Free()
        if self.comm_instance is not None:
            self.comm_instance.Free()


class MPIKFoldManager(MPIManager):
    def __init__(self, NFolds, comm, data, algo, model_builder, num_epochs, train_list,
                 val_list, num_masters=1,
                 num_process=1,
                 synchronous=False,
                 verbose=False, custom_objects={},
                 early_stopping=None, target_metric=None,
                 monitor=False,
                 checkpoint=None, checkpoint_interval=5):
        self.comm_world = comm
        self.comm_fold = None
        self.fold_num = None
        if NFolds == 1:
            ## make a regular MPIManager
            self.manager = MPIManager(comm, data, algo, model_builder, num_epochs, train_list,
                                      val_list, num_masters, num_process,
                                      synchronous,
                                      verbose, custom_objects,
                                      early_stopping, target_metric,
                                      monitor,
                                      checkpoint=checkpoint, checkpoint_interval=checkpoint_interval)
            return

        if int(comm.Get_size() / float(NFolds)) <= 1:
            logging.warning("There is less than one master+one worker per fold, this isn't going to work")

        ## actually split further the work in folds
        rank = comm.Get_rank()
        fold_num = int(rank * NFolds / comm.Get_size())
        self.fold_num = fold_num
        self.comm_fold = comm.Split(fold_num)
        logging.debug(
            "For node {}, with block rank {}, send in fold {}".format(MPI.COMM_WORLD.Get_rank(), rank, fold_num))
        self.manager = None

        if val_list:
            logging.warning("MPIKFoldManager would not expect to be given a validation list")
        all_files = train_list + val_list
        from sklearn.model_selection import KFold
        folding = KFold(n_splits=NFolds)
        folds = list(folding.split(all_files))
        train, test = folds[fold_num]
        train_list_on_fold = list(np.asarray(all_files)[train])
        val_list_on_fold = list(np.asarray(all_files)[test])
        self.manager = MPIManager(self.comm_fold, data, algo, model_builder, num_epochs, train_list_on_fold,
                                  val_list_on_fold, num_masters, num_process,
                                  synchronous,
                                  verbose, custom_objects,
                                  early_stopping, target_metric, monitor,
                                  checkpoint=checkpoint, checkpoint_interval=checkpoint_interval)

    def free_comms(self):
        self.manager.free_comms()

    def train(self):
        self.manager.train()

    def figure_of_merit(self):
        fom = self.manager.figure_of_merit()
        if self.comm_fold is not None:
            foms = self.comm_world.allgather(fom)
            # filter out the None values
            foms = list(filter(None, foms))
            ## make the average and rms
            avg_fom = np.mean(foms)
            std_fom = np.std(foms)
            if self.comm_fold.Get_rank() == 0:
                logging.info("Figure of merits over {} folds is {}+/-{}".format(len(foms), avg_fom, std_fom))
            return avg_fom
        else:
            if fom is not None:
                logging.info("Figure of merits from single value {}".format(fom))
            return fom