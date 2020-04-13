### Monitor class

import os
from datetime import datetime
from threading import Thread
import psutil
import time


class Monitor(object):
    """ Class that monitors CPU utilization for a given time """

    def __init__(self, sampling_rate=None, pid=None, save_filename=None):
        self.sampling_rate = sampling_rate if sampling_rate is not None else 0.5
        self.pid = pid if pid is not None else os.getpid()

        self.should_stop = False
        self.thread = None
        self.accounting_enabled = False
        self.save_filename = save_filename

        self.samples = []

    def _monitor(self):

        while not self.should_stop:

            cpu_process = psutil.Process(self.pid)

            # get the name of the file executed
            name = cpu_process.name()

            try:
                # get the number of CPU cores that can execute this process
                cores = len(cpu_process.cpu_affinity())
            except psutil.AccessDenied:
                cores = 0
                # get the CPU usage percentage
            cpu_usage = cpu_process.cpu_percent()

            # get the number of total threads spawned by this process
            n_threads = cpu_process.num_threads()

            # total process read and written bytes
            io_counters = cpu_process.io_counters()
            read_bytes = io_counters.read_bytes
            write_bytes = io_counters.write_bytes

            try:
                # get the memory usage in MB
                memory_usage = cpu_process.memory_full_info().uss / (1024. * 1024.)
            except psutil.AccessDenied:
                memory_usage = 0

            try:
                # get the process priority (a lower value means a more prioritized process)
                nice = int(cpu_process.nice())
            except psutil.AccessDenied:
                nice = 0

            # get the time the process was spawned
            try:
                create_time = datetime.fromtimestamp(cpu_process.create_time())
            except OSError:
                # system processes, using boot time instead
                create_time = datetime.fromtimestamp(psutil.boot_time())

            # get the status of the process (running, idle, etc.)
            status = cpu_process.status()

            current_sample = {
                'pid': self.pid, 'name': name, 'create_time': create_time,
                'cores': cores, 'cpu_usage': cpu_usage, 'status': status, 'nice': nice,
                'memory_usage': memory_usage, 'read_bytes': read_bytes, 'write_bytes': write_bytes,
                'n_threads': n_threads
            }

            self.samples.append(current_sample)

            time.sleep(self.sampling_rate)

    def start_monitor(self):
        self.should_stop = False
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def stop_monitor(self):
        self.should_stop = True
        self.thread.join()

    def get_stats(self):
        return self.samples