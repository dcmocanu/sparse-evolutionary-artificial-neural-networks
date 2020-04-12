### Monitor class

import os
import logging
from threading import Thread
import psutil
import time


class Monitor(object):
    """ Class that monitors CPU utilization for a given time """

    def __init__(self, sampling_rate=None, pid=None):
        self.sampling_rate = sampling_rate if sampling_rate is not None else 0.5
        self.pid = pid if pid is not None else os.getpid()

        self.should_stop = False
        self.thread = None
        self.accounting_enabled = False
        self.stats = []

    def _monitor(self):
        current_sample = []
        while not self.should_stop:

            cpu_process = psutil.Process(self.pid)
            used_cpu = psutil.cpu_percent() / float(psutil.cpu_count())  # CPU utilization in %
            used_cpumem = cpu_process.memory_info().rss / (1024. * 1024.)  # Memory use in MB

            current_sample.append((used_cpu, used_cpumem))

            time.sleep(self.sampling_rate)

        self.stats.append([round(sum(x) / len(x)) for x in zip(*current_sample)])

    def start_monitor(self):
        self.should_stop = False
        self.thread = Thread(target=self._monitor)
        self.thread.start()

    def stop_monitor(self):
        self.should_stop = True
        self.thread.join()

    def get_stats(self):
        return self.stats