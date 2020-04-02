from mpi4py import MPI
import time
import logging
from os.path import abspath

level_map = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'error': logging.ERROR,
}

base_prefix = "{ptype} {world}:{parent}:{process}"
file_handler = None
stream_handler = None

start_time = time.time()


class ElapsedTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        global start_time
        total_millis = int(record.created - start_time) * 1000 + int(record.msecs)

        millis = total_millis % 1000
        millis = int(millis)
        seconds = (total_millis / 1000) % 60
        seconds = int(seconds)
        minutes = (total_millis / (1000 * 60)) % 60
        minutes = int(minutes)
        hours = (total_millis / (1000 * 60 * 60)) % 24
        hours = int(hours)

        elapsed = "{:04d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, millis)
        return elapsed


class MPIFileHandler(logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD):
        self.baseFilename = abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm.Dup()
        if delay:
            # We don't open the stream, but we still need to call the
            # Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg + self.terminator).encode(self.encoding))
            # self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None


def initialize_logger(filename=None, file_level='info', stream=True, stream_level='info'):
    global file_handler
    global stream_handler
    level = get_log_level(file_level)
    logger = logging.getLogger()
    logger.setLevel(level)
    if filename is not None:
        file_handler = MPIFileHandler(filename)
        logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
    set_logging_prefix(MPI.COMM_WORLD.rank)


def get_log_level(levelstr='info'):
    levelstr = levelstr.lower()
    return level_map[levelstr]


def set_logging_prefix(world_rank, parent_rank='-', process_rank='-', process_type='P'):
    global file_handler
    global stream_handler
    prefix = base_prefix.format(ptype=process_type, world=world_rank, parent=parent_rank, process=process_rank)
    formatter = ElapsedTimeFormatter('%(asctime)s ' + prefix + ' [%(levelname)s] %(message)s')
    if file_handler is not None:
        file_handler.setFormatter(formatter)
    if stream_handler is not None:
        stream_handler.setFormatter(formatter)


def get_logger():
    return logging.getLogger()
