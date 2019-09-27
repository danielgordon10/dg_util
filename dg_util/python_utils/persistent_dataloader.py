import itertools
import threading

import torch
import torch.multiprocessing as multiprocessing
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch._six import queue
from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader, _MultiProcessingDataLoaderIter, _DatasetKind


class PersistentDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        device=None,
    ):
        super(PersistentDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
        )
        self.iterator = PersistentDataLoaderIter(self, device)

    def __iter__(self):
        return self.iterator


class PersistentDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader, device=None):




        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self.num_workers > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self.worker_init_fn = loader.worker_init_fn
        self.worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
        self.worker_result_queue = multiprocessing_context.Queue()
        self.worker_pids_set = False
        self.shutdown = False
        self.send_idx = 0  # idx of the next task to be sent to workers
        self.rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self.task_info = {}
        self.tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        self.workers_done_event = multiprocessing_context.Event()

        self.index_queues = []
        self.workers = []
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        self.workers_status = []
        for i in range(self.num_workers):
            index_queue = multiprocessing_context.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self.dataset_kind, self.dataset, index_queue,
                      self.worker_result_queue, self.workers_done_event,
                      self.auto_collation, self.collate_fn, self.drop_last,
                      self.base_seed + i, self.worker_init_fn, i, self.num_workers))
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self.workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self.index_queues.append(index_queue)
            self.workers.append(w)
            self.workers_status.append(True)

        if self.pin_memory:
            self.pin_memory_thread_done_event = threading.Event()
            self.data_queue = queue.Queue()
            ###### CHANGED PART
            if device is None:
                device = torch.cuda.current_device()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self.worker_result_queue, self.data_queue,
                      device,
                      self.pin_memory_thread_done_event))
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self.pin_memory_thread = pin_memory_thread
        else:
            self.data_queue = self.worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self.workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self.worker_pids_set = True

        # prime the prefetch loop
        for _ in range(2 * self.num_workers):
            self._try_put_index()

    def __next__(self):
        while True:
            # If the worker responsible for `self.rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self.rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self.rcvd_idx < self.send_idx:
                info = self.task_info[self.rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self.workers_status[worker_id]:  # has data or is still active
                    break
                del self.task_info[self.rcvd_idx]
                self.rcvd_idx += 1
            else:
                # no valid `self.rcvd_idx` is found (i.e., didn't break)
                ######## CHHANGED PART
                self.sampler_iter = iter(self.index_sampler)
                for _ in range(2 * self.num_workers):
                    self._try_put_index()
                #self._shutdown_workers()
                #raise StopIteration

            # Now `self.rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self.task_info[self.rcvd_idx]) == 2:
                data = self.task_info.pop(self.rcvd_idx)[1]
                return self._process_data(data)

            assert not self.shutdown and self.tasks_outstanding > 0
            idx, data = self._get_data()
            self.tasks_outstanding -= 1

            if self.dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.task_info[idx] += (data,)
            else:
                del self.task_info[idx]
                return self._process_data(data)
