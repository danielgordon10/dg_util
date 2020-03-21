import itertools
import threading

import torch
import torch.multiprocessing as multiprocessing
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch._six import queue
from torch.utils.data import _utils, IterableDataset, Sampler, SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.dataloader import (
    DataLoader,
    _MultiProcessingDataLoaderIter,
    _DatasetKind,
    _BaseDataLoaderIter,
    _InfiniteConstantSampler,
)


# CHANGED PART
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
        never_ending=False,
    ):
        self.never_ending = never_ending
        if dataset is not None:
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
            if self.num_workers == 0:
                self.iterator = _SingleProcessDataLoaderIter(self, never_ending=self.never_ending)
            else:
                self.iterator = PersistentDataLoaderIter(self, device, never_ending=self.never_ending)
        else:
            # Assert that all non-used args are default
            assert batch_size == 1
            assert not shuffle
            assert sampler is None
            assert batch_sampler is None
            assert collate_fn is None
            assert drop_last is False
            assert worker_init_fn is None

            self.dataset = None
            self._dataset_kind = None
            self._IterableDataset_len_called = None
            self.sampler = None
            self.batch_sampler = BatchSampler(SequentialSampler(list(range(1000))), 1, False)
            self.drop_last = False
            self.collate_fn = None
            self.worker_init_fn = None

            self.setup(num_workers, pin_memory, timeout, multiprocessing_context)
            if self.num_workers == 0:
                self.iterator = None
            else:
                self.iterator = PersistentDataLoaderIter(self, device, never_ending=self.never_ending)

    def setup(self, num_workers=0, pin_memory=False, timeout=0, multiprocessing_context=None):

        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; " "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.multiprocessing_context = multiprocessing_context

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        self.__initialized = True

    # CHANGED PART
    def set_dataset(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        drop_last=False,
        worker_init_fn=None,
    ):
        torch._C._log_api_usage_once("python.data_loader")

        self.dataset = dataset
        self.worker_init_fn = worker_init_fn

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and `IterableDataset` ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to _workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific _workers.
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle)
                )
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler)
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler)
                )
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive " "with batch_size, shuffle, sampler, and " "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with "
                    "shuffle, sampler, and drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self._drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn

        if self.num_workers == 0:
            self.iterator = _SingleProcessDataLoaderIter(self, never_ending=self.never_ending)
        else:
            self.iterator.set_dataset(
                self._dataset_kind,
                dataset,
                self._index_sampler,
                collate_fn,
                drop_last,
                worker_init_fn,
                self._auto_collation,
            )

    def __iter__(self):
        return self.iterator


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader, never_ending=False):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0
        self.never_ending = never_ending

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last
        )

    def __next__(self):
        try:
            index = self._next_index()  # may raise StopIteration
        except StopIteration:
            self._sampler_iter = iter(self._index_sampler)
            if not self.never_ending:
                raise StopIteration
            else:
                index = self._next_index()  # may raise StopIteration

        try:
            data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        except StopIteration:
            self._dataset_fetcher = _DatasetKind.create_fetcher(
                self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last
            )
            if not self.never_ending:
                raise StopIteration
            else:
                data = self._dataset_fetcher.fetch(index)  # may raise StopIteration

        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

    next = __next__  # Python 2 compatibility


# END CHANGED PART


class PersistentDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader, device=None, never_ending=False):

        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        assert self._num_workers > 0
        self.never_ending = never_ending

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue(2 * self._num_workers)
        self._worker_pids_set = False
        self._shutdown = False
        self._send_idx = 0  # idx of the next task to be sent to _workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [_rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in _task_info.values() if len(v) == 1)
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        self._workers_status = []

        # CHANGED PART
        def delayed_worker_loop(dataset_queue, index_queue, data_queue, done_event, seed, worker_id, num_workers):
            (dataset_kind, dataset, auto_collation, collate_fn, drop_last, worker_init_fn) = dataset_queue.get()
            _utils.worker._worker_loop(
                dataset_kind,
                dataset,
                index_queue,
                data_queue,
                done_event,
                auto_collation,
                collate_fn,
                drop_last,
                seed,
                worker_init_fn,
                worker_id,
                num_workers,
            )

        self.dataset_queue = multiprocessing_context.Queue()

        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue(2)
            # index_queue.cancel_join_thread()
            if loader.dataset is None:
                w = multiprocessing_context.Process(
                    target=delayed_worker_loop,
                    args=(
                        self.dataset_queue,
                        index_queue,
                        self._worker_result_queue,
                        self._workers_done_event,
                        self._base_seed + i,
                        i,
                        self._num_workers,
                    ),
                )
                pass
            else:
                w = multiprocessing_context.Process(
                    target=_utils.worker._worker_loop,
                    args=(
                        self._dataset_kind,
                        self._dataset,
                        index_queue,
                        self._worker_result_queue,
                        self._workers_done_event,
                        self._auto_collation,
                        self._collate_fn,
                        self._drop_last,
                        self._base_seed + i,
                        self._worker_init_fn,
                        i,
                        self._num_workers,
                    ),
                )
            # END CHANGED PART
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._workers_status.append(True)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue(2 * self._num_workers)
            # CHANGED PART
            if device is None:
                device = torch.cuda.current_device()
            # END CHANGED PART
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue, device, self._pin_memory_thread_done_event),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to _workers (see comment above), we only register
            # _pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True

        # CHANGED PART
        if loader.dataset is not None:
            # prime the prefetch loop
            for _ in range(2 * self._num_workers):
                self._try_put_index()
        # END CHANGED PART

    # CHANGED PART
    def set_dataset(self, dataset_kind, dataset, index_sampler, collate_fn, drop_last, worker_init_fn, auto_collation):
        self._index_sampler = index_sampler
        self._sampler_iter = iter(self._index_sampler)
        for _ in range(self._num_workers):
            self.dataset_queue.put((dataset_kind, dataset, auto_collation, collate_fn, drop_last, worker_init_fn))
        # prime the prefetch loop
        for _ in range(2 * self._num_workers):
            self._try_put_index()

    # END CHANGED PART

    def __next__(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                # CHANGED PART
                self._sampler_iter = iter(self._index_sampler)
                # self._shutdown_workers()
                # Still indicate end of epoch
                print('ended dataset')
                if not self.never_ending:
                    for _ in range(2 * self._num_workers):
                        self._try_put_index()
                    raise StopIteration

                # END CHANGED PART

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < 2 * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            # CHANGED PART
            if self.never_ending:
                self._sampler_iter = iter(self._index_sampler)
                index = self._next_index()
            else:
                return
            # END CHANGED PART
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1