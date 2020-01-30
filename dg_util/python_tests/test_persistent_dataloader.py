import pytest
import numpy as np
import torch
from dg_util.python_utils import pytorch_util as pt_util
from torch.utils.data import Dataset
from dg_util.python_utils.persistent_dataloader import PersistentDataLoader
import pdb
from torch.utils.data._utils.collate import default_collate

class DummyDataset(Dataset):
    def __init__(self, data_size):
        super(DummyDataset, self).__init__()
        self.data_size = data_size
        self.data = list(range(data_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def worker_init_fn(worker_id):
        self = torch.utils.data.get_worker_info().dataset
        self.worker_id = worker_id
        self.seed = torch.initial_seed()
        print('worker_init', type(self).__name__, 'worker', self.worker_id, 'seed', self.seed)

    @staticmethod
    def collate_fn(batch):
        return pt_util.to_numpy(batch)


def _test_persistent_data_loader(
    data_size,
    batch_size,
    num_workers,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    collate_fn=None,
    drop_last=False,
    worker_init_fn=None,
    device=None,
    delayed_start=False,
):
    if not delayed_start:
        dataset = DummyDataset(data_size)
        data_loader = PersistentDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            device=device,
        )

    else:
        data_loader = PersistentDataLoader(dataset=None, num_workers=num_workers, pin_memory=True, device=device)

        dataset = DummyDataset(data_size)
        data_loader.set_dataset(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

    counts = 0
    num_data_points = 0
    for data in data_loader:
        counts += 1
        num_data_points += len(data)
        if collate_fn is not None:
            assert isinstance(data, np.ndarray)
            data = pt_util.from_numpy(data)

        if not drop_last and counts == len(data_loader) and len(dataset) % batch_size != 0:
            assert len(data) == len(dataset) % batch_size
        else:
            assert len(data) == batch_size
        if not shuffle:
            gt_data = torch.arange((counts - 1) * batch_size, counts * batch_size, dtype=torch.int64)
            assert torch.all(data == gt_data[:len(data)])

    if drop_last:
        assert num_data_points == len(dataset) // batch_size * batch_size
        assert counts == int(data_size / batch_size)
    else:
        assert num_data_points == len(dataset)
        assert counts == np.ceil(data_size * 1.0 / batch_size)

    # Make sure it can still run a second time
    counts = 0
    for data in data_loader:
        counts += 1
    assert counts > 0


def test_persistent_dataloader_normal():
    _test_persistent_data_loader(
        10, 6, 4, shuffle=True, delayed_start=False, drop_last=True,
    )

def test_persistent_dataloader_single_proc():
    _test_persistent_data_loader(
        10, 6, 0, shuffle=True, delayed_start=False, drop_last=True,
    )


def test_persistent_data_loader():
    tf = [True, False]
    for batch_size in range(1, 6):
        for drop_last in tf:
            for delayed_start in tf:
                for num_procs in [0, 1, 4]:
                    for shuffle in tf:
                        for wif in [None, DummyDataset.worker_init_fn]:
                            for cof in [None, DummyDataset.collate_fn]:
                                _test_persistent_data_loader(
                                    10, batch_size, num_procs, shuffle=shuffle, delayed_start=delayed_start, drop_last=drop_last,
                                    worker_init_fn=wif, collate_fn=cof,
                                )
