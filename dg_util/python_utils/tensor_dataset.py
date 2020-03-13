from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, data, labels, batch_size):
        super(TensorDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.data_size = len(self.data)
        self.batch_size = batch_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        sample = {"data": self.data[idx], "label": self.labels[idx]}
        return sample

    def __iter__(self):
        return TensorDataIterator(self.data, self.labels, self.batch_size)


class TensorDataIterator(object):
    def __init__(self, data, labels, batch_size):
        super(TensorDataIterator, self).__init__()
        self.data = data
        self.labels = labels
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.batch_on = 0

    def __len__(self):
        return self.data_size // self.batch_size

    def __getitem__(self, idx):
        sample = {"data": self.data[idx], "label": self.labels[idx]}
        return sample

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_on >= self.data_size:
            raise StopIteration()
        else:
            ii = self.batch_on
            self.batch_on += self.batch_size
            data = self.data[ii : min(ii + self.batch_size, self.data_size)]
            label = self.labels[ii : min(ii + self.batch_size, self.data_size)]
            sample = {"data": data, "labels": label}
            return sample

    next = __next__
