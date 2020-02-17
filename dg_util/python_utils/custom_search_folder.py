import glob
import pdb
import os
from collections import Counter

import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

try:
    import cPickle as pickle
except ImportError:
    import pickle


def default_name_function(path):
    return os.path.split(path)[0]


class CustomSearchFolder(Dataset):
    def __init__(
        self,
        base_folder,
        image_search_str,
        name_to_label_function=default_name_function,
        transform=None,
        target_transform=None,
        loader=default_loader,
        check_for_new_data=True,
    ):
        super(CustomSearchFolder, self).__init__()

        restored = False
        if not (check_for_new_data or not os.path.exists(os.path.join(base_folder, "paths.pkl"))):
            try:
                dataset = pickle.load(open(os.path.join(base_folder, "paths.pkl"), "rb"))
                self.files, self.labels, self.ind_to_label, self.label_to_ind, self.counter = dataset
                restored = True
            except:
                print("Failed to restore from file")
        if not restored:
            self.files = list(tqdm.tqdm(glob.iglob(os.path.join(base_folder, image_search_str))))
            self.files.sort()
            labels = [name_to_label_function(fi) for fi in self.files]
            label_set = list(set(labels))
            label_set.sort()
            label_to_ind = {label: ii for ii, label in enumerate(label_set)}
            self.labels = [label_to_ind[label] for label in labels]
            self.ind_to_label = label_set
            self.label_to_ind = label_to_ind
            self.counter = Counter()
            for label in self.labels:
                self.counter[label] += 1
            dataset = (self.files, self.labels, self.ind_to_label, self.label_to_ind, self.counter)
            pickle.dump(dataset, open(os.path.join(base_folder, "paths.pkl"), "wb"))

        print("loaded", len(self.files), "images with", len(self.ind_to_label), "classes")

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.files[index]
        target = self.labels[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.files)
