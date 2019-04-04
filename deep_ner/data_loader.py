import random
import numpy as np
from typing import Dict, Union, List, Tuple


class BaseDataLoader:

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader(BaseDataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0) Ignor Now
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        # TODO make multiprocessing
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.index_arr = [i for i in range(len(self.dataset))]
        self.internal_index = 0

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.index_arr)

        self.internal_index = 0
        return self

    def __next__(self):
        # TODO ask about output format
        if self.internal_index >= len(self.dataset):
            raise StopIteration

        X_batch = list()
        y_batch = list()
        batch_counter = 0
        while self.internal_index < len(self.dataset) and batch_counter < self.batch_size:
            index = self.index_arr[self.internal_index]
            if self.dataset.mode == 'train':
                x, y = self.dataset.__getitem__(index)
                X_batch.append(x)
                y_batch.append(y)
            else:
                x = self.dataset.__getitem__(index)
                X_batch.append(x)

            self.internal_index += 1
            batch_counter += 1

        # Add random samples for const batch size
        if batch_counter < self.batch_size:
            for _ in range(self.batch_size - batch_counter):
                index = random.choice(self.index_arr)

                if self.dataset.mode == 'train':
                    x, y = self.dataset.__getitem__(index)
                    X_batch.append(x)
                    y_batch.append(y)
                else:
                    x = self.dataset.__getitem__(index)
                    X_batch.append(x)

        if self.dataset.mode == 'train':
            X_batch, y_batch = self.transform_dataset(X_batch, y_batch)
            return X_batch, y_batch
        else:
            X_batch, y_batch = self.transform_dataset(X_batch, y_batch=None)
            return X_batch

    def __len__(self):
        return len(self.dataset)

    def transform_dataset(self, x_batch, y_batch):
        x_batch_tr = None
        for sample in x_batch:
            if x_batch_tr is None:
                x_batch_tr = [list() for _ in sample]
            for i_ch, channel in enumerate(sample):
                x_batch_tr[i_ch].append(channel)

        x_batch_tr = [np.stack(sample_ls) for sample_ls in x_batch_tr]
        y_batch_tr = np.vstack(y_batch) if y_batch is not None else None

        return x_batch_tr, y_batch_tr
