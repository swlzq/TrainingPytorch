# author: LiuZhQ
# time  : 2019/7/29

from importlib import import_module
from torch.utils.data import DataLoader


class Data(object):
    def __init__(self, opts):
        if not opts.test_only:
            print('==> Load train dataset: {}'.format(opts.train_dataset))
            module_train = import_module('dataset.' + opts.train_dataset.lower())
            train_dataset = getattr(module_train, opts.train_dataset)(opts, train=True)
            self.train_loder = DataLoader(
                dataset=train_dataset,
                batch_size=opts.batch_size,
                num_workers=opts.num_workers,
                shuffle=True,
                pin_memory=opts.pin_memory
            )
        print('==> Load test dataset: {}'.format(opts.test_dataset))
        module_test = import_module('dataset.' + opts.test_dataset.lower())
        test_dataset = getattr(module_test, opts.test_dataset)(opts, train=False)
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            shuffle=False,
            pin_memory=opts.pin_memory
        )

    def get_train_loader(self):
        return self.train_loder

    def get_test_loader(self):
        return self.test_loader
