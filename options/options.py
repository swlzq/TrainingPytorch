# author: LiuZhQ
# time  : 2019/7/19

import os
import time

__all__ = ['Options']


class Options(object):
    def __init__(self):
        self.__init_directory_path()
        self.__init_model_setting()
        self.__init_optimizer_setting()
        self.__init_dataset_setting()
        self.__init_interval_setting()

        # Each experiment has different flag to identity
        _date = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
        self.flag = '_' + _date + '_'
        self.result_path = os.path.join(self.root_path, self.result_path, self.model_name + self.flag)

        assert not os.path.exists(self.result_path), '{} has existed.'.format(self.result_path)
        os.makedirs(self.result_path)

    # Get all options' dictionary
    def get_all_opts(self):
        return vars(self)

    def __init_directory_path(self):
        self.root_path = './../'  # Project root directory path
        self.data_path = './'  # Dataset root directory path
        self.result_path = 'results'  # Result root directory path
        self.pretrained_model_path = 'pretrained_models'  # Path to store previous trained models
        self.resume_path = ''  # Checkpoint path to resume training

    def __init_model_setting(self):
        self.model_name = 'resnet'  # Model name
        self.input_size = 224  # Model's input size
        self.num_classes = 10  # For classification task
        self.use_cuda = True  # Use cuda or cpu
        self.fine_tune = True  # Fine tuning or train from scratch

    def __init_optimizer_setting(self):
        self.lr = 0.001  # Number of learning rate
        self.step_size = 20  # Number of lr change interval
        self.gamma = 0.1  # Number of lr multiple coefficient
        self.momentum = 0.9  # Number of momentum
        self.weight_decay = 5e-4  # Number of half l2 regularization coefficient
        self.patient = 10  # Number of lr change patient

    def __init_dataset_setting(self):
        self.batch_size = 128  # Number of batch size
        self.num_workers = 4  # Number of threading

    def __init_interval_setting(self):
        self.begin_epoch = 1  # Number of beginning training epoch
        self.epochs = 20  # Number of total training epochs
        self.log_interval = 20  # Number of logging training status interval
        self.checkpoint_interval = 5  # Number of save checkpoint interval


if __name__ == '__main__':
    opts = Options()
    opts.get_all_opts()
