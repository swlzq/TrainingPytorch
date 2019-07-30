# author: LiuZhQ
# time  : 2019/7/19

import os
import time

__all__ = ['Option']


class Option(object):
    def __init__(self):
        # ==================== Initialize directory path ====================
        self.root_path = 'F:/project_python/TrainingPytorch'  # Project root directory path
        self.data_path = 'F:/dataset/state-farm-distracted-driver-detection'  # Dataset root directory path
        self.result_path = 'result'  # Result root directory path
        self.pretrained_model_path = 'pretrained_model'  # Path to store previous trained models
        self.resume_path = ''  # Checkpoint path to resume training
        self.test_path = ''  # Test model path
        # ==================== Initialize directory path ====================

        # ==================== Initialize model setting ====================
        self.model_name = 'resnet18'  # Model name
        self.input_size = 224  # Model's input size
        self.num_classes = 10  # For classification task
        self.use_cuda = True  # Use cuda or cpu
        # ==================== Initialize model setting ====================

        # ==================== Initialize optimizer setting ====================
        self.lr = 0.001  # Number of learning rate
        self.step_size = 20  # Number of lr change interval
        self.gamma = 0.1  # Number of lr multiple coefficient
        self.momentum = 0.9  # Number of momentum
        self.weight_decay = 5e-4  # Number of half l2 regularization coefficient
        self.patient = 10  # Number of lr change patient
        # ==================== Initialize optimizer setting ====================

        # ==================== Initialize dataset setting ====================
        self.train_dataset = 'DRIVER'  # Dataset name for training
        self.test_dataset = 'DRIVER'  # Dataset name for testing
        self.batch_size = 128  # Number of batch size
        self.num_workers = 2  # Number of threading
        self.pin_memory = True  # If open pin memory
        # ==================== Initialize dataset setting ====================

        # ==================== Initialize training setting ====================
        self.fine_tune = False  # Fine tuning or train from scratch
        self.test_only = False  # If true, not training
        self.begin_epoch = 0  # Number of beginning training epoch
        self.epochs = 20  # Number of total training epochs
        self.log_interval = 20  # Number of logging training status interval
        self.checkpoint_interval = 5  # Number of save checkpoint interval
        # ==================== Initialize training setting ====================

        self._init_file_settting()

    def _init_file_settting(self):
        # Each experiment has different flag to identity
        _date = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
        self.flag = '_' + _date + '_'
        self.result_path = os.path.join(self.root_path, self.result_path, self.model_name + self.flag)

        assert not os.path.exists(self.result_path), '{} has existed.'.format(self.result_path)
        os.makedirs(self.result_path)

        # Create pretrained model directory
        self.pretrained_model_path = os.path.join(self.root_path, self.pretrained_model_path)
        if not os.path.exists(self.pretrained_model_path):
            os.makedirs(self.pretrained_model_path)
