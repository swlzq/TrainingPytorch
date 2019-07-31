# author: LiuZhQ
# time  : 2019/7/30

import argparse
import time
import os
import torch
import random
import numpy as np


class Arguments(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Optional parameters')
        # ==================== Initialize directory path ====================
        parser.add_argument('--root_path', default='F:/project_python/TrainingPytorch',
                            type=str, help='Project root directory path')
        parser.add_argument('--data_path', default='F:/dataset/state-farm-distracted-driver-detection',
                            type=str, help='Dataset root directory path')
        parser.add_argument('--result_path', default='results',
                            type=str, help='Result root directory path')
        parser.add_argument('--pretrained_model_path', default='pretrained_models',
                            type=str, help='Path to store previous trained models')
        parser.add_argument('--resume_path', default='',
                            type=str, help='Checkpoint path to resume training')
        parser.add_argument('--test_path', default='',
                            type=str, help='Test models path')
        # ==================== Initialize directory path ====================

        # ==================== Initialize models setting ====================
        parser.add_argument('--model_name', default='resnet18',
                            type=str, help='Model name')
        parser.add_argument('--input_size', default=224,
                            type=int, help='Input size of models')
        parser.add_argument('--num_classes', default=10,
                            type=int, help='For classification task')
        parser.add_argument('--use_cuda', action='store_true',
                            help='If False use cpu else cuda')
        # ==================== Initialize models setting ====================

        # ==================== Initialize optimizer setting ====================
        parser.add_argument('--lr', default=0.001,
                            type=float, help='Number of learning rate')
        parser.add_argument('--step_size', default=20,
                            type=int, help='Number of lr change interval')
        parser.add_argument('--gamma', default=0.1,
                            type=float, help='Number of lr multiple coefficient')
        parser.add_argument('--momentum', default=0.9,
                            type=float, help='Number of momentum')
        parser.add_argument('--weight_decay', default=5e-4,
                            type=float, help='Number of half l2 regularization coefficient')
        parser.add_argument('--patient', default=10,
                            type=int, help='Number of lr change patient')
        # ==================== Initialize optimizer setting ====================

        # ==================== Initialize dataset setting ====================
        parser.add_argument('--train_dataset', default='DRIVER',
                            type=str, help='Dataset name for training')
        parser.add_argument('--test_dataset', default='DRIVER',
                            type=str, help='Dataset name for testing')
        parser.add_argument('--batch_size', default=128,
                            type=int, help='Number of batch size')
        parser.add_argument('--num_workers', default=4,
                            type=int, help='Number of threading')
        parser.add_argument('--pin_memory', action='store_true',
                            help='If False not use pin memory')
        # ==================== Initialize dataset setting ====================

        # ==================== Initialize training setting ====================
        parser.add_argument('--seed', default=7777,
                            type=int, help='Ensure training can be reproduced')
        parser.add_argument('--fine_tune', action='store_true',
                            help='If False not fine tune')
        parser.add_argument('--test_only', action='store_true',
                            help='If true, not training')
        parser.add_argument('--begin_epoch', default=0,
                            type=int, help='Number of beginning training epoch')
        parser.add_argument('--epochs', default=20,
                            type=int, help='Number of total training epochs')
        parser.add_argument('--log_interval', default=20,
                            type=int, help='Number of logging training status interval')
        parser.add_argument('--checkpoint_interval', default=5,
                            type=int, help='Number of save checkpoints interval')
        # ==================== Initialize training setting ====================
        self.args = parser.parse_args()

        self._init_file_setting()
        self._set_seed()

    def _init_file_setting(self):
        # Each experiment has different flag to identity
        _date = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
        flag = '_' + _date + '_'
        self.args.result_path = os.path.join(self.args.root_path,
                                             self.args.result_path, self.args.model_name + flag)
        assert not os.path.exists(self.args.result_path), '{} has existed.'.format(self.args.result_path)
        # Create pretrained models directory
        self.args.pretrained_model_path = os.path.join(self.args.root_path, self.args.pretrained_model_path)
        if not os.path.exists(self.args.pretrained_model_path):
            os.makedirs(self.args.pretrained_model_path)

    def _set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def get_args(self):
        return self.args
