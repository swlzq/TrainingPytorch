# author: LiuZhQ
# time  : 2019/7/19

import os
import torch
import torch.nn as nn

__all__ = ['Checkpoint']


class Checkpoint(object):
    def __init__(self, result_path='./'):
        '''
        Class to save/load checkpoint/state_dict.
        :param result_path: Path to save each experiment model、state_dict、parameters etc.
        '''
        self.model_path = os.path.join(result_path, 'model')
        self.checkpoint_path = os.path.join(result_path, 'checkpoint')
        self.checkpoint_params = {'model': None, 'optimizer': None, 'epoch': None, 'best_result': None}

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def save_checkpoint(self, model, optimizer, epoch, best_result):
        '''
        :param model: Network model.
        :param optimizer: Optimizer.
        :param epoch: Current epoch.
        :param best_result: Current lowest loss or highest accuracy etc.
        :return:
        '''
        if isinstance(model, nn.DataParallel):
            model = model.module

        self.checkpoint_params['model'] = model.state_dict()
        self.checkpoint_params['optimizer'] = optimizer.state_dict()
        self.checkpoint_params['epoch'] = epoch
        self.checkpoint_params['best_result'] = best_result

        torch.save(self.checkpoint_params,
                   os.path.join(self.checkpoint_path, 'checkpoint_{:03d}.pth'.format(epoch)))

    def save_state_dict(self, model, save_name='model', best_flag=False):
        '''
        :param model:
        :param save_name: Set saved model state_dict name.
        :param best_flag: Distinguish if state_dict is best or not.
        :return:
        '''
        if isinstance(model, nn.DataParallel):
            model = model.module
        state_dict = model.state_dict()

        save_name = 'best_{}.pth'.format(save_name) if best_flag else '{}.pth'.format(save_name)
        torch.save(state_dict, os.path.join(self.model_path, save_name))

    def load_checkpoint(self, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            print('==> Load checkpoint from {}'.format(checkpoint_file))
            self.checkpoint_params = torch.load(checkpoint_file)
            model_state_dict = self.checkpoint_params['model']
            optimizer_state_dict = self.checkpoint_params['optimizer']
            epoch = self.checkpoint_params['epoch']
            best_result = self.checkpoint_params['best_result']
            return model_state_dict, optimizer_state_dict, epoch, best_result
        else:
            assert False, 'File not exists: {}'.format(checkpoint_file)

    def load_state_dict(self, model_file, device='cuda', gpus=False):
        '''
        :param model_file:
        :param device: 'cuda' or 'cpu.
        :param gpus: If true, remove 'module.'.
        :return:
        '''
        if os.path.isfile(model_file):
            print('==> Load model state_dict from {}'.format(model_file))
            state_dict = torch.load(model_file, map_location=device)
            if gpus:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            return state_dict
        else:
            assert False, 'File not exists: {}'.format(model_file)
