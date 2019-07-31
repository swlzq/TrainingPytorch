# author: LiuZhQ
# time  : 2019/7/19

import os
import torch
import torch.nn as nn

__all__ = ['Checkpoint']


class Checkpoint(object):
    def __init__(self, result_path='./'):
        """
        Class to save/load checkpoints/state_dict.
        :param result_path: Path to save each experiment models、state_dict、parameters etc.
               -- checkpoints -- checkpoints.pth
              |
        result -- models -- state_dict.pth
              |
               -- code -- net
        """
        self.model_path = os.path.join(result_path, 'models')
        self.checkpoint_path = os.path.join(result_path, 'checkpoints')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, best_result):
        """
        :param model: Network models.
        :param optimizer: Optimizer.
        :param scheduler: Learning rate scheduler.
        :param epoch: Current epoch.
        :param best_result: Current lowest loss or highest accuracy etc.
        :return:
        """
        print('==> Save checkpoints ...')
        if isinstance(model, nn.DataParallel):
            model = model.module

        checkpoint_params = dict()
        checkpoint_params['models'] = model.state_dict()
        checkpoint_params['optimizer'] = optimizer.state_dict()
        checkpoint_params['scheduler'] = scheduler.state_dict()
        checkpoint_params['epoch'] = epoch
        checkpoint_params['best_result'] = best_result

        torch.save(checkpoint_params,
                   os.path.join(self.checkpoint_path, 'checkpoint_{:03d}.pth'.format(epoch + 1)))

    def save_state_dict(self, model, save_name='model', best_flag=False):
        """
        :param model:
        :param save_name: Set saved models state_dict name.
        :param best_flag: Distinguish if state_dict is best or not.
        :return:
        """
        if best_flag:
            print('==> Save best model ...')
        else:
            print('==> Save model ...')
        if isinstance(model, nn.DataParallel):
            model = model.module
        state_dict = model.state_dict()

        save_name = 'best_{}.pth'.format(save_name) if best_flag else '{}.pth'.format(save_name)
        torch.save(state_dict, os.path.join(self.model_path, save_name))

    def save_state_dict_from_checkpoint(self, checkpoint_file, save_name='model'):
        if os.path.isfile(checkpoint_file):
            print('==> Save model from checkpoint: {} ...'.format(checkpoint_file))
            checkpoint_params = torch.load(checkpoint_file)
            model_state_dict = checkpoint_params['model']
            save_name = save_name + '.pth'
            torch.save(model_state_dict, os.path.join(self.model_path, save_name))
        else:
            assert False, 'File not exists: {}'.format(checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            print('==> Load checkpoint from {}'.format(checkpoint_file))
            checkpoint_params = torch.load(checkpoint_file)
            model_state_dict = checkpoint_params['models']
            optimizer_state_dict = checkpoint_params['optimizer']
            scheduler_state_dict = checkpoint_params['scheduler']
            epoch = checkpoint_params['epoch']
            best_result = checkpoint_params['best_result']
            return model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, best_result
        else:
            assert False, 'File not exists: {}'.format(checkpoint_file)

    def load_state_dict(self, model_file, device='cuda', gpus=False):
        """
        :param model_file:
        :param device: 'cuda' or 'cpu.
        :param gpus: If true, remove 'module.'.
        :return:
        """
        if os.path.isfile(model_file):
            print('==> Load models state_dict from {}'.format(model_file))
            state_dict = torch.load(model_file, map_location=device)
            if gpus:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            return state_dict
        else:
            assert False, 'File not exists: {}'.format(model_file)
