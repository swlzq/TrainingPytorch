# author: LiuZhQ
# time  : 2019/7/20

import os
import torch
import torch.nn as nn

import models.resnet as resnet
from models.MobileNetV2 import MobileNetV2

__all__ = ['Model']


class Model(object):
    def __init__(self, opts):
        self.model_name = opts.model_name
        self.num_classes = opts.num_classes
        self.fine_tune = opts.fine_tune
        self.pretrained_model_path = opts.pretrained_model_path if self.fine_tune else None

    def get_model(self):
        if self.fine_tune:
            print('==> Load pretrained models: {}'.format(self.model_name))
        else:
            print('==> Load models: {}'.format(self.model_name))

        if self.model_name.startswith('resnet'):
            model = self._resnet()
        elif self.model_name == 'MobileNetV2':
            model = self._mobilenetv2()
        else:
            assert False, '{} not exists.'.format(self.model_name)
        return model

    def _resnet(self):
        model = getattr(resnet, self.model_name)
        model = model(self.fine_tune, self.pretrained_model_path)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def _mobilenetv2(self):
        model = MobileNetV2()
        if self.fine_tune:
            path = os.path.join(self.pretrained_model_path, 'mobilenet_v2.pth.tar')
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
        model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        return model
