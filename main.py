# author: LiuZhQ
# time  : 2019/7/19

import random
import torch
import numpy as np

from trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(7777)
    trainer = Trainer()
    trainer.run()
