# author: LiuZhQ
# time  : 2019/7/20

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Recorder']


class Recorder(object):
    def __init__(self, root_path, result_path):
        self.root_path = root_path  # project's root path
        self.result_path = result_path  # experiment result's root path
        self.code_path = os.path.join(self.result_path, 'code')  # saved code path
        self.opt_file = os.path.join(self.code_path, 'opt.txt')  # saved options argument path
        self.experiment_file = os.path.join(self.result_path, 'experiment.csv')  # save training results

        # File or directory list to record
        self.record_file_list = ['models', 'options']

        # Experiment dataset for plotting[[train_acc][test_acc][train_loss][test_acc]]
        self.log_data = [[] for _ in range(4)]
        self.log_label = ['train_acc', 'test_acc', 'train_loss', 'test_loss']

        if not os.path.exists(self.code_path):
            os.makedirs(self.code_path)

    # record needed files
    def record_file(self):
        print('==> Record needed files ...')
        for file in self.record_file_list:
            src_path = os.path.join(self.root_path, file)
            tag_path = os.path.join(self.code_path, file)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, tag_path)
            elif os.path.isfile(src_path):
                shutil.copyfile(src_path, tag_path)
            else:
                assert False, '{} not existed.'.format(file)

    def write_opt(self, opts):
        print('==> Save options ...')
        opts = vars(opts)
        with open(self.opt_file, 'w') as f:
            for k, v in opts.items():
                f.write(str(k) + ": " + str(v) + '\n')

    def log_csv(self, data):
        """
        :param data: training dataset, eg. [{acc:0, loss:0}]
        """
        #
        assert isinstance(data, list), 'dataset must be a list'
        df = pd.DataFrame(data)
        if not os.path.isfile(self.experiment_file):
            df.to_csv(self.experiment_file, index=False)
        else:
            df.to_csv(self.experiment_file, mode='a', header=False, index=False)

    def log_txt(self, data, log_name):
        """
        :param data: log dataset
        :param log_name: log file's name
        """
        #
        name = log_name + '.txt'
        path = os.path.join(self.result_path, name)
        open_mode = 'w' if not os.path.isfile(path) else 'a'
        with open(path, open_mode) as f:
            f.write(data + '\n')

    def add_log(self, data):
        for idx in range(len(self.log_data)):
            self.log_data[idx].append(data[idx])

    def plt_all(self):
        self.plt(self.log_data[:2], self.log_label[:2], 'epochs', 'accuracy')
        self.plt(self.log_data[2:], self.log_label[2:], 'epochs', 'loss')

    def plt(self, data, label, xlabel, ylabel):
        assert isinstance(data, list), 'dataset must be list'
        axis = np.arange(1, len(data[0]) + 1)
        fig = plt.figure()
        plt.title('Train and test {}'.format(ylabel))
        for idx, data in enumerate(data):
            plt.plot(
                axis,
                data,
                label=label[idx]
            )
        plt.legend(loc=1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        savefig_path = os.path.join(self.result_path, 'experiment_{}.png'.format(ylabel))
        plt.savefig(savefig_path)
        plt.close(fig)
