# author: LiuZhQ
# time  : 2019/7/20

import os
import shutil
import pandas as pd

__all__ = ['Recorder']


class Recorder(object):
    def __init__(self, root_path, result_path):
        self.root_path = root_path  # project's root path
        self.result_path = result_path  # experiment result's root path
        self.code_path = os.path.join(self.result_path, 'code')  # saved code path
        self.opt_file = os.path.join(self.code_path, 'opt.txt')  # saved option argument path
        self.experiment_file = os.path.join(self.result_path, 'experiment.csv')  # save training results

        # File or directory list to record
        self.record_file_list = ['model', 'option']

        if not os.path.exists(self.code_path):
            os.makedirs(self.code_path)

    # record needed files
    def record_file(self):
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
        with open(self.opt_file, 'w') as f:
            for k, v in opts.items():
                f.write(str(k) + ": " + str(v) + '\n')

    def log_csv(self, data):
        '''
        :param data: training data, eg. [{acc:0, loss:0}]
        '''
        #
        assert isinstance(data, list), 'data must be a list'
        df = pd.DataFrame(data)
        if not os.path.isfile(self.experiment_file):
            df.to_csv(self.experiment_file, index=False)
        else:
            df.to_csv(self.experiment_file, mode='a', header=False, index=False)

    def log_txt(self, data, log_name):
        '''

        :param data: log data
        :param log_name: log file's name
        '''
        #
        name = log_name + '.txt'
        path = os.path.join(self.result_path, name)
        open_mode = 'w' if not os.path.isfile(path) else 'a'
        with open(path, open_mode) as f:
            f.write(data + '\n')


if __name__ == '__main__':
    recorder = Recorder(root_path='../', result_path='../results')
    recorder.record_file()
    recorder.log_csv([{'acc': 0., 'loss': 0.}])
    recorder.log_csv([{'acc': 0., 'loss': 0.}])
    recorder.log_txt('test', 'log')
    recorder.log_txt('test2', 'log')
