# author: LiuZhQ
# time  : 2019/7/20

import os
import shutil

__all__ = ['Recorder']


class Recorder(object):
    def __init__(self, root_path, result_path):
        self.root_path = root_path
        self.result_path = result_path
        self.code_path = os.path.join(self.result_path, 'code')
        self.opt_file = os.path.join(self.code_path, 'opt.txt')

        # File or directory list to record
        self.record_file_list = ['model', 'main.py']

        if not os.path.exists(self.code_path):
            os.makedirs(self.code_path)

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


if __name__ == '__main__':
    recorder = Recorder(root_path='../', result_path='../results')
    recorder.record_file()
