# author: LiuZhQ
# time  : 2019/7/29

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from checkpoint import Checkpoint
from option import Option
from recorder import Recorder
from model import Model
from data import Data
from utils import AverageMeter, calculate_accuracy


class Trainer(object):
    def __init__(self):
        # Initialize object should be first
        self._init_object()
        self._init_device()
        self.recorder.record_file()  # Save needed codes
        self.recorder.write_opt(self.opts.get_all_opts())  # Save experiment arguments

        # ## Network ## Criterion ## Optimizer ## DataLoader ##
        self.net = self.model.get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.opts.lr, momentum=self.opts.momentum,
                                   weight_decay=self.opts.weight_decay)
        self.train_loader = self.data.get_train_loader()
        self.test_loader = self.data.get_test_loader()
        # ## Network ## Criterion ## Optimizer ## DataLoader ##

        # Resume training
        if self.opts.resume_path:
            self._resume()

        # LR scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts.step_size,
                                                   gamma=self.opts.gamma, last_epoch=self.opts.begin_epoch - 2)

        self.best_result = {'epoch': 1, 'accuracy': 0.}

    def _init_object(self):
        self.opts = Option()
        self.ckp = Checkpoint(self.opts.result_path)
        self.recorder = Recorder(self.opts.root_path, self.opts.result_path)
        self.model = Model(self.opts)
        self.data = Data(self.opts)

    def _init_device(self):
        if torch.cuda.is_available() and self.opts.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _resume(self):
        '''
        Load checkpoint to resume training.
        :return:
        '''
        model_state_dict, optimizer_state_dict, epoch, best_result = self.ckp.load_checkpoint(
            self.opts.resume_path)
        self.opts.begin_epoch = epoch + 1
        self.best_result = best_result
        self.net.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def train(self):
        '''
        Train network with train dataset.
        :return:
        '''
        epoch = self.scheduler.last_epoch + 1
        print('==> Start training epoch {} ...'.format(epoch))
        self.net.train()
        train_loss = AverageMeter()
        train_accuracy = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        start_batch_time = time.time()
        start_data_time = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            data_time.update(time.time() - start_data_time)
            
            # Compute output and loss
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            # Record loss and accuracy
            accuracy = calculate_accuracy(outputs, labels, topk=(1,))
            train_loss.update(loss.item(), inputs.size(0))
            train_accuracy.update(accuracy[0].item(), inputs.size(0))

            # Compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Measure elapsed time
            batch_time.update(time.time() - start_batch_time)
            start_batch_time = time.time()
            if (i + 1) % self.opts.log_interval == 0:
                print('Train Epoch: [{}/{}]([{}/{}])\t'
                      'Loss: {:.4f}({:.4f})\t'
                      'Accuracy: {:.4f}({:.4f})\t'
                      'LR: {}\t'
                      'Batch Time: {:.3f}({:.3f})\t'
                      'Data Time: {:.3f}({:.3f})'.format(
                    epoch, self.opts.epochs, i + 1, len(self.train_loader),
                    train_loss.val, train_loss.avg,
                    train_accuracy.val, train_accuracy.avg,
                    self.optimizer.param_groups[0]['lr'],
                    batch_time.val, batch_time.avg,
                    data_time.val, data_time.sum
                ))
            start_data_time = time.time()
        return train_accuracy.val, train_loss.val

    def test(self):
        '''
        Test or validate result with test dataset.
        :return:
        '''
        epoch = self.scheduler.last_epoch + 1
        print('==> Start Validatiing epoch {} ...'.format(epoch))
        self.net.eval()
        val_loss = AverageMeter()
        val_accuracy = AverageMeter()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(self.test_loader, ncols=80)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                # Record loss and accuracy
                accuracy = calculate_accuracy(outputs, labels, topk=(1,))
                val_loss.update(loss.item(), inputs.size(0))
                val_accuracy.update(accuracy[0].item(), inputs.size(0))
            print('Val Epoch: [{}/{}]\t'
                  'Loss: {:.4f}\t'
                  'Accuracy: {:.4f}(Best accuracy: {:.4f})'.format(
                epoch, self.opts.epochs,
                val_loss.avg,
                val_accuracy.avg, self.best_result['accuracy']
            ))

        return val_accuracy.avg, val_loss.avg

    def run(self):
        '''
        Enable trainer.
        :return:
        '''
        total_time = time.time()
        if self.opts.test_only and self.opts.test_path != '':
            state_dict = self.ckp.load_state_dict(self.opts.test_path)
            self.net.load_state_dict(state_dict)
            self.test()
        else:
            for epoch in range(self.opts.begin_epoch, self.opts.epochs + 1):
                epoch_time = time.time()
                self.scheduler.step()
                train_accuracy, train_loss = self.train()
                test_accuracy, test_loss = self.test()
                save_text = [{'epoch': epoch,
                              'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
                              'train_loss': train_loss, 'test_loss': test_loss}]
                self.recorder.log_csv(save_text)
                if test_accuracy >= self.best_result['accuracy']:
                    print('==> save best result ...')
                    self.ckp.save_state_dict(self.net, self.opts.model_name, best_flag=True)
                    self.best_result['epoch'] = epoch
                    self.best_result['accuracy'] = test_accuracy
                if epoch % self.opts.checkpoint_interval == 0:
                    self.ckp.save_checkpoint(self.net, self.optimizer, epoch, self.best_result)
                epoch_time = time.time() - epoch_time
                print('==> Epoch{} cost {}m {}s'.format(epoch, epoch_time // 60, epoch % 60))
        total_time = time.time() - total_time
        hour = total_time // 3600
        min = (total_time % 3600) // 60
        sec = total_time % 60
        print('==> Total training time is {}h:{}m:{}s'.format(hour, min, sec))