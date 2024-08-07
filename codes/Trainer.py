import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

'''
loader_dict:    {
    train:      dataloader                      ->  the dataloader of train datasets
    test:       dataloader                      ->  the dataloader of test datasets
}

loader_dict:    {
    model:      ModelObj                        ->  the modelObj of training 
    optim:      optimObj                        ->  the optimObj of training
    loss:       lossObj                         ->  the lossObj of training
    epochs:     int                             ->  the epoch number of training
    checkpoint: int                             ->  the checkpoint for loss/batch update
}

batch_size:     int                             ->  the batch_size for training
'''


class EMA:
    def __init__(self, history=24, bias=False, path='mtyProject/sscqxhdl_prediction/Models/'):
        self.model = 0
        self.decay = 1 - (1 / history)
        self.bias = bias
        self.path = path

        self.counter = 0
        self.shadow = {}
        self.backup = {}

    def register(self, model):
        self.model = model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                current_v = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data.detach().clone()

                if self.bias:
                    current_v /= 1 - self.decay ** self.counter
                    self.counter += 1

                self.shadow[name] = current_v.detach().clone()

    def apply(self, device):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()

                temp = self.shadow[name]
                temp = temp.to(device=torch.device(device))
                param.data = temp.detach().clone()

    def revert(self, device):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup

                temp = self.backup[name]
                temp = temp.to(device=torch.device(device))
                param.data = temp.clone()
        self.backup = {}

    def store(self):
        torch.save(self.shadow, self.path + 'New_Model_EMA_shadow.pth')
        torch.save(self.backup, self.path + 'New_Model_EMA_backup.pth')

    def load(self):
        self.shadow = torch.load(self.path + 'New_Model_EMA_shadow.pth')
        self.backup = torch.load(self.path + 'New_Model_EMA_backup.pth')


class Trainer:
    def __init__(self, train_dict, loader, EMA_config, mode='none', path='fd_prediction/', name='New_Model.params'):

        self.modelFunct = train_dict['model']
        self.optimizer = train_dict['optim']
        self.lossFunct = train_dict['loss']
        self.epochs = train_dict['epochs']

        self.train_loader = loader
        self.EMA = EMA(history=EMA_config[0], bias=EMA_config[1]) if EMA_config else False
        self.mode = mode
        self.path = path
        self.name = name

        class Printer:
            def __init__(self, epochs):
                self.epochs = epochs

                self.acc = 0
                self.acc_inited = False
                self.space = len(str(epochs))

                self.loss_accumulator = 0
                self.case_accumulator = 0
                self.loss_collector = []

            def _init_acc(self, loss_sample):
                self.acc = len(str(f'{loss_sample.item():.3f}')) + 1

            def update(self, loss_sample, data_sample):
                if not self.acc_inited:
                    self.acc_inited = True
                    self._init_acc(loss_sample)
                self.loss_accumulator += loss_sample.item()
                self.case_accumulator += len(data_sample)

            def summarize(self, epoch):
                print(f'[Epoch]: {epoch + 1:{self.space}d}/{self.epochs} | ', end='')
                print(f'[Loss]: {self.loss_accumulator / self.case_accumulator:{self.acc}.3f}')
                self.loss_collector.append(self.loss_accumulator / self.case_accumulator)

                self.loss_accumulator = 0
                self.case_accumulator = 0

            def get_loss(self):
                return self.loss_collector

        self.printer = Printer(self.epochs)

    def _device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _construct_optimizer(self):
        update_params = filter(lambda p: p.requires_grad, self.modelFunct.parameters())
        optimizer_type = self.optimizer[0]
        lr = self.optimizer[1]
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(update_params, lr=lr)
        elif optimizer_type.lower() == 'adagrad':
            self.optimizer = torch.optim.Adagrad(update_params, lr=lr)
        else:
            self.optimizer = torch.optim.Adagrad(update_params, lr=lr)

    def train(self):
        self.modelFunct = self.modelFunct.to(self._device())
        if self.EMA:
            self.EMA.register(self.modelFunct)
            if os.path.isfile(self.EMA.path + 'New_Model_EMA_shadow.pth'):
                self.EMA.load()
        self.modelFunct.train()
        self._construct_optimizer()

        for epoch in range(self.epochs):

            for batch_idx, (x, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                label = label.to(self._device())

                if type(x) is tuple or type(x) is list:
                    x = map(lambda ele: ele.to(self._device()), x)
                    loss = self.lossFunct(self.modelFunct(*x), label)
                else:
                    x = x.to(self._device())
                    loss = self.lossFunct(self.modelFunct(x), label)

                loss.backward()
                self.optimizer.step()
                self.printer.update(loss, label)

            if self.EMA: self.EMA.step()
            self.printer.summarize(epoch)

        loss_df = pd.Series(self.printer.get_loss(), index=np.arange(1, len(self.printer.get_loss()) + 1), name='Loss')
        loss_df.plot(kind='line',
                     figsize=(16, 9),
                     lw=0.9,
                     xlabel='Epochs (Batches)',
                     ylabel='Loss Values',
                     title='Training Summary',
                     color='red')
        plt.show()
        plt.savefig('mtyProject/loss/loss.png')
        plt.close()

        if self.EMA: self.EMA.apply('cuda')
        if self.EMA: self.EMA.store()

        torch.save(self.modelFunct.state_dict(), self.path + self.name)

        if self.mode == 'regression':
            self._test_on_test_regression(200)

        print('>>>[Trainer]: Done!')
        return self.modelFunct, self.path + self.name

    def _test_on_test_regression(self, clip=200):
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        if self.EMA: self.EMA.apply(device='cpu')
        self.modelFunct.eval()

        for idx, (data, label) in enumerate(self.train_loader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.modelFunct(*data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.modelFunct(data)

            if pred_label.shape[1] > 1:
                chip_p = pred_label.detach().squeeze().numpy()[:, -1]
                chip_l = label.detach().squeeze().numpy()[:, -1]
            else:
                chip_p = pred_label.detach().squeeze().numpy()
                chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(chip_p)
            label_collector.extend(chip_l)

            if clip >= 0 and len(label_collector) >= clip:
                break

        res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        res.plot(kind='line',
                 figsize=(16, 9),
                 alpha=0.8,
                 lw=0.8,
                 xlabel='Samples',
                 ylabel='Values',
                 title='Fit on Training Data',
                 color=('red', 'blue'))
        plt.show()
        plt.savefig('mtyProject/loss/train.png')
        plt.close()

        if self.EMA: self.EMA.revert(device='cpu')


class Trainer2F:
    def __init__(self, f1_config, f2_config, loader, EMA_config, mode='single', path='fd_prediction/',
                 name='New_Model.params'):

        self.F1_modelFunct = f1_config['model']
        self.F1_optimizer = f1_config['optim']
        self.F1_lossFunct = f1_config['loss']
        self.F1_epochs = f1_config['epochs']

        self.F2_modelFunct = f2_config['model']
        self.F2_optimizer = f2_config['optim']
        self.F2_lossFunct = f2_config['loss']
        self.F2_epochs = f2_config['epochs']

        self.loader = loader
        self.EMA_config = EMA_config
        self.mode = mode
        self.path = path
        self.name = name

    def train(self):
        if self.EMA_config:
            pre_EMA_file = [self.path + 'New_Model_EMA_shadow.pth', self.path + 'New_Model_EMA_backup.pth']
            if os.path.isfile(pre_EMA_file[0]):
                os.remove(pre_EMA_file[0])
            if os.path.isfile(pre_EMA_file[1]):
                os.remove(pre_EMA_file[1])

        trainer = Trainer({
            'model': self.F1_modelFunct,
            'loss': self.F1_lossFunct,
            'optim': self.F1_optimizer,
            'epochs': self.F1_epochs,
        },
            loader=self.loader,
            EMA_config=self.EMA_config,
            mode=self.mode,
            path=self.path,
            name=self.name)
        _, state_dict_path = trainer.train()
        print('>>>[Trainer]: Model Face1 Done!')

        self.F2_modelFunct.load_state_dict(torch.load(state_dict_path))
        for name, param in self.F2_modelFunct.named_parameters():
            if 'F1' in name:
                param.requires_grad = False

        trainer = Trainer({
            'model': self.F2_modelFunct,
            'loss': self.F2_lossFunct,
            'optim': self.F2_optimizer,
            'epochs': self.F2_epochs,
        },
            loader=self.loader,
            EMA_config=self.EMA_config,
            mode=self.mode,
            path=self.path,
            name=self.name)
        model_prototype, state_dict_path = trainer.train()
        print('>>>[Trainer]: Model Face2 Done!')

        return model_prototype, state_dict_path

