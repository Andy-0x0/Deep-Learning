import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    def __init__(self, train_dict, loader_dict, EMA_config, mode, path='fd_prediction/', name='New_Model.params'):

        self.modelFunct = train_dict['model']
        self.optimizer = train_dict['optim']
        self.lossFunct = train_dict['loss']
        self.epochs = train_dict['epochs']

        self.train_loader = loader_dict['train']
        self.test_loader = loader_dict['test']

        self.EMA = EMA(history=EMA_config[0], bias=EMA_config[1]) if EMA_config else False

        self.mode = mode['objective']
        self.indicator = mode['indicator']

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
                self.train_acc_collector = []
                self.test_acc_collector = []

            def _init_acc(self, loss_sample):
                self.acc = len(str(f'{loss_sample.item():.3f}')) + 1
                self.acc_inited = True

            def update(self, loss_sample, data_sample):
                if not self.acc_inited: self._init_acc(loss_sample)
                self.loss_accumulator += loss_sample.item()
                self.case_accumulator += len(data_sample)

            def summarize(self, epoch, precision_train=0, precision_test=0):
                print(f'[Epoch]: {epoch + 1:{self.space}d}/{self.epochs} | ', end='')
                print(f'[Loss]: {self.loss_accumulator / self.case_accumulator:{self.acc}.3f}', end='')
                self.loss_collector.append(self.loss_accumulator / self.case_accumulator)

                if precision_test or precision_train:
                    print(' -> ', end='')
                    if precision_train and not precision_test:
                        print(f"[Train {mode['indicator'].title()}]: {precision_train * 100:>6.2f}%", end='')
                        self.train_acc_collector.append(precision_train * 100)
                    elif precision_test and not precision_train:
                        print(f"[Test {mode['indicator'].title()}]: {precision_test * 100:>6.2f}%", end='')
                        self.test_acc_collector.append(precision_test * 100)
                    else:
                        print(f"[Train {mode['indicator'].title()}]: {precision_train * 100:>6.2f}% | ", end='')
                        print(f"[Test {mode['indicator'].title()}]: {precision_test * 100:>6.2f}%", end='')
                        self.train_acc_collector.append(precision_train)
                        self.test_acc_collector.append(precision_test)
                print()

                self.loss_accumulator = 0
                self.case_accumulator = 0

            def get_loss(self):
                return self.loss_collector

            def get_train_acc(self):
                return self.train_acc_collector

            def get_test_acc(self):
                return self.test_acc_collector

        self.printer = Printer(self.epochs)

    @staticmethod
    def _device():
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
            self.optimizer = torch.optim.SGD(update_params, lr=lr)

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
                    pred = self.modelFunct(*x)
                    loss = self.lossFunct(pred, label)
                else:
                    x = x.to(self._device())
                    pred = self.modelFunct(x)
                    loss = self.lossFunct(pred, label)

                loss.backward()
                self.optimizer.step()
                self.printer.update(loss, label)

            if self.EMA: self.EMA.step()

            train_acc = 0
            test_acc = 0
            if self.mode == 'regression':
                train_acc = self._on_test_regression(-1, viz=False, train=True)
                test_acc = self._on_test_regression(-1, viz=False, train=False)
            elif self.mode == 'classification':
                train_acc = self._on_test_classification(-1, viz=False, train=True)
                test_acc = self._on_test_classification(-1, viz=False, train=False)
            else:
                pass
            self.printer.summarize(epoch, train_acc, test_acc)

        loss_df = pd.DataFrame(self.printer.get_loss(), index=np.arange(1, len(self.printer.get_loss()) + 1), columns=['loss'])

        if self.EMA: self.EMA.apply('cuda')
        if self.EMA: self.EMA.store()

        torch.save(self.modelFunct.state_dict(), self.path + self.name)

        ax1 = None
        if self.mode == 'regression':
            plt.figure(figsize=(28, 12))
            plt.subplot(122)
            self._on_test_regression(200, viz=True, train=True)

            ax1 = plt.subplot(121)
            loss_df.loc[:, 'train_acc'] = self.printer.get_train_acc()
            loss_df.loc[:, 'test_acc'] = self.printer.get_test_acc()

        elif self.mode == 'classification':
            plt.figure(figsize=(28, 12))
            plt.subplot(122)
            self._on_test_classification(200, viz=True, train=True)

            ax1 = plt.subplot(121)
            loss_df.loc[:, 'train_acc'] = self.printer.get_train_acc()
            loss_df.loc[:, 'test_acc'] = self.printer.get_test_acc()

        else:
            plt.figure(figsize=(16, 9))
            pass

        loss_df.plot(y='loss',
                     kind='line',
                     lw=0.9,
                     ax=ax1,
                     color='red',
                     label='Loss')
        ax1.set_xlabel('Epochs (Batches)')
        ax1.set_ylabel('Loss Values')
        ax1.set_title('Training Summary')
        ax1.fill_between(np.arange(1, 1 + len(loss_df)),
                         loss_df['loss'].to_list(),
                         np.zeros(len(loss_df)), color='red',
                         alpha=0.2)
        ax1.legend(loc='upper left')

        if self.mode == 'classification' or self.mode == 'regression':
            ax2 = ax1.twinx()
            loss_df.plot(y=['train_acc', 'test_acc'],
                         kind='line',
                         lw=0.9,
                         ax=ax2,
                         color=['green', 'blue'],
                         label=['Train Accuracy', 'Test Accuracy'])
            ax2.set_ylabel('Accuracy')
            ax2.fill_between(np.arange(1, 1 + len(loss_df)),
                             loss_df['train_acc'].to_list(),
                             loss_df['test_acc'].to_list(), color='green',
                             alpha=0.1)
            ax2.fill_between(np.arange(1, 1 + len(loss_df)),
                             loss_df['test_acc'].to_list(),
                             np.zeros(len(loss_df)), color='blue',
                             alpha=0.1)
            ax2.legend(loc='upper right')

        plt.show()
        plt.close()

        print('>>>[Trainer]: Done!')
        return self.modelFunct, self.path + self.name

    def _on_test_regression(self, clip=200, viz=True, train=True):
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        if self.EMA: self.EMA.apply(device='cpu')
        self.modelFunct.eval()

        for idx, (data, label) in enumerate(self.train_loader if train else self.test_loader):
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

            if 0 <= clip <= len(label_collector):
                break

        res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        res.plot(kind='line',
                 alpha=0.8,
                 lw=0.8,
                 xlabel='Samples',
                 ylabel='Values',
                 title='Fit on Training Data',
                 color=('red', 'blue'))

        if self.EMA: self.EMA.revert(device='cpu')
        self.modelFunct.train()

        return 0

    def _on_test_classification(self, clip=200, viz=True, train=True):
        '''
        :param clip: int        -> The first clip number of samples to calculate the return value
        :return: float          -> The Precision Rate run on the testing dataset
        '''
        # Precision Works:
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        if self.EMA: self.EMA.apply(device='cpu')
        self.modelFunct.eval()

        for idx, (data, label) in enumerate(self.train_loader if train else self.test_loader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.modelFunct(*data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.modelFunct(data)

            chip_p = pred_label.detach().squeeze().numpy().argmax(axis=1)
            chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(chip_p)
            label_collector.extend(chip_l)

            if 0 <= clip <= len(label_collector):
                break

        if clip >= 0:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector}).iloc[:clip, :]
        else:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        indicator = -1
        if self.indicator.lower().startswith('acc'):
            indicator = accuracy_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        elif self.indicator.lower().startswith('f1'):
            indicator = f1_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        elif self.indicator.lower().startswith('prec'):
            indicator = precision_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        elif self.indicator.lower().startswith('rec'):
            indicator = recall_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        else:
            indicator = 0

        # Visualization Works:
        if viz:
            res.loc[:, 'Matching'] = (res.loc[:, 'Label'] == res.loc[:, 'Prediction']).astype(int)
            sq_length = int(np.floor(np.sqrt(res.shape[0])))
            match_map = np.zeros((sq_length, sq_length))
            for i in range(match_map.size):
                match_map[i // sq_length, i % sq_length] = res.loc[i, 'Matching']

            def heatmap(grid, cbar_kw=None, cbar_label=""):
                plt.title('Matching on Training Data')
                ax = plt.gca()

                if cbar_kw is None:
                    cbar_kw = {}

                cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#E8F5E9", "#1B5E20"])
                im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)

                cbar = ax.figure.colorbar(im, ax=ax, aspect=10, **cbar_kw)
                cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

                ax.set_xticks(np.arange(grid.shape[1]))
                ax.set_yticks(np.arange(grid.shape[0]))

                ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

                ax.spines[:].set_visible(False)

                ax.set_xticks(np.arange(grid.shape[1] + 1) - .5, minor=True)
                ax.set_yticks(np.arange(grid.shape[0] + 1) - .5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
                ax.tick_params(which="minor", bottom=False, left=False)

                return im, cbar

            def annotate_heatmap(im, grid=None, text_colors=("black", "white")):
                if grid is None:
                    grid = im.get_array().data

                threshold = 0.5
                kw = dict(horizontalalignment="center", verticalalignment="center")

                texts = []
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        kw.update(color=text_colors[int(grid[i, j] > threshold)])
                        text = im.axes.text(j, i, 'M' if grid[i, j] else 'F', **kw)
                        texts.append(text)

                return texts

            im, cbar = heatmap(match_map, cbar_label="Matching status")
            annotate_heatmap(im)

        if self.EMA: self.EMA.revert(device='cpu')
        self.modelFunct = self.modelFunct.to(self._device())
        self.modelFunct.train()

        return indicator


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
