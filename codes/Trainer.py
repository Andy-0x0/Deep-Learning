import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Training Class (1 face) ==============================================================================================
class Trainer:
    def __init__(self,
        train_dict:dict = None,
        loader_dict:dict = None,
        mode_dict:dict = None,
        path:str = 'fd_prediction/New_Model.params'
    ) -> None:
        '''
        Initialization

        :param train_dict:
                - model: The model object inherent torch.nn.Module
                - optim: The tuple of representation of the optimizer and its parameters
                - loss: The loss object inherent torch.nn.Module | selected from torch.nn
                - epochs: The number of epochs for training
        :param loader_dict:
                - train: The training dataloader which is a instance of torch.DataLoader
                - test: The testing dataloader which is a instance of torch.DataLoader
        :param mode:
                - objective: The string representation of regression or classification or neither
                - indicator: The string representation of the evaluating indicator
        :param path: The path where the trained model will be located
        '''

        self.modelFunct = train_dict['model']
        self.optimizer = train_dict['optim']
        self.lossFunct = train_dict['loss']
        self.epochs = train_dict['epochs']

        self.train_loader = loader_dict['train']
        self.test_loader = loader_dict['test']

        self.mode = mode_dict['objective']
        self.indicator = mode_dict['indicator']

        self.path = path

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
                        print(f"[Train {mode_dict['indicator'].title()}]: {precision_train * 100:>6.2f}%", end='')
                        self.train_acc_collector.append(precision_train * 100)
                    elif precision_test and not precision_train:
                        print(f"[Test {mode_dict['indicator'].title()}]: {precision_test * 100:>6.2f}%", end='')
                        self.test_acc_collector.append(precision_test * 100)
                    else:
                        print(f"[Train {mode_dict['indicator'].title()}]: {precision_train * 100:>6.2f}% | ", end='')
                        print(f"[Test {mode_dict['indicator'].title()}]: {precision_test * 100:>6.2f}%", end='')
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

    def _on_test_regression(self, clip=200, viz=True, train=True):
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
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

        if clip >= 0:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector}).iloc[:clip, :]
        else:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        if self.indicator.lower().startswith('r'):
            indicator = r2_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        elif self.indicator.lower().startswith('l1') or self.indicator.lower().startswith('mse'):
            indicator = mean_absolute_error(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        elif self.indicator.lower().startswith('l2') or self.indicator.lower().startswith('mae'):
            indicator = mean_squared_error(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        else:
            indicator = r2_score(res.loc[:, 'Label'], res.loc[:, 'Prediction'])

        if viz:
            plt.title("Fitting on Training Data")
            plt.plot(np.arange(1, len(res) + 1), res.loc[:, 'Prediction'], label='Prediction', color='r', alpha=0.8, lw=0.8)
            plt.plot(np.arange(1, len(res) + 1), res.loc[:, 'Label'], label='Real', color='b', alpha=0.8, lw=0.8)
            plt.legend(loc='upper right')
            plt.xlabel('Samples')
            plt.ylabel('Values')

        # Restore the Environment when leaving
        self.modelFunct = self.modelFunct.to(device=self._device())
        self.modelFunct.train()
        return indicator

    def _on_test_classification(self, clip=200, viz=True, train=True):
        '''
        :param clip: int        -> The first clip number of samples to calculate the return value
        :return: float          -> The Precision Rate run on the testing dataset
        '''
        # Precision Works:
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
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

        self.modelFunct = self.modelFunct.to(self._device())
        self.modelFunct.train()

        return indicator

    def train(self):
        # 1. Set the model into "training on device mode"
        self.modelFunct = self.modelFunct.to(device=self._device())
        self.modelFunct.train()

        # 2. Set the optimizer
        self._construct_optimizer()

        # Training process starts -------------------------------------------
        for epoch in range(self.epochs):

            # 3.1 Training the model and do the update
            for batch_idx, (x, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                label = label.to(self._device())

                # Flexible unzip input x into the model and calculate the loss
                if type(x) is tuple or type(x) is list:
                    x = list(map(lambda ele: ele.to(self._device()), x))
                    pred = self.modelFunct(*x)
                    loss = self.lossFunct(pred, label)
                else:
                    x = x.to(self._device())
                    pred = self.modelFunct(x)
                    loss = self.lossFunct(pred, label)

                loss.backward()
                self.optimizer.step()
                self.printer.update(loss, label)

            # 3.2 Print the update per epoch
            if self.mode == 'regression':
                train_acc = self._on_test_regression(-1, viz=False, train=True)
                test_acc = self._on_test_regression(-1, viz=False, train=False)
                self.printer.summarize(epoch, train_acc, test_acc)
            elif self.mode == 'classification':
                train_acc = self._on_test_classification(-1, viz=False, train=True)
                test_acc = self._on_test_classification(-1, viz=False, train=False)
                self.printer.summarize(epoch, train_acc, test_acc)
            else:
                self.printer.summarize(epoch)
        # Training process ends -------------------------------------------

        # Visualization starts --------------------------------------------
        # 4. Plot the left
        def syn_plot(ax):
            if isinstance(ax, plt.Axes):
                # Plot the accuracy indicator which only "classification" and "Regression" have
                ax_copy = ax.twinx()
                loss_df.plot(y=['train_acc', 'test_acc'],
                             kind='line',
                             lw=0.9,
                             ax=ax_copy,
                             color=['green', 'blue'],
                             label=['Train Accuracy', 'Test Accuracy'])
                ax_copy.fill_between(np.arange(1, 1 + len(loss_df)),
                                     loss_df['train_acc'].to_list(),
                                     loss_df['test_acc'].to_list(),
                                     color='green',
                                     alpha=0.1)
                ax_copy.fill_between(np.arange(1, 1 + len(loss_df)),
                                     loss_df['test_acc'].to_list(),
                                     np.zeros(len(loss_df)),
                                     color='blue',
                                     alpha=0.1)
                ax_copy.set_ylabel('Accuracy')
                ax_copy.legend(loc='upper right')
            else:
                ax = plt
            # Plot the loss which all categories have
            loss_df.plot(y='loss', kind='line', lw=0.9, ax=ax, color='red', label='Loss')
            ax.fill_between(np.arange(1, 1 + len(loss_df)), loss_df['loss'].to_list(), np.zeros(len(loss_df)),
                            color='red', alpha=0.2)
            ax.legend(loc='upper left')
            ax.set_xlabel('Epochs (Batches)')
            ax.set_ylabel('Loss Values')
            ax.set_title('Training Summary')

        loss_df = pd.DataFrame(self.printer.get_loss(), index=np.arange(1, len(self.printer.get_loss()) + 1), columns=['loss'])

        if self.mode == 'regression':
            plt.figure(figsize=(28, 12))

            plt.subplot(122)
            self._on_test_regression(200, viz=True, train=True)

            ax1 = plt.subplot(121)
            loss_df.loc[:, 'train_acc'] = self.printer.get_train_acc()
            loss_df.loc[:, 'test_acc'] = self.printer.get_test_acc()
            syn_plot(ax1)

        elif self.mode == 'classification':
            plt.figure(figsize=(28, 12))
            plt.subplot(122)
            self._on_test_classification(200, viz=True, train=True)

            ax1 = plt.subplot(121)
            loss_df.loc[:, 'train_acc'] = self.printer.get_train_acc()
            loss_df.loc[:, 'test_acc'] = self.printer.get_test_acc()
            syn_plot(ax1)

        else:
            plt.figure(figsize=(16, 9))
            syn_plot(None)

        plt.show()
        plt.close()
        # Visualization ends --------------------------------------------

        torch.save(self.modelFunct.state_dict(), self.path)
        print('>>>[Trainer]: Done!')

        return self.modelFunct, self.path





