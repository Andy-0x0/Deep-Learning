import datetime
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


class Trainer:
    def __init__(self, loader_dict, train_dict, mode='none', path='fd_prediction/', name='New_Model.params', display='E'):
        self.train_loader = loader_dict['train']
        self.test_loader = loader_dict['test']

        self.modelFunct = train_dict['model']
        self.optimizer = train_dict['optim']
        self.lossFunct = train_dict['loss']
        self.epochs = train_dict['epochs']
        self.checkpoint = train_dict['checkpoint']

        self.mode = mode
        self.path = path
        self.name = name
        self.display = display

    def _device_decide(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def train(self):
        loss_collector_train = []
        outer_loss = 0
        outer_scaler = 0
        acc_collector_test = []
        self.modelFunct = self.modelFunct.to(self._device_decide())
        self.modelFunct.train()

        for epoch in range(self.epochs):
            if self.display == 'B':
                print(f'[Epoch]: {epoch + 1}/{self.epochs} ================')

            for batch_idx, (x, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                label = label.to(self._device_decide())
                if type(x) is tuple or type(x) is list:
                    x = map(lambda ele: ele.to(self._device_decide()), x)
                    loss = self.lossFunct(self.modelFunct(* x), label)
                else:
                    x = x.to(self._device_decide())
                    loss = self.lossFunct(self.modelFunct(x), label)

                loss.backward()

                self.optimizer.step()

                if self.display == 'E':
                    outer_loss += loss.item()
                    outer_scaler += len(label)
                else:
                    if (batch_idx + 1) % self.checkpoint == 0:
                        print(f'\t[Batch]: {batch_idx + 1} | [Loss]: {loss.item():.3f}')
                        loss_collector_train.append(loss.item())

            if self.display == 'E':
                print(f'[Epoch]: {epoch + 1}/{self.epochs} | ', end='')
                print(f'[Loss]: {outer_loss / outer_scaler:.3f}')
                loss_collector_train.append(outer_loss / outer_scaler)
                outer_loss, outer_scaler = 0, 0
            else:
                print()

            if self.mode == 'classification':
                acc = self._test_on_test_classification()
                acc_collector_test.append(acc)

        loss_df = pd.Series(loss_collector_train, index=np.arange(1, len(loss_collector_train) + 1), name='Loss')
        loss_df.plot(kind='line',
                     figsize=(16, 9),
                     lw=0.9,
                     xlabel='Epochs (Batches)',
                     ylabel='Loss Values',
                     title='Training Summary',
                     color='red',
                     )
        plt.show()
        plt.close()
        torch.save(self.modelFunct.state_dict(), self.path + self.name)

        if self.mode == 'classification':
            plt.plot(np.linspace(1, len(loss_collector_train), self.epochs), acc_collector_test, color='blue',
                     label='Acc on Test')
            plt.show()
            plt.close()
        elif self.mode == 'regression':
            self._test_on_test_regression(200)

        print('>>>[Trainer]: Done!')
        return self.modelFunct, self.path + self.name

    def _test_on_test_classification(self):
        correct = 0
        total = 0
        for idx, (data, label) in enumerate(self.test_loader):
            data = data.to(device=self._device_decide())
            label = label.to(device=self._device_decide())
            pred_label = self.modelFunct(data)

            correct += (pred_label.argmax(dim=1) == label).sum(axis=0).to(device=torch.device('cpu'))
            total += label.numel()

        print(f'\t[Accuracy]  {100 * (correct / total):.2f}%')
        return correct / total

    def _test_on_test_regression(self, clip=200):
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        self.modelFunct.eval()

        for idx, (data, label) in enumerate(self.train_loader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.modelFunct(* data)
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

        if clip > 0:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})
            res = res.iloc[:min(clip, len(res)), :]
        else:
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
        plt.close()


class Trainer2F:
    def __init__(self, loader_config, f1_config, f2_config, mode='single', path='fd_prediction/', name='New_Model.params', display='E'):
        self.train_loader = loader_config['train']
        self.test_loader = loader_config['test']

        self.F1_modelFunct = f1_config['model']
        self.F1_optimizer = f1_config['optim']
        self.F1_lossFunct = f1_config['loss']
        self.F1_epochs = f1_config['epochs']
        self.F1_checkpoint = f1_config['checkpoint']

        self.F2_modelFunct = f2_config['model']
        self.F2_optimizer = f2_config['optim']
        self.F2_lossFunct = f2_config['loss']
        self.F2_epochs = f2_config['epochs']
        self.F2_checkpoint = f2_config['checkpoint']

        self.mode = mode
        self.path = path
        self.name = name
        self.display = display

    def train(self):
        update_params = self.F1_modelFunct.parameters()
        optimizer_type = self.F1_optimizer[0]
        lr = self.F1_optimizer[1]
        if optimizer_type.lower() == 'adam':
            self.F1_optimizer = torch.optim.Adam(update_params, lr=lr)
        elif optimizer_type.lower() == 'adagrad':
            self.F1_optimizer = torch.optim.Adagrad(update_params, lr=lr)
        else:
            self.F1_optimizer = torch.optim.Adagrad(update_params, lr=lr)

        trainer_f1 = Trainer({
            'train': self.train_loader,
            'test': self.test_loader
        },
        {
            'model': self.F1_modelFunct,
            'loss': self.F1_lossFunct,
            'optim': self.F1_optimizer,
            'epochs': self.F1_epochs,
            'checkpoint': self.F1_checkpoint,
        },
        mode=self.mode,
        path=self.path,
        name=self.name,
        display=self.display)
        _, state_dict_path = trainer_f1.train()
        print('>>>[Trainer]: Model Face1 Done!')

        self.F2_modelFunct.load_state_dict(torch.load(state_dict_path))
        for name, param in self.F2_modelFunct.named_parameters():
            if 'F1' in name:
                param.requires_grad = False

        update_params = filter(lambda p: p.requires_grad, self.F2_modelFunct.parameters())
        optimizer_type = self.F2_optimizer[0]
        lr = self.F2_optimizer[1]
        if optimizer_type.lower() == 'adam':
            self.F2_optimizer = torch.optim.Adam(update_params, lr=lr)
        elif optimizer_type.lower() == 'adagrad':
            self.F2_optimizer = torch.optim.Adagrad(update_params, lr=lr)
        else:
            self.F2_optimizer = torch.optim.Adagrad(update_params, lr=lr)

        trainer = Trainer({
            'train': self.train_loader,
            'test': self.test_loader
        },
        {
            'model': self.F2_modelFunct,
            'loss': self.F2_lossFunct,
            'optim': self.F2_optimizer,
            'epochs': self.F2_epochs,
            'checkpoint': self.F2_checkpoint,
        },
        mode=self.mode,
        path=self.path,
        name=self.name,
        display=self.display)
        model_prototype, state_dict_path = trainer.train()
        print('>>>[Trainer]: Model Face2 Done!')

        return model_prototype, state_dict_path






