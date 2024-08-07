import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
import matplotlib.pyplot as plt

import torch

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
    

class Evaluator:
    def __init__(self, model_path, model_ptototype, dataloader, EMA_config, reverse_para):
        self.state_dict_path = model_path
        self.model = model_ptototype

        self.dataloader = dataloader
        self.EMA = EMA(history=EMA_config[0], bias=EMA_config[1]) if EMA_config else False
        self.reverse_para = reverse_para

        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model = self.model.to(device=torch.device('cpu'))
        if self.EMA: self.EMA.load()
        self.model.eval()
        
    def evaluate(self):
        pred_collector = []
        label_collector = []
        
        if self.EMA: self.EMA.apply(device='cpu')
        
        def reverse_normalization(line, mean, std):
            return (line * std) + mean

        def score_line(pred_label_line, label_line):
            y_bar = label_line.mean(axis=0)
            y_hat = pred_label_line
            y = label_line

            tss = np.sum((y - y_bar) ** 2)
            rss = np.sum((y_hat - y) ** 2)
            ess = np.sum((y_hat - y_bar) ** 2)

            return ess / tss

        for idx, (data, label) in enumerate(self.dataloader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.model(* data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.model(data)

            if pred_label.shape[1] > 1:
                chip_p = pred_label.detach().squeeze().numpy()[:, -1]
                chip_l = label.detach().squeeze().numpy()[:, -1]
            else:
                chip_p = pred_label.detach().squeeze().numpy()
                chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(reverse_normalization(chip_p, self.reverse_para[0], self.reverse_para[1]))
            label_collector.extend(reverse_normalization(chip_l, self.reverse_para[0], self.reverse_para[1]))
        
        test_df = pd.DataFrame({'Prediction': pred_collector, 'Real': label_collector})
        r_square = score_line(test_df.loc[:, 'Prediction'].to_numpy(), test_df.loc[:, 'Real'].to_numpy())
        if self.EMA: self.EMA.revert(device='cpu')
        
        print(f'>>>[Evaluator]: Done!')
        
        return test_df, r_square


class Drawer:
    def __init__(self, path_config, title_config, model_prototype, dataloader, EMA_config, reverse_para, display=True):
        self.state_dict_path = path_config['model']
        self.result_path = path_config['result']

        self.title_list = title_config['title_list']
        self.sub_title_list = title_config['sub_title_list']
        self.title_prefix = title_config['prefix']

        self.model = model_prototype

        self.dataloader = dataloader
        self.EMA = EMA(history=EMA_config[0], bias=EMA_config[1]) if EMA_config else False
        self.display = display
        self.reverse_para = reverse_para

        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model = self.model.to(device=torch.device('cpu'))
        if self.EMA: 
            self.EMA.register(self.model)
            self.EMA.load()
        self.model.eval()

    def draw(self, clip=500):
        pred_collector = []
        label_collector = []
        
        if self.EMA: self.EMA.apply(device='cpu')
        
        def reverse_normalization(line, mean, std):
            return (line * std) + mean

        def score_line(pred_label_line, label_line):
            y_bar = label_line.mean(axis=0)
            y_hat = pred_label_line
            y = label_line

            tss = np.sum((y - y_bar) ** 2)
            rss = np.sum((y_hat - y) ** 2)
            ess = np.sum((y_hat - y_bar) ** 2)

            return ess / tss

        for idx, (data, label) in enumerate(self.dataloader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.model(* data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.model(data)

            if pred_label.shape[1] > 1:
                chip_p = pred_label.detach().squeeze().numpy()[:, -1]
                chip_l = label.detach().squeeze().numpy()[:, -1]
            else:
                chip_p = pred_label.detach().squeeze().numpy()
                chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(reverse_normalization(chip_p, self.reverse_para[0], self.reverse_para[1]))
            label_collector.extend(reverse_normalization(chip_l, self.reverse_para[0], self.reverse_para[1]))
            
            if clip >= 0 and len(label_collector) >= clip:
                break
        
        figure_df = pd.DataFrame({'Prediction': pred_collector, 'Real': label_collector}, index=self.sub_title_list)
        titles = sorted(list(set(figure_df.index.map(lambda x: str(x.date())))))
        r_square = score_line(figure_df.loc[:, 'Prediction'].to_numpy(), figure_df.loc[:, 'Real'].to_numpy())

        corr_collector = []
        for title in titles:
            corr = np.corrcoef([figure_df.loc[title, 'Prediction'], figure_df.loc[title, 'Real']])[0, 1]
            corr_collector.append(corr)

            figure_df.loc[title, :].plot(figsize=(16, 9),
                                        lw=1,
                                        alpha=1,
                                        xlabel='Time/h',
                                        ylabel='Normalized Value',
                                        title=self.title_prefix + ' Prediction for ' + title)
            plt.grid(axis='y')
            plt.savefig(self.result_path + self.title_prefix + ' Prediction for ' + title + '.png')
            plt.close()


        res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        res.plot(kind='line',
                 figsize=(24, 9),
                 alpha=1,
                 lw=1,
                 xlabel='Samples',
                 ylabel='Values',
                 title='Fit on Testing Data')
        plt.grid(axis='y')
        plt.text(0.4, 0.7, f'R-Square: {r_square:.3f}', fontsize=12, transform=plt.gca().transAxes)

        if self.display:
            plt.show()
            plt.close()
        else:
            plt.savefig(self.result_path + self.title_prefix + ' Overview.png')
            plt.close()

        corr_fig = pd.Series(corr_collector, index=titles)
        corr_fig.plot(kind='bar',
                      figsize=(16, 9),
                      title='Correlation Overview',
                      xlabel='Date/d',
                      ylabel='Correlation Coefficient')
        pd.Series([0.5] * len(titles), index=titles).plot(kind='line',
                                                          color='red',
                                                          style='--',
                                                          ylim=(0, 1),
                                                          figsize=(16, 9),
                                                          title='Correlation Overview',
                                                          xlabel='Date/d',
                                                          ylabel='Correlation Coefficient')
        plt.xticks(rotation=90)
        plt.savefig(self.result_path + self.title_prefix + ' Correlation Coefficient.png')
        plt.close()
        if self.EMA: self.EMA.revert(device='cpu')
        print(f'>>>[Drawer]: Done!')
        
        return res, r_square


