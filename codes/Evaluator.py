import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix, classification_report


# Binary Classification ================================================================================================
class BinaryClassificationEvaluator:
    def __init__(self, loader, model, types):
        self.loader = loader
        self.modelFunct = model
        self.types = types

    def evaluate(self):
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        self.modelFunct.eval()

        pred_collector = []
        label_collector = []
        for batch_idx, (x, label) in enumerate(self.loader):
            label = label.to(device=torch.device('cpu'))
            if type(x) is tuple or type(x) is list:
                x = map(lambda ele: ele.to(device=torch.device('cpu')), x)
                pred_label = self.modelFunct(*x)
            else:
                x = x.to(device=torch.device('cpu'))
                pred_label = self.modelFunct(x)

            chip_p = pred_label.detach().squeeze().numpy().argmax(axis=1)
            chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(chip_p)
            label_collector.extend(chip_l)

        res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        cm = confusion_matrix(res.loc[:, 'Label'], res.loc[:, 'Prediction'])
        cm = pd.DataFrame(cm, index=['N\u0302', 'P\u0302'], columns=['N', 'P'])

        report = classification_report(res.loc[:, 'Label'],
                                       res.loc[:, 'Prediction'],
                                       target_names=self.types,
                                       zero_division=0)

        print(' Confusion Matrix '.center(54, '='))
        print(cm)

        print(' Classification Report '.center(54, '='))
        print(report)
