import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


class Visualizer:
    def __init__(self, xs, ys, highlights):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.highlights= highlights

    def visualize(self, title='New Graph', x_title='X-Axis', y_title='Y-Axis', legends=None, zoom_in=(0.3, 0.7)):
        plt.figure(figsize=(16, 9))
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        if len(self.ys.shape) > 1:
            temp_y = np.array(self.ys)[int(len(self.ys) * zoom_in[0]): int(len(self.ys) * zoom_in[1]), :]
            temp_min = temp_y.min(axis=0)
            min_ys = min(temp_min)

            temp_max = temp_y.max(axis=0)
            max_ys = max(temp_max)

            for i in range(0, self.ys.shape[1]):
                plt.plot(self.xs[int(len(self.ys) * zoom_in[0]): int(len(self.ys) * zoom_in[1])], self.ys[:, i][int(len(self.ys) * zoom_in[0]): int(len(self.ys) * zoom_in[1])], label=str(legends[i]))

        else:
            temp_y = np.array(self.ys)
            min_ys = min(temp_y)

            max_ys = max(temp_y)

            plt.plot(self.xs, self.ys, label=str(legends))

        margin = abs(max_ys - min_ys) * 0.2

        for gap1, gap2 in self.highlights:
            plt.fill_between([gap1, gap2],
                             [min_ys - margin, min_ys - margin],
                             [max_ys + margin, max_ys + margin],
                             alpha=0.2,
                             color='green')

        plt.ylim(min_ys, max_ys)
        plt.legend(loc='upper right')
        plt.show()
        plt.close()


# Using Sample
if __name__ == '__main__':
    viz = Visualizer(np.linspace(1, 10, 1000), np.sin(np.linspace(1, 10, 1000)), [(1, 2), (5, 7)])
    viz.visualize(title='Sin function', legends='Y values')
