import numpy as np
import pandas as pd


class EMASmoother:
    def __init__(self, window, adjust=True):
        self.window = window
        self.adjust = adjust
        self.t = 0

        if self.adjust:
            self.alpha = 2 / (self.window + 1)
            self.old_value = [0, 0]
        else:
            self.alpha = 1 - (1 / self.window)
            self.old_value = 0

        self.is_init = False

    def dynamic_EMA(self, raw):
        if not self.is_init:
            self.is_init = True

            if self.adjust:
                self.old_value[0] = raw
                self.old_value[1] = 1
                self.t += 1
            else:
                self.old_value = raw

        if self.adjust:
            self.old_value[0] = self.old_value * (1 - self.alpha) + raw
            self.old_value[1] += (1 - self.alpha) ** self.t
            self.t += 1
        else:
            self.old_value = self.alpha * self.old_value + (1 - self.old_value) * raw

        if self.adjust:
            return self.old_value[0] / self.old_value[1]
        else:
            return self.old_value

    def static_EMA(self, raws):
        if len(raws.shape) > 1:
            return pd.DataFrame(raws).ewm(span=self.window, adjust=self.adjust).mean()
        else:
            return pd.Series(raws).ewm(span=self.window, adjust=self.adjust).mean()






