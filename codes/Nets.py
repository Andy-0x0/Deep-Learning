import random
import pandas as pd
import numpy as np
import math

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.nn.modules as mods


# Global Seed Initialization ===========================================================================================
def seed_setting(seed_n=42):
    print('Using Random Seed ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)

    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


# Weight Initialization ================================================================================================
def construct_stylized_init(linear='kaiming_uniform', cnn='kaiming_uniform', rnn='xvaier_uniform'):
    def _execute_init_block(layer, method):
        if method == 'xavier_normal':
            torch.nn.init.xavier_normal_(layer)
        elif method == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(layer)
        elif method == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(layer)
        elif method == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(layer)
        elif type(method) is list or type(method) is tuple:
            torch.nn.init.constant_(layer, val=method[-1])
        else:
            pass

    def init_stylized(layer_model_obj):
        if type(layer_model_obj) is torch.nn.Linear:
            _execute_init_block(layer_model_obj.weight, linear)

        if type(layer_model_obj) is torch.nn.Conv2d:
            _execute_init_block(layer_model_obj.weight, cnn)

        if type(layer_model_obj) is torch.nn.GRU or type(layer_model_obj) is torch.nn.LSTM:
            for param in layer_model_obj._flat_weights_names:
                if 'weight' in param:
                    _execute_init_block(layer_model_obj._parameters[param], rnn)

    return init_stylized


# CNNs | [Batch, Channel, Height, Width] -> [Batch, Channel]
# LeNet ================================================================================================================
class LeNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.1):
        super(LeNet, self).__init__()

        self.Conv1 = torch.nn.Conv2d(in_channel, in_channel * 8, padding=2, kernel_size=5)
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = torch.nn.Conv2d(in_channel * 8, in_channel * 16, kernel_size=5)
        self.MaxPool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.Flatten = torch.nn.Flatten()

        self.COMPRESS_L1 = 240
        self.COMPRESS_L2 = 120
        self.COMPRESS_L3 = 84

        self.Linear1 = torch.nn.Linear(self.COMPRESS_L1, self.COMPRESS_L2)
        self.Linear2 = torch.nn.Linear(self.COMPRESS_L2, self.COMPRESS_L3)
        self.Linear3 = torch.nn.Linear(self.COMPRESS_L3, out_channel)
        self.Dropout1 = torch.nn.Dropout(dropout)
        self.Dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.MaxPool1(self.Conv1(x))
        x = self.MaxPool2(self.Conv2(x))

        x = self.Flatten(x)

        x = self.Dropout1(F.relu(self.Linear1(x)))
        x = self.Dropout2(F.relu(self.Linear2(x)))
        x = self.Linear3(x)

        return x


# AlexNet ==============================================================================================================
class AlexNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.1):
        super(AlexNet, self).__init__()
        self.DILATE_L1 = 96
        self.DILATE_L2 = 256
        self.DILATE_L3 = 384
        self.DILATE_L4 = 384
        self.DILATE_L5 = 256

        self.Conv1 = torch.nn.Conv2d(in_channel, self.DILATE_L1, kernel_size=7, stride=3)
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2 = torch.nn.Conv2d(self.DILATE_L1, self.DILATE_L2, kernel_size=5, padding=2)
        self.MaxPool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv31 = torch.nn.Conv2d(self.DILATE_L2, self.DILATE_L3, kernel_size=3, padding=1)
        self.Conv32 = torch.nn.Conv2d(self.DILATE_L3, self.DILATE_L4, kernel_size=3, padding=1)
        self.Conv33 = torch.nn.Conv2d(self.DILATE_L4, self.DILATE_L5, kernel_size=3, padding=1)
        self.MaxPool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Flatten = torch.nn.Flatten()

        self.COMPRESS_L1 = 4096
        self.COMPRESS_L2 = 4096
        self.COMPRESS_L3 = 1024

        self.Linear1 = torch.nn.Linear(self.COMPRESS_L1, self.COMPRESS_L2)
        self.Linear2 = torch.nn.Linear(self.COMPRESS_L2, self.COMPRESS_L3)
        self.Linear3 = torch.nn.Linear(self.COMPRESS_L3, out_channel)
        self.Dropout1 = torch.nn.Dropout(dropout)
        self.Dropout2 = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.MaxPool1(self.Conv1(x))
        x = self.MaxPool2(self.Conv2(x))
        x = self.MaxPool3(self.Conv33(self.Conv32(self.Conv31(x))))

        x = self.Flatten(x)

        x = self.Dropout1(F.relu(self.Linear1(x)))
        x = self.Dropout2(F.relu(self.Linear2(x)))
        x = self.Linear3(x)

        return x


# VggNet ===============================================================================================================
class VggNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, layers, linear_out, dropout=0.1):
        super(VggNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layers = layers
        self.linear_out = linear_out

        self.COMPRESS_L1 = 4096
        self.COMPRESS_L2 = 1024
        self.COMPRESS_L3 = 256

        self.VggBlockCluster = self._construct()
        self.Linear1 = torch.nn.Linear(self.COMPRESS_L1, self.COMPRESS_L2)
        self.Linear2 = torch.nn.Linear(self.COMPRESS_L2, self.COMPRESS_L3)
        self.Linear3 = torch.nn.Linear(self.COMPRESS_L3, self.linear_out)
        self.Dropout1 = torch.nn.Dropout(dropout)
        self.Dropout2 = torch.nn.Dropout(dropout)

    def _construct(self):
        models = []

        for index, num in enumerate(self.layers):
            for i in range(num):
                if i == 0:
                    models.append(torch.nn.Conv2d(self.in_channel, self.out_channel[index], kernel_size=3, padding=1))
                    models.append(torch.nn.ReLU())
                else:
                    models.append(
                        torch.nn.Conv2d(self.out_channel[index], self.out_channel[index], kernel_size=3, padding=1))
                    models.append(torch.nn.ReLU())
            models.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            self.in_channel = self.out_channel[index]

        models.append(torch.nn.Flatten())
        return Sequential(*models)

    def forward(self, x):
        x = self.VggBlockCluster(x)

        x = self.Dropout1(F.relu(self.Linear1(x)))
        x = self.Dropout2(F.relu(self.Linear2(x)))
        x = self.Linear3(x)

        return x


# NinNet ===============================================================================================================
class NiNNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernels, strides, paddings, linear_out, dropout=0.1):
        super(NiNNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.linear_out = linear_out

        self.COMPRESS_L1 = 4096
        self.COMPRESS_L2 = 1024
        self.COMPRESS_L3 = 256

        self.NiNBlockCluster = self._construct()
        self.Linear1 = torch.nn.Linear(self.COMPRESS_L1, self.COMPRESS_L2)
        self.Linear2 = torch.nn.Linear(self.COMPRESS_L2, self.COMPRESS_L3)
        self.Linear3 = torch.nn.Linear(self.COMPRESS_L3, self.linear_out)
        self.Dropout1 = torch.nn.Dropout(dropout)
        self.Dropout2 = torch.nn.Dropout(dropout)

    def _construct(self):
        models = []

        for index, out in enumerate(self.out_channel):
            models.append(torch.nn.Conv2d(self.in_channel,
                                          out,
                                          kernel_size=self.kernels[index],
                                          stride=self.strides[index],
                                          padding=self.paddings[index]))
            models.append(torch.nn.MaxPool2d(kernel_size=3, stride=2))
            self.in_channel = self.out_channel[index]

        models.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        models.append(torch.nn.Flatten())
        return Sequential(*models)

    def forward(self, x):
        x = self.NiNBlockCluster(x)

        x = self.Dropout1(F.relu(self.Linear1(x)))
        x = self.Dropout2(F.relu(self.Linear2(x)))
        x = self.Linear3(x)

        return x


# GoogLeNet ============================================================================================================
class _InceptionBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, out_channel3, out_channel4):
        super(_InceptionBlock, self).__init__()
        self.Path1_Conv1by1 = torch.nn.Conv2d(in_channel, out_channel1, kernel_size=1)

        self.Path2_Conv1by1 = torch.nn.Conv2d(in_channel, out_channel2[0], kernel_size=1)
        self.Path2_Conv = torch.nn.Conv2d(in_channel, out_channel2[-1], kernel_size=3, padding=1)

        self.Path3_Conv1by1 = torch.nn.Conv2d(in_channel, out_channel3[0], kernel_size=1)
        self.Path3_Conv = torch.nn.Conv2d(in_channel, out_channel3[-1], kernel_size=5, padding=2)

        self.Path4_MaxPool = torch.nn.MaxPool2d(kernel_size=3, padding=1)
        self.Path4_Conv1by1 = torch.nn.Conv2d(in_channel, out_channel4, kernel_size=1)

    def forward(self, x):
        path1_x = F.relu(self.Path1_Conv1by1(x))

        path2_x = F.relu(self.Path2_Conv1by1(x))
        path2_x = F.relu(self.Path2_Conv(path2_x))

        path3_x = F.relu(self.Path3_Conv1by1(x))
        path3_x = F.relu(self.Path3_Conv(path3_x))

        path4_x = self.Path4_MaxPool(x)
        path4_x = F.relu(self.Path4_Conv1by1(path4_x))

        return torch.cat((path1_x, path2_x, path3_x, path4_x), dim=1)


class GoogLeNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GoogLeNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.googLeNet = self._construct()

    @staticmethod
    def _build_inception_blocks(in_channel, out_channel_seq):
        asemble_line = []
        layers = len(out_channel_seq)

        def calculate_next_inc(seq):
            total_channels = 0
            for ele in seq:
                if isinstance(ele, (tuple, list)):
                    total_channels += ele[-1]
                else:
                    total_channels += int(ele)
            return total_channels

        for index in range(layers):
            current = out_channel_seq[index]

            if not isinstance(current, (list, tuple)) and current == 'MaxPool':
                asemble_line.append(torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
                continue
            elif not isinstance(current, (list, tuple)) and current == 'End':
                asemble_line.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
                asemble_line.append(torch.nn.Flatten())
                continue

            if index == 0:
                asemble_line.append(_InceptionBlock(in_channel,
                                                    current[0], current[1], current[2], current[3]))
            else:
                previous = out_channel_seq[index - 1]
                asemble_line.append(_InceptionBlock(calculate_next_inc(*previous),
                                                    current[0], current[1], current[2], current[3]))

        return asemble_line

    def _construct(self):
        ini_out_channel = 64 * self.in_channel
        ini_block = Sequential(
            torch.nn.Conv2d(self.in_channel, ini_out_channel, kernel_size=7, padding=3, stride=2),
            self._make_customized_norm(out_channel, self.norm),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            torch.nn.Conv2d(ini_out_channel, ini_out_channel, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ini_out_channel, ini_out_channel * 3, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )

        main_block = Sequential(
            * self._build_inception_blocks(ini_out_channel * 3,
                                           [[64, (96, 128), (16, 32), 32],
                                                          [128, (128, 192), (32, 96), 64],
                                                          'MaxPool',
                                                          [192, (96, 208), (16, 48), 64],
                                                          [160, (112, 224), (24, 64), 64],
                                                          [128, (128, 256), (24, 64), 64],
                                                          [112, (144, 288), (32, 64), 64],
                                                          [256, (160, 320), (32, 128), 128],
                                                          'MaxPool',
                                                          [256, (160, 320), (32, 128), 128],
                                                          [384, (192, 384), (48, 128), 128],
                                                          'End'])
        )

        end_block = Sequential(
            torch.nn.Linear(1024, self.out_channel)
        )

        googLeNet = Sequential(
            ini_block,
            main_block,
            end_block
        )

        return googLeNet

    def forward(self, x):
        return self.googLeNet(x)


# ResNet ===============================================================================================================
class _ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, sub=False, norm='bn'):
        super(_ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1 + int(sub))
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        self.norm1 = _make_customized_norm(out_channel, norm)
        self.norm2 = _make_customized_norm(out_channel, norm)

        self.conv1by1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1 + int(sub))
        self.use1by1 = (in_channel != out_channel)

    @staticmethod
    def _make_customized_norm(out_channel, norm):
        if 'bn' in norm.lower():
            return torch.nn.BatchNorm2d(out_channel)
        elif 'gn' in norm.lower():
            group = norm.split('-')[-1]
            return torch.nn.GroupNorm(group, out_channel)
        else:
            return lambda x: x

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))

        if self.use1by1:
            x = self.conv1by1(x)
        y += x

        return F.relu(y)


class ResNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn'):
        super(ResNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm = norm

        self.ResNetLayer = self._construct()
        self.Norm = self._make_customized_norm(out_channel, norm)

    @staticmethod
    def _make_customized_norm(out_channel, norm):
        if 'bn' in norm.lower():
            return torch.nn.BatchNorm2d(out_channel)
        elif 'gn' in norm.lower():
            group = norm.split('-')[-1]
            return torch.nn.GroupNorm(group, out_channel)
        else:
            return lambda x: x

    def _build_res_blocks(self, in_channel, out_channel, layers, sub=False):
        asemble_line = []

        for i in range(layers):
            if i == 0:
                asemble_line.append(_ResBlock(in_channel, out_channel, sub=sub, norm=self.norm))
            else:
                asemble_line.append(_ResBlock(out_channel, out_channel, sub=False, norm=self.norm))

        return asemble_line

    def _construct(self):
        COMPRESS_L1 = 8192
        COMPRESS_L2 = 4096
        COMPRESS_L3 = 1024

        ini_out_channel = 16 * self.in_channel
        ini_block = Sequential(
            torch.nn.Conv2d(self.in_channel, ini_out_channel, kernel_size=7, padding=3, stride=2),
            self._make_customized_norm(out_channel, self.norm),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        resnet = Sequential(
            ini_block,

            Sequential(*self._build_res_blocks(ini_out_channel, ini_out_channel, 2)),
            Sequential(*self._build_res_blocks(ini_out_channel, ini_out_channel * 2, 2, sub=True)),
            Sequential(*self._build_res_blocks(ini_out_channel * 2, ini_out_channel * 4, 2)),
            Sequential(*self._build_res_blocks(ini_out_channel * 4, ini_out_channel * 8, 2)),
            Sequential(*self._build_res_blocks(ini_out_channel * 8, ini_out_channel * 16, 2)),
            torch.nn.AvgPool2d(kernel_size=1),
            torch.nn.Flatten(),

            torch.nn.Linear(COMPRESS_L1, COMPRESS_L2),
            self._make_customized_norm(COMPRESS_L2, self.norm),
            torch.nn.ReLU(),

            torch.nn.Linear(COMPRESS_L2, COMPRESS_L3),
            self._make_customized_norm(COMPRESS_L3, self.norm),
            torch.nn.ReLU(),

            torch.nn.Linear(COMPRESS_L3, self.out_channel),
        )

        return resnet

    def forward(self, x):
        x = self.ResNetLayer(x)
        x = self.Norm(x)
        return x


# RNNs | [Batch, Seq, Dimensions] -> [Batch, Seq, Hidden] or [Batch, Hidden]
# GRU (Many-to-Many) ===================================================================================================
class GRUSeq2Seq(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=3, dropout=0.0):
        super(GRUSeq2Seq, self).__init__()
        self.Gru = torch.nn.GRU(in_dims, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        o, _ = self.Gru(x)
        return o


# GRU (Many-to-One) ====================================================================================================
class GRUSeq2Point(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=3, dropout=0.0):
        super(GRUSeq2Point, self).__init__()
        self.Gru = torch.nn.GRU(in_dims, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        o, _ = self.Gru(x)
        o = o[:, -1, :]
        return o.squeeze()


# LSTM (Many-to-Many) ==================================================================================================
class LSTMSeq2Seq(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=3, dropout=0.0):
        super(LSTMSeq2Seq, self).__init__()
        self.Lstm = torch.nn.LSTM(in_dims, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        o, _ = self.Lstm(x)
        return o


# LSTM (Many-to-One) ===================================================================================================
class LSTMSeq2Point(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=3, dropout=0.0):
        super(LSTMSeq2Point, self).__init__()
        self.Lstm = torch.nn.LSTM(in_dims, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        o, _ = self.Lstm(x)
        o = o[:, -1, :]
        return o.squeeze()


# Transformers | [Batch, Seq, Dimensions] -> [Batch, Seq, Hidden] or [Batch, Hidden]
# Transformer (Many-to-One) ============================================================================================
class SinCosPositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq, hidden, dropout=0.1):
        super(SinCosPositionalEncoding, self).__init__()
        self.Dropout = torch.nn.Dropout(dropout)
        self.Panel = torch.zeros((1, max_seq, hidden))
        position = (torch.arange(max_seq, dtype=torch.float).reshape(-1, 1) /
                    torch.pow(10000, torch.arange(0, hidden, 2, dtype=torch.float) / hidden))
        self.Panel[:, :, 0::2] = torch.sin(position)
        self.Panel[:, :, 1::2] = torch.cos(position)

    def forward(self, x):
        x = x + self.Panel[:, :x.shape[1], :].to(x.device)
        x = self.Dropout(x)
        return x


class TransformerSeq2Point(torch.nn.Module):
    def __init__(self, transformer_config, output_config):
        super(TransformerSeq2Point, self).__init__()

        # Inside a Transformer
        if transformer_config['mode'] == 'ts':
            self.EmbeddingEnc = torch.nn.Linear(transformer_config['in_dim_enc'], transformer_config['hidden_dim'])
            self.EmbeddingDec = torch.nn.Linear(transformer_config['in_dim_dec'], transformer_config['hidden_dim'])
        else:
            self.EmbeddingEnc = torch.nn.Embedding(transformer_config['in_dim_enc'], transformer_config['hidden_dim'])
            self.EmbeddingDec = torch.nn.Embedding(transformer_config['in_dim_dec'], transformer_config['hidden_dim'])
        self.PositionalEncodingEnc = SinCosPositionalEncoding(10000,
                                                              transformer_config['hidden_dim'],
                                                              transformer_config['dropout'])
        self.PositionalEncodingDec = SinCosPositionalEncoding(10000,
                                                              transformer_config['hidden_dim'],
                                                              transformer_config['dropout'])
        self.Transformer = torch.nn.Transformer(d_model=transformer_config['hidden_dim'],
                                                nhead=transformer_config['num_heads'],
                                                num_encoder_layers=transformer_config['num_layers_enc'],
                                                num_decoder_layers=transformer_config['num_layers_dec'],
                                                dropout=transformer_config['dropout'],
                                                batch_first=True)

        # Outside a Transformer
        if output_config['mode'].upper() == 'G':
            self.Rnn = GRUSeq2Point(transformer_config['hidden_dim'],
                                    output_config['hidden_dim'],
                                    output_config['num_layers'],
                                    output_config['dropout'])
        else:
            self.Rnn = LSTMSeq2Point(transformer_config['hidden_dim'],
                                     output_config['hidden_dim'],
                                     output_config['num_layers'],
                                     output_config['dropout'])

    def forward(self, enc_x, dec_x):
        enc_x = self.EmbeddingEnc(enc_x)
        enc_x = self.PositionalEncodingEnc(enc_x)
        dec_x = self.EmbeddingDec(dec_x)
        dec_x = self.PositionalEncodingDec(dec_x)

        o = self.Transformer(enc_x, dec_x)
        o = self.Rnn(o)

        return o


# Hybrid ===============================================================================================================
# Face1 Model for ResNetTransformer2ResNetRnn
class ResNet18GRUF1(torch.nn.Module):
    def __init__(self, resnet_config, rnn_config, linear_out=1, mode='L'):
        super(ResNet18GRUF1, self).__init__()
        self.ResNet = ResNet(resnet_config['in_channel'], resnet_config['out_channel'])

        if mode.upper() == 'G':
            self.Rnn = GRUSeq2Point(rnn_config['in_dim'],
                                    rnn_config['hidden_dim'],
                                    rnn_config['num_layers'],
                                    rnn_config['dropout'])
        else:
            self.Rnn = LSTMSeq2Point(rnn_config['in_dim'],
                                     rnn_config['hidden_dim'],
                                     rnn_config['num_layers'],
                                     rnn_config['dropout'])

        self.BatchNorm = torch.nn.BatchNorm1d(rnn_config['hidden_dim'] + resnet_config['out_channel'])
        self.Linear1 = torch.nn.Linear(rnn_config['hidden_dim'] + resnet_config['out_channel'],
                                       2 * rnn_config['hidden_dim'] + resnet_config['out_channel'])
        self.Linear2 = torch.nn.Linear(2 * rnn_config['hidden_dim'] + resnet_config['out_channel'],
                                       rnn_config['hidden_dim'])
        self.Linear3 = torch.nn.Linear(rnn_config['hidden_dim'], linear_out)

    def forward(self, x, s):

        x = self.ResNet(x)                      # (batch_size, out_channel)
        s = self.Rnn(s)                         # (batch_size, hidden_dim)

        o = torch.concat([x, s], dim=1)  # (batch_size, out_channel + hidden_dim)

        o = self.BatchNorm(o)
        o = F.relu(self.Linear1(o))
        o = F.relu(self.Linear2(o))
        o = self.Linear3(o)

        return o


# Face2 Model for ResNetTransformer2ResNetRnn
class ResNetTransformer(torch.nn.Module):
    def __init__(self, resnet_config, transformer_config, rnn_config, linear_out=1):
        super(ResNetTransformer, self).__init__()

        # for seq_cnn preprocessing
        self.ResNet = ResNet(resnet_config['in_channel'], resnet_config['out_channel'])

        # for seq_cnn and seq_label main
        self.Transformer = TransformerSeq2Point(
            {
                'mode': transformer_config['mode'],
                'in_dim_enc': transformer_config['in_dim_enc'],
                'in_dim_dec': transformer_config['in_dim_dec'],
                'hidden_dim': transformer_config['hidden_dim'],
                'num_heads': transformer_config['num_heads'],
                'num_layers_enc': transformer_config['num_layers_enc'],
                'num_layers_dec': transformer_config['num_layers_dec'],
                'dropout': transformer_config['dropout']
            },
            {
                'mode': rnn_config['mode'],
                'hidden_dim': rnn_config['hidden_dim'],
                'num_layers': rnn_config['num_layers'],
                'dropout': rnn_config['dropout']
            }
        )

        # for squeezing into next label's prediction
        self.BatchNorm = torch.nn.BatchNorm1d(resnet_config['out_channel'])
        self.GroupNorm = torch.nn.GroupNorm(3, resnet_config['out_channel'])
        self.Linear1 = torch.nn.Linear(resnet_config['out_channel'], resnet_config['out_channel'] * 2)
        self.Linear2 = torch.nn.Linear(resnet_config['out_channel'] * 2, resnet_config['out_channel'] // 2)
        self.Linear3 = torch.nn.Linear(resnet_config['out_channel'] // 2, linear_out)

    def forward(self, seq_cnn, seq_label):
        # ResNet
        with torch.no_grad():
            batch_size = seq_cnn.shape[0]
            seq = seq_cnn.shape[1]
            channel = seq_cnn.shape[2]
            height = seq_cnn.shape[3]
            width = seq_cnn.shape[4]
        seq_cnn = seq_cnn.view(-1, channel, height, width)
        seq_cnn = self.ResNet(seq_cnn)
        seq_cnn = seq_cnn.view(batch_size, seq, -1)

        # Transformer
        o = self.Transformer(seq_cnn, seq_label)

        # FullConnection
        # o = self.GroupNorm(o)
        o = self.BatchNorm(o)

        o = F.relu(self.Linear1(o))
        o = F.relu(self.Linear2(o))
        o = self.Linear3(o)
        return o


class ResNetTransformer2ResNetRnn(torch.nn.Module):
    def __init__(self, face1dict, face2dict):
        super(ResNetTransformer2ResNetRnn, self).__init__()

        self.F1_net = ResNetTransformer(resnet_config=face1dict['resnet_config'],
                                        transformer_config=face1dict['transformer_config'],
                                        rnn_config=face1dict['rnn_config'],
                                        linear_out=20)

        self.F2_net = ResNet18GRUF1(face2dict['resnet_config'],
                                    face2dict['rnn_config'],
                                    linear_out=10)

        self.BatchNorm_transformer = torch.nn.BatchNorm1d(20)
        self.BatchNorm_resnetgru = torch.nn.BatchNorm1d(10)

        # self.GroupNorm_transformer = torch.nn.GroupNorm(4, 20)
        # self.GroupNorm_resnetgru = torch.nn.GroupNorm(2, 10)

        self.Linear_F2_cat = torch.nn.Linear(20 + 10, 1)

    def forward(self, seq_cnn, cnn, seq_label):
        o = self.BatchNorm_transformer(self.F1_net(seq_cnn, seq_label))
        r = self.BatchNorm_resnetgru(self.F2_net(cnn, seq_label))

        o = self.Linear_F2_cat(torch.concat((o, r), dim=-1))

        return o

# class ResNetRnn2ResNetTransformer(torch.nn.Module):
#     def __init__(self, face1dict, face2dict, face=1):
#         super(ResNetRnn2ResNetTransformer, self).__init__()
#         self.face = face

#         self.F1_net = ResNet18GRUF1(face1dict['resnet_config'],
#                                     face1dict['rnn_config'],
#                                     linear_out=1)

#         self.F2_net = ResNetTransformer(resnet_config=face2dict['resnet_config'],
#                                         transformer_config=face2dict['transformer_config'],
#                                         rnn_config=face2dict['rnn_config'],
#                                         linear_out=1)

#         self.Linear_F2_cat = torch.nn.Linear(2, 1)

#     def forward(self, seq_cnn, cnn, seq_label):
#         o = self.F1_net(cnn, seq_label)

#         if self.face == 2:
#             r = self.F2_net(seq_cnn, seq_label)
#             o = self.Linear_F2_cat(torch.concat((o, r), dim=-1))

#         return o