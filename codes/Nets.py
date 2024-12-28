import random
import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from torchvision.models.resnet import ResNet as PResNet
from torchvision.models.resnet import BasicBlock, Bottleneck


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

        self.Conv1 = torch.nn.Conv2d(in_channel, in_channel * 8, padding=2, kernel_size=3)
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = torch.nn.Conv2d(in_channel * 8, in_channel * 16, kernel_size=5)
        self.MaxPool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.Flatten = torch.nn.Flatten()

        self.COMPRESS_L1 = 768
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
class ResNet(PResNet):
    def __init__(
        self,
        in_channel: int = 1,
        num_classes: int = 2,
        status: tuple = (18, False),
        block: type[BasicBlock, Bottleneck] = BasicBlock,
        layers: list[int] = (2, 2, 2, 2),
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.in_channel = in_channel
        self.prototype = status[0]
        self.pretrained = status[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self._revise_model()

    def _revise_model(self):
        # Select the prototype to use
        if self.prototype == 18:
            self.layers = [2, 2, 2, 2]
            self.block = BasicBlock
        elif self.prototype == 34:
            self.layers = [3, 4, 6, 3]
            self.block = BasicBlock
        elif self.prototype == 50:
            self.layers = [3, 4, 6, 3]
            self.block = Bottleneck
        elif self.prototype == 101:
            self.layers = [3, 4, 23, 3]
            self.block = Bottleneck
        elif self.prototype == 152:
            self.layers = [3, 8, 36, 3]
            self.block = Bottleneck
        else:
            self.layers = [2, 2, 2, 2]
            self.block = BasicBlock

        # Revise the accepted input channel size
        self.conv1 = torch.nn.Conv2d(self.in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self._forward_impl(x)


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
# Transformers ============================================================================================
# Positional Encoding according to "Attention is All You Need"
class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq:int = 4096, in_dim:int = 512) -> None:
        '''
        initialization

        :param max_seq: A number that guarantees being greater that you inputs' seq dimension
        :param in_dim: The number of you inputs' feature dimension (dim-0)
        '''
        super(PositionalEncoding, self).__init__()
        # according to "Attention is All You Need" PosEnc sin|cos (pos / 10000^(2i/in_dim))
        # pos:          dim-1 -> the sequence dimension
        # i & in_dim:   dim-0 -> the feature(embedding) dimension
        positional_enc = (torch.arange(max_seq).reshape(-1, 1) / torch.pow(10000, torch.arange(0, in_dim, 2) / in_dim))

        # Apply the positional encoding on the large panel to cover the [Batch, Seq, In_dim] inputs
        self.Panel = torch.zeros((1, max_seq, in_dim))      # [Batch=1, Max_Seq=4096, In_dim=512]
        self.Panel[:, :, 0::2] = torch.sin(positional_enc)
        self.Panel[:, :, 1::2] = torch.cos(positional_enc)

    def forward(self, x):
        # cover the input with the larger positional encoding panel, cut the panel to [Batch, Max_seq=seq, In_dim]
        with torch.no_grad():
            self.Panel = self.Panel.to(device=x.device)
            x = x + self.Panel[:, :x.shape[1], :]
            return x


# Smart Embedding that can change the mode based on time series / discrete instances i.e. words
class AdaEmbedding(torch.nn.Module):
    def __init__(self, mode:str = 'ts', vocab_dim:int = 10000, embedding_dim:int = 512) -> None:
        '''
        Initialization

        :param mode: The mode for embedding a time series 'ts' or embedding a discrete instance 'di'
        :param vocab_dim: The total vocabulary size if in 'di' mode
                          The original feature size if in 'ts' mode
        :param embedding_dim: The embedding dimension to embed the discrete instance if in 'di' mode
                              The original input feature dimension of the time series if in 'ts' mode
        '''
        super(AdaEmbedding, self).__init__()
        # No embedding if mode == 'ts'
        if mode == 'ts':
            self.embedding = torch.nn.Linear(vocab_dim, embedding_dim)

        # embedding as arranged if mode == 'di'
        elif mode == 'di':
            self.embedding = torch.nn.Embedding(vocab_dim, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# Smart Transformer for Classification usage
class Transformer(torch.nn.Module):
    def __init__(
        self,
        mode:tuple[str, str] = ('ts', 'di'),
        vocab_dim:tuple[int, int] = (128, 2),
        embedding_dim:int|str = 'auto',
        positional_dim:int = 4096,
        head_num:int = 8,
        encoder_num:int = 6,
        decoder_num:int = 6,
        dropout:float = 0.1,
        batch_first:bool = True,
        aft_mode: str = 'G',
        aft_hidden_lay: int = 5,
        aft_hidden_dim:int = 128
    ) -> None:
        '''
        Initialization

        :param mode: The mode for embedding a time series 'ts' or embedding a discrete instance 'di'
        :param vocab_dim: The total vocabulary size if in 'di' mode
                          The original feature size if in 'ts' mode
        :param embedding_dim: The embedding dimension to embed the discrete instance if in 'di' mode, 'auto' for automatically decidsion
                              The original input feature dimension of the time series if in 'ts' mode, 'auto' for automatically decidsion
        :param positional_dim: A number that guarantees being greater that you inputs' seq dimension
        :param head_num: The number of the multi-head attention
        :param encoder_num: The number of the encoder layers
        :param decoder_num: The number of the decoder layers
        :param dropout: The rate for dropout
        :param batch_first: Recognizing the input shape as [Batch, Seq, Embedding_dim]
        '''
        super(Transformer, self).__init__()

        self.embedding_dim = None

        # Embedding Layer
        # Choose the embedding dimension smartly
        def choose_embedding_dim(dictionary_size):
            EMB_DIMS = np.array([64, 128, 256, 512, 768, 1024])
            indicator_dim = np.floor(np.sqrt(dictionary_size))

            attempt_dim = EMB_DIMS - indicator_dim
            for idx in range(len(attempt_dim)):
                if attempt_dim[idx] > 0:
                    return EMB_DIMS[idx]

            return int(indicator_dim)

        if mode == ('ts', 'ts'):
            # Probably Regression Task: in_dim=d -> out_dim=d
            # using the in_dim as the "embedding" if auto
            # using the assigned dim as the "embedding" if int
            if isinstance(embedding_dim, str) and embedding_dim == 'auto':
                self.embedding_dim = int(max(vocab_dim))
            elif isinstance(embedding_dim, (int, float)):
                self.embedding_dim = int(embedding_dim)
            self.embedding_enc = AdaEmbedding(mode='ts', vocab_dim=vocab_dim[0], embedding_dim=self.embedding_dim)
            self.embedding_dec = AdaEmbedding(mode='ts', vocab_dim=vocab_dim[1], embedding_dim=self.embedding_dim)

        elif mode == ('ts', 'di'):
            # Probably Classification Task: in_dim=d -> out_dim=l
            # using the maximum of in_dim and smart(out_dim) as the "embedding" if auto
            # using the assigned dim as the "embedding" if int
            if isinstance(embedding_dim, str) and embedding_dim == 'auto':
                self.embedding_dim = int(max(vocab_dim[0], choose_embedding_dim(vocab_dim[1])))
            elif isinstance(embedding_dim, (int, float)):
                self.embedding_dim = int(embedding_dim)
            self.embedding_enc = AdaEmbedding(mode='ts', vocab_dim=vocab_dim[0], embedding_dim=self.embedding_dim)
            self.embedding_dec = AdaEmbedding(mode='di', vocab_dim=vocab_dim[1], embedding_dim=self.embedding_dim)

        elif mode == ('di', 'ts'):
            # Probably Rating Task: in_dim=l -> out_dim=d
            # using the maximum of smart(in_dim) and out_dim as the "embedding" if auto
            # using the assigned dim as the "embedding" if int
            if isinstance(embedding_dim, str) and embedding_dim == 'auto':
                self.embedding_dim = int(max(choose_embedding_dim(vocab_dim[0]), vocab_dim[1]))
            elif isinstance(embedding_dim, (int, float)):
                self.embedding_dim = int(embedding_dim)
            self.embedding_enc = AdaEmbedding(mode='di', vocab_dim=vocab_dim[0], embedding_dim=self.embedding_dim)
            self.embedding_dec = AdaEmbedding(mode='ts', vocab_dim=vocab_dim[1], embedding_dim=self.embedding_dim)

        elif mode == ('di', 'di'):
            # Probably Text Prediction Task: in_dim=l -> out_dim=l
            # using the maximum of smart(in_dim) and smart(out_dim) as the "embedding" if auto
            # using the assigned dim as the "embedding" if int
            if isinstance(embedding_dim, str) and embedding_dim == 'auto':
                self.embedding_dim = int(max(choose_embedding_dim(vocab_dim[0]), choose_embedding_dim(vocab_dim[1])))
            elif isinstance(embedding_dim, (int, float)):
                self.embedding_dim = int(embedding_dim)
            self.embedding_enc = AdaEmbedding(mode='di', vocab_dim=vocab_dim[0], embedding_dim=self.embedding_dim)
            self.embedding_dec = AdaEmbedding(mode='di', vocab_dim=vocab_dim[1], embedding_dim=self.embedding_dim)

        else:
            raise Exception(f"Unknown mode: '{mode}' for TransformerClassifier!")

        # Positional Encoding Layer
        self.positional_enc = PositionalEncoding(positional_dim, self.embedding_dim)
        self.positional_dec = PositionalEncoding(positional_dim, self.embedding_dim)

        # Encoder-Decoder Layer
        self.transformer = torch.nn.Transformer(d_model=self.embedding_dim,
                                                nhead=head_num,
                                                num_encoder_layers=encoder_num,
                                                num_decoder_layers=decoder_num,
                                                dropout=dropout,
                                                batch_first=batch_first)

        # Afterwards Layer
        self.batch_norm1 = torch.nn.BatchNorm1d(num_features=self.embedding_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(num_features=aft_hidden_dim)
        if aft_mode == 'G':
            self.after = GRUSeq2Point(in_dims=self.embedding_dim, hidden_dim=aft_hidden_dim, num_layers=aft_hidden_lay)
        else:
            self.after = LSTMSeq2Point(in_dims=self.embedding_dim, hidden_dim=aft_hidden_dim, num_layers=aft_hidden_lay)

    def forward(self, enc_x, dec_x):
        enc_x = self.embedding_enc(enc_x)
        enc_x = self.positional_enc(enc_x)
        dec_x = self.embedding_dec(dec_x)
        dec_x = self.positional_dec(dec_x)

        o = self.transformer(enc_x, dec_x)

        o = o.permute(0, 2, 1)
        o = self.batch_norm1(o)
        o = o.permute(0, 2, 1)

        o = self.after(o)
        self.batch_norm2(o)

        return o


# Hybrid ===============================================================================================================
