import torch
from torch import nn

from residual_stacked_encoder import ResidualStackedEncoder
from shortcut_stacked_encoder import ShortcutStackedEncoder
from enum import Enum


class LayersType(Enum):
    Shortcut = 1
    Residual = 2


class ResidualLSTMEncoder(nn.Module):
    def __init__(self,
                 max_sentence_length,
                 embedding_vectors,
                 padding_index,
                 layers_def,
                 hidden_mlp,
                 output_size,
                 device,
                 layers_type):
        super(ResidualLSTMEncoder, self).__init__()
        self.output_size = output_size
        self.max_sentence_length = max_sentence_length
        self.device = device

        if layers_type == LayersType.Shortcut:
            self.stacked_encoder = ShortcutStackedEncoder(embedding_vectors=embedding_vectors,
                                                          padding_index=padding_index,
                                                          layers_def=layers_def,
                                                          max_sentence_length=max_sentence_length,
                                                          device=device)
        elif layers_type == LayersType.Residual:
            self.stacked_encoder = ResidualStackedEncoder(embedding_vectors=embedding_vectors,
                                                          padding_index=padding_index,
                                                          layers_def=layers_def,
                                                          max_sentence_length=max_sentence_length,
                                                          device=device)
        else:
            raise Exception(
                "Unknown layer type. Please use LayersType enum to specify which layers you would like to use.")

        self.dropout = nn.Dropout(0.1)

        self.hidden = nn.Linear(4 * self.stacked_encoder.output_size, hidden_mlp)
        self.hidden_activation = nn.ReLU()

        self.output_layer = nn.Linear(hidden_mlp, output_size)

        self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x_1, l_1, x_2, l_2):
        x_1_encode = self.stacked_encoder(x_1, l_1, sort=True)
        x_2_encode = self.stacked_encoder(x_2, l_2, sort=False)

        minus_encode = torch.abs(x_1_encode - x_2_encode)
        mult_encode = x_1_encode * x_2_encode

        concat = torch.cat((x_1_encode, x_2_encode, minus_encode, mult_encode), dim=1)

        concat = self.dropout(concat)
        out = self.hidden(self.hidden_activation(concat))

        out = self.dropout(out)
        out = self.output_layer(out)

        return out
        # return self.output_activation(out)


class LSTMLayer:
    def __init__(self, hidden_size, num_layers, bidirectional):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = self.hidden_size
        if bidirectional:
            self.output_size *= 2
