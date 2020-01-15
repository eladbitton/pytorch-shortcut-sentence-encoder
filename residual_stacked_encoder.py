import torch
from torch import nn


class LSTMLayer:
    def __init__(self, hidden_size, num_layers, bidirectional):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_size = self.hidden_size
        if bidirectional:
            self.output_size *= 2


class ResidualStackedEncoder(nn.Module):
    def __init__(self,
                 max_sentence_length,
                 embedding_vectors,
                 padding_index,
                 layers_def,
                 pool_kernel,
                 device):
        super(ResidualStackedEncoder, self).__init__()
        self.padding_index = padding_index
        self.embedding_dim = len(embedding_vectors[0])
        self.device = device
        self.embedding = nn.Embedding.from_pretrained(embedding_vectors, freeze=False)
        self.max_sentence_length = max_sentence_length

        layers = []
        input_size = self.embedding_dim
        for layer in layers_def:
            lstm = nn.LSTM(input_size,
                           hidden_size=layer.hidden_size,
                           batch_first=True,
                           bidirectional=layer.bidirectional,
                           num_layers=layer.num_layers)
            layers.append(lstm)

            input_size += layer.output_size

        self.lstm_layers = nn.Sequential(*layers)

        # Row max pooling
        self.pooling = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)

        # Output size
        self.output_size = (layers_def[-1].output_size * max_sentence_length) // pool_kernel

    def forward(self, x, l, sort):
        x = self.embedding(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, l,
                                                    enforce_sorted=sort,
                                                    batch_first=True)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x,
                                                      batch_first=True,
                                                      padding_value=self.padding_index)

        for i, layer in enumerate(self.lstm_layers):
            lstm_out, _ = self.forward_lstm_layer(layer, x, l, sort)
            if i != len(self.lstm_layers) - 1:
                x = torch.cat((lstm_out, x), dim=2)
            else:
                x = lstm_out

        x = self.pooling(x)

        x = x.view(len(x), -1)

        pad = torch.zeros(len(x), self.output_size).to(self.device)
        pad[:, :x.shape[1]] = x

        return pad

    def forward_lstm_layer(self, layer, x, l, sort):
        lstm_in = torch.nn.utils.rnn.pack_padded_sequence(x, l,
                                                          enforce_sorted=sort,
                                                          batch_first=True)

        lstm_out, (hn, cn) = layer(lstm_in)

        lstm_out_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                                    batch_first=True,
                                                                    padding_value=self.padding_index)

        return lstm_out_unpack, (hn, cn)
