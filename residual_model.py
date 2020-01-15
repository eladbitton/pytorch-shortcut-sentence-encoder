import torch
from torch import nn

from residual_stacked_encoder import ResidualStackedEncoder


class ResidualLSTMEncoder(nn.Module):
    def __init__(self,
                 max_sentence_length,
                 embedding_vectors,
                 padding_index,
                 layers_def,
                 pool_kernel,
                 hidden_mlp,
                 output_size,
                 device):
        super(ResidualLSTMEncoder, self).__init__()
        self.output_size = output_size
        self.max_sentence_length = max_sentence_length
        self.device = device

        self.stacked_encoder = ResidualStackedEncoder(embedding_vectors=embedding_vectors,
                                                      padding_index=padding_index,
                                                      layers_def=layers_def,
                                                      pool_kernel=pool_kernel,
                                                      max_sentence_length=max_sentence_length,
                                                      device=device)

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

        return self.output_activation(out)
