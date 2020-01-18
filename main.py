"""
Implementation for Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
https://arxiv.org/pdf/1708.02312.pdf
"""

import torch
from torch import optim

from model_trainer import ModelTrainer
from performencer import Performencer
from residual_model import ResidualLSTMEncoder
from residual_stacked_encoder import LSTMLayer
from snli_data import Data
from GloveEmbedding import GloveEmbedding


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



def get_layers_small():
    layers = [LSTMLayer(hidden_size=16,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=16,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers

def get_layers():
    layers = [LSTMLayer(hidden_size=512,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=512,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=512,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers


def train_and_eval():
    # Device
    device = get_device()
    # Embedding
    embedding = GloveEmbedding("./models/glove/glove.6B.300d.txt", 300)

    # Train and dev data
    train_file = './data/snli_1.0_train.jsonl'
    train_data = Data(train_file, embedding)
    dev_file = './data/snli_1.0_dev.jsonl'
    dev_data = Data(dev_file, embedding)
    test_file = './data/snli_1.0_test.jsonl'
    test_data = Data(test_file, embedding)

    # Get the model stacked layers definition
    layers = get_layers()

    # Create the model
    model = ResidualLSTMEncoder(embedding_vectors=embedding.vectors,
                                padding_index=embedding.PAD_WORD_INDEX,
                                layers_def=layers,
                                pool_kernel=2,
                                output_size=len(train_data.c2i),
                                max_sentence_length=Data.MAX_SENTENCE_SIZE,
                                hidden_mlp=1600,
                                device=device)
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters())

    # Create a model trainer object
    model_trainer = ModelTrainer(net=model,
                                 device=device,
                                 optimizer=optimizer)

    batch_size = 50
    epochs = 5

    # Train the model
    model_trainer.train(train_data, dev_data,
                        train_log_file='train_1.txt', dev_log_file='dev_1.txt',
                        epochs=epochs, batch_size=batch_size)

    # Save the model
    model_trainer.save_model('./models/model_1')

    # Test the model
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    test_performencer = Performencer(name='Test',
                                     output_size=model.output_size)
    model_trainer.eval(test_loader, test_performencer)
    test_performencer.pinpoint()
    test_performencer.log_to_file('test_1.txt')


if __name__ == "__main__":
    train_and_eval()
