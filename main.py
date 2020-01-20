"""
Implementation for Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
https://arxiv.org/pdf/1708.02312.pdf
"""

import torch
from torch import optim

from model_trainer import ModelTrainer
from performencer import Performencer
from residual_model import ResidualLSTMEncoder, LayersType
from residual_model import LSTMLayer
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
              LSTMLayer(hidden_size=768,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=1024,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers

def get_layers_resid_small():
    layers = [LSTMLayer(hidden_size=30,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=30,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=30,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers


def get_layers_resid():
    layers = [LSTMLayer(hidden_size=600,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=600,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=600,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers


def train_and_eval(embedding, layers, batch_size):
    # Device
    device = get_device()

    # Training parameters
    epochs = 5

    # Train and dev data
    train_file = './data/snli_1.0_train.jsonl'
    train_data = Data(train_file, embedding)
    dev_file = './data/snli_1.0_dev.jsonl'
    dev_data = Data(dev_file, embedding)
    test_file = './data/snli_1.0_test.jsonl'
    test_data = Data(test_file, embedding)

    # Create the model
    model = ResidualLSTMEncoder(embedding_vectors=embedding.vectors,
                                padding_index=train_data.padding_index,
                                layers_def=layers,
                                output_size=len(train_data.c2i),
                                max_sentence_length=Data.MAX_SENTENCE_SIZE,
                                hidden_mlp=800,
                                device=device,
                                layers_type=LayersType.Residual)
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters())

    # Create a model trainer object
    model_trainer = ModelTrainer(net=model,
                                 device=device,
                                 optimizer=optimizer)

    # Train the model
    model_trainer.train(train_data, dev_data,
                        train_log_file='train_resid_1.txt', dev_log_file='dev_resid_1.txt',
                        epochs=epochs, batch_size=batch_size)

    # Save the model
    model_trainer.save_model('./models/model_resid_1')

    # Test the model
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    test_performencer = Performencer(name='Test',
                                     output_size=model.output_size)
    model_trainer.eval(test_loader, test_performencer)
    test_performencer.pinpoint()
    test_performencer.log_to_file('test_resid_1.txt')


def main():
    import sys
    if len(sys.argv) == 2:
        low_mem = sys.argv[1] == '--low_mem'
    else:
        low_mem = False

    if low_mem:
        print("Running train with low memory preset")
        embedding = GloveEmbedding("./models/glove/glove.6B.50d.txt", 50)
        layers = get_layers_resid_small()
        batch_size = 5
    else:
        print("Running train with high memory preset")
        embedding = GloveEmbedding("./models/glove/glove.6B.300d.txt", 300)
        layers = get_layers_resid()
        batch_size = 400

    train_and_eval(embedding, layers, batch_size)


if __name__ == "__main__":
    main()
