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


def get_layers_shortcut_small():
    layers = [LSTMLayer(hidden_size=16,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=16,
                        num_layers=1,
                        bidirectional=True)
              ]

    return layers


def get_layers_shortcut():
    layers = [LSTMLayer(hidden_size=512,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=1024,
                        num_layers=1,
                        bidirectional=True),
              LSTMLayer(hidden_size=2048,
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


def train_and_eval(embedding, layers, batch_size, layers_type):
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
                                layers_type=layers_type)
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # optimizer = optim.Adagrad(model.parameters())

    # Create a model trainer object
    model_trainer = ModelTrainer(net=model,
                                 device=device,
                                 optimizer=optimizer)

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


def main():
    import sys

    if len(sys.argv) != 3:
        print("Please supply the necessary args. Run: python ./main.py [--low_mem|--high_mem] [--residual|--shortcut]")
        return

    if sys.argv[1] != '--low_mem' and sys.argv[1] != '--high_mem':
        print(
            "Invalid argument: %s. Run: python ./main.py [--low_mem|--high_mem] [--residual|--shortcut]" % sys.argv[1])
        return

    if sys.argv[2] != '--residual' and sys.argv[2] != '--shortcut':
        print(
            "Invalid argument: %s. Run: python ./main.py [--low_mem|--high_mem] [--residual|--shortcut]" % sys.argv[2])
        return

    low_mem = sys.argv[1] == '--low_mem'
    residual = sys.argv[2] == '--residual'

    if low_mem:
        print("Running train with low memory preset")
        embedding = GloveEmbedding("./models/glove/glove.6B.50d.txt", 50)
        batch_size = 5
    else:
        print("Running train with high memory preset")
        embedding = GloveEmbedding("./models/glove/glove.6B.300d.txt", 300)
        batch_size = 200

    if residual:
        layers_type = LayersType.Residual
        if low_mem:
            layers = get_layers_resid_small()
        else:
            layers = get_layers_resid()
    else:
        layers_type = LayersType.Shortcut
        if low_mem:
            layers = get_layers_shortcut_small()
        else:
            layers = get_layers_shortcut()

    train_and_eval(embedding, layers, batch_size, layers_type)


if __name__ == "__main__":
    main()
