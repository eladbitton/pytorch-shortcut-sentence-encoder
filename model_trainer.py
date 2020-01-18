from random import shuffle

import torch

from performencer import Performencer
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, net, device, optimizer, print_every=100000):
        self.net = net
        self.net.to(device)
        self.device = device
        self.optimizer = optimizer
        self.print_every = print_every

    def train(self, train_dataset, dev_dataset, train_log_file, dev_log_file, epochs=10, batch_size=500):
        print("Started Training")
        print("Train Examples: %d" % len(train_dataset))
        if dev_dataset:
            print("Dev Examples: %d" % len(dev_dataset))

        if self.print_every / batch_size != self.print_every // batch_size:
            print("Please use a batch size which divides by %d" % self.print_every)
            return

        # Create data loader
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)
        devloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)

        # Create optimizer
        optimizer = self.optimizer

        train_performencer = Performencer(name='Train',
                                          output_size=self.net.output_size)
        dev_performencer = Performencer(name='Dev',
                                        output_size=self.net.output_size)

        dev_every_performencer = Performencer(name='Dev Every 50000',
                                        output_size=self.net.output_size)

        # Unload batches to list of batches
        batches = []
        for data in trainloader:
            batches.append(data)

        num_examples = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch %d#" % epoch)
            shuffle(batches)
            for i, data in tqdm(enumerate(batches, 0)):
                self.net.train()

                # Unpack inputs
                (x_1, l_1), (x_2, l_2), label = data

                # To device
                x_1 = x_1.to(self.device)
                x_2 = x_2.to(self.device)
                label = label.to(self.device)

                # Zero grad
                optimizer.zero_grad()

                # Forward
                outputs = self.net(x_1, l_1, x_2, l_2)

                # Log performance
                train_performencer.add_acc(outputs, label)
                loss = train_performencer.add_loss(outputs, label)

                loss.backward()
                optimizer.step()

                num_examples += batch_size

                if num_examples % self.print_every == 0:
                    self.eval(devloader, dev_every_performencer)
                    dev_every_performencer.pinpoint()

            # Save train accuracy and loss
            train_performencer.pinpoint()

            # Evaluate model on the dev set
            self.eval(devloader, dev_performencer)

            # Save dev accuracy and loss
            dev_performencer.pinpoint()

        train_performencer.log_to_file(train_log_file)
        dev_performencer.log_to_file(dev_log_file)

    def eval(self, dev_loader, performencer):
        self.net.eval()

        for i, data in enumerate(dev_loader, 0):
            # Unload inputs and labels
            (x_1, l_1), (x_2, l_2), label = data

            # To the device
            x_1 = x_1.to(self.device)
            x_2 = x_2.to(self.device)
            label = label.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.net(x_1, l_1, x_2, l_2)

            performencer.add_acc(outputs, label)
            performencer.add_loss(outputs, label)

    def save_model(self, file_path):
        torch.save(self.net.state_dict(), file_path)

    def load_model(self, file_path):
        self.net.load_state_dict(torch.load(file_path))
