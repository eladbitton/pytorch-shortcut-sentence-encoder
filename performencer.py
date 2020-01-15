import json

import torch
from torch import nn


class Performencer:
    def __init__(self, name, output_size):
        self.name = name
        self.total = 0
        self.true = 0

        self.loss_sum = 0.0
        self.loss_batches = 0

        self.accuracies = []
        self.losses = []


        self.output_size = output_size
        self.criterion = nn.CrossEntropyLoss()

    def pinpoint(self, should_print=True):
        self.accuracies.append(self.true / self.total)
        self.total = 0
        self.true = 0

        self.losses.append(self.loss_sum / self.loss_batches)
        self.loss_sum = 0.0
        self.loss_batches = 0

        if should_print:
            print("-----------------")
            print("%s Accuracy: %f" % (self.name, self.accuracies[-1]))
            print("%s Loss: %f" % (self.name, self.losses[-1]))
            print()

    def add_acc(self, outputs, label):
        total, count = self.categorical_accuracy(outputs, label)
        self.total += total
        self.true += count

    def add_loss(self, outputs, label):
        # Calc loss
        loss = self.criterion(outputs, label)

        self.loss_sum += loss.item()
        self.loss_batches += 1

        return loss

    def categorical_accuracy(self, outputs, label):
        outputs = outputs.cpu()
        label = label.cpu()

        # Argmax output
        max_preds = outputs.argmax(dim=1, keepdim=True)
        total = 0
        correct = 0
        for p, l in zip(max_preds, label):
            total += 1
            if p == l:
                correct += 1

        return total, correct

    # def categorical_accuracy(self, outputs, label):
    #     outputs = outputs.cpu()
    #     label = label.cpu()
    #
    #     # Argmax output
    #     max_preds = outputs.argmax(dim=1, keepdim=True)
    #
    #     # Filter values
    #     filter_count = 0
    #     if self.filter_value != -1:
    #         for o, l in zip(max_preds.squeeze(1), label):
    #             if o.item() == self.filter_value and l.item() == self.filter_value:
    #                 filter_count += 1
    #
    #     # Get all non pad elements
    #     non_pad_elements = (label != self.tag_pad_index).nonzero()
    #
    #     # Get all correct predictions
    #     correct = max_preds[non_pad_elements].squeeze(1).eq(label[non_pad_elements])
    #
    #     # Calculate accuracy
    #     count = (correct.sum().item() - filter_count)
    #     total = torch.FloatTensor([label[non_pad_elements].shape[0]]).item() - filter_count
    #
    #     return total, count

    def log_to_file(self, file_name):
        json_acc = json.dumps(self.accuracies)

        with open("./accuracies/%s" % file_name, "w+") as file:
            file.write(json_acc)

    #
    # def categorical_accuracy(self, preds, y):
    #     max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    #     non_pad_elements = (y != self.tag_pad_index).nonzero()
    #     correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    #     return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])
    #
    # def get_acc(self, outputs, label):
    #     outputs = outputs.cpu()
    #     label = label.cpu()
    #
    #     outputs = outputs.argmax(dim=1, keepdim=True)
    #
    #     if self.filter_value is not None:
    #         count_filter = sum(1 for a, b in zip(outputs, label)
    #                            if a.item() == self.filter_value and b.item() == self.filter_value)
    #     else:
    #         count_filter = 0
    #
    #     ignore_index_count = sum(1 for b in label if b.item() == self.tag_pad_index)
    #
    #     count = sum(1 for a, b in zip(outputs, label) if a.item() == b.item()) - count_filter
    #     total = len(outputs) - count_filter - ignore_index_count
    #     if total != 0:
    #         return count / total
    #     else:
    #         return 0
