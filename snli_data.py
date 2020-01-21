import json
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    MAX_SENTENCE_SIZE = 80

    def __init__(self, file_path, embedding):
        self.embedding = embedding
        self.padding_index = 0
        # First sentences and lengths
        self.X_1 = []
        self.L_1 = []

        # Second sentences and lengths
        self.X_2 = []
        self.L_2 = []

        # Correct category
        self.Y = []

        # Category to index
        self.c2i = {
            "neutral": 0,
            "contradiction": 1,
            "entailment": 2
        }

        # Load the data
        self.load_data(file_path)

        print("%s loaded with %d examples" % (file_path, len(self)))

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')

        for line in lines:
            try:
                json_obj = json.loads(line)
            except Exception:
                continue

            sentence_1 = json_obj["sentence1"].lower()
            sentence_2 = json_obj["sentence2"].lower()

            # Get category
            category = json_obj["gold_label"]
            category_index = self.c2i.get(category)
            if category_index is None:
                # Invalid category
                continue
            self.Y.append(category_index)

            se_1, l_s_1 = self.encode_sentence(sentence_1)
            se_2, l_s_2 = self.encode_sentence(sentence_2)

            # Add lengths
            self.L_1.append(l_s_1)
            self.L_2.append(l_s_2)

            se_1 = torch.tensor(se_1, dtype=torch.long)
            se_2 = torch.tensor(se_2, dtype=torch.long)

            # Add to X
            self.X_1.append(se_1)
            self.X_2.append(se_2)

        # Padding
        self.X_1 = self.pad_tensor(self.X_1)
        self.X_2 = self.pad_tensor(self.X_2)

        # Sort by L_1
        self.sort_by_sentence_length()

    def pad_tensor(self, tens):
        return torch.nn.utils.rnn.pad_sequence(tens,
                                               batch_first=True,
                                               padding_value=self.padding_index)

    def encode_sentence(self, sentence):
        emb = [self.embedding[word] for word in sentence]

        if len(emb) > Data.MAX_SENTENCE_SIZE:
            emb = emb[:Data.MAX_SENTENCE_SIZE]
            length = Data.MAX_SENTENCE_SIZE
        elif len(emb) < Data.MAX_SENTENCE_SIZE:
            length = len(emb)
            emb = emb + [self.padding_index] * (Data.MAX_SENTENCE_SIZE - len(emb))
        else:
            length = len(emb)

        return emb, length

    def sort_by_sentence_length(self):
        # Sort by length
        xyl = [[x_1, l_1, x_2, l_2, y] for x_1, l_1, x_2, l_2, y in
               zip(self.X_1, self.L_1, self.X_2, self.L_2, self.Y)]
        xyl.sort(key=lambda x: x[1], reverse=True)

        # Unpack
        self.X_1 = [x_1 for (x_1, l_1, x_2, l_2, y) in xyl]
        self.L_1 = [l_1 for (x_1, l_1, x_2, l_2, y) in xyl]
        self.X_2 = [x_2 for (x_1, l_1, x_2, l_2, y) in xyl]
        self.L_2 = [l_2 for (x_1, l_1, x_2, l_2, y) in xyl]
        self.Y = [y for (x_1, l_1, x_2, l_2, y) in xyl]

    def __getitem__(self, item):
        return (self.X_1[item], self.L_1[item]), (self.X_2[item], self.L_2[item]), self.Y[item]

    def __len__(self):
        return len(self.X_1)
