import torch


class GloveEmbedding:
    PAD_WORD = "<PAD_WORD>"
    PAD_WORD_INDEX = 0
    WORD_NOT_FOUND = "<WNF>"
    WORD_NOT_FOUND_INDEX = 1

    def __init__(self, file_path, embedding_dim):
        self.embed_dim = embedding_dim
        self.w2i = {
            GloveEmbedding.PAD_WORD: GloveEmbedding.PAD_WORD_INDEX,
            GloveEmbedding.WORD_NOT_FOUND: GloveEmbedding.WORD_NOT_FOUND_INDEX
        }
        self.vectors = [[0.0] * embedding_dim,
                        [0.0] * embedding_dim]

        # self.w2i = dict()
        # self.vectors = []

        self.load_file(file_path)

    def load_file(self, file_path):
        model_dict = self.load_glove_model(file_path)

        for word in model_dict:
            vec = model_dict[word]
            self.vectors.append(vec)

            if word not in self.w2i:
                self.w2i[word] = len(self.w2i)

        # self.vectors.append([0.0] * self.embed_dim) # padding
        # self.vectors.append([0.0] * self.embed_dim) # word not found

        self.vectors = torch.tensor(self.vectors, dtype=torch.float)

    def load_glove_model(self, glove_file_path):
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            model = {}
            for i, line in enumerate(f):
                # if i > 1000:
                #     break
                try:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = [float(val) for val in splitLine[1:]]
                    if len(embedding) == self.embed_dim:
                        model[word] = embedding
                except Exception:
                    pass
            print("Done.", len(model), " words loaded!")
        return model

    def __getitem__(self, word):
        try:
            return self.w2i[word]
        except:
            return GloveEmbedding.WORD_NOT_FOUND_INDEX
