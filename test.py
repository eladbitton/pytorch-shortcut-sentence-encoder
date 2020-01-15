from GloveEmbedding import GloveEmbedding
from snli_data import Data

train_file = './data/snli_1.0_train.jsonl'
embedding = GloveEmbedding("./models/glove/glove.6B.50d.txt", 50)
train_data = Data(train_file, embedding)

print(train_data[0])