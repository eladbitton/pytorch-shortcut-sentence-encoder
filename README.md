# Shortcut-Stacked Sentence Encoders for Multi-Domain Inference
A pytorch implementation for the paper "Shortcut-Stacked Sentence Encoders for Multi-Domain Inference".
https://arxiv.org/pdf/1708.02312.pdf

# The Data
The data for this model is the SNLI dataset. 
https://nlp.stanford.edu/projects/snli/

Download the data zip file from here: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
and extract it under `./data` directory

# Embedding model
This model uses a pretrained embedding vector. Specifically the model uses `Glove` embedding. 
You download pretrained word vectors from https://nlp.stanford.edu/projects/glove/.
According to the paper they have used glove.840B.300d but you can use a smaller one for reducing the computation.

Put the word embedding file under the directory `./models/glove/`.

# Dependencies
The only packages used are `pytorch` and `tqdm`.
Tested on Pytorch 1.3 and python 3.5.
Code should work on python 3.5+.


# How to run
python3.5 ./main.py


# Citations
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).


