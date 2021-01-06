'''
This is the improved version of main_v1.py
The main improvements are:
1. Now the input is a customizable csv, instead of hard coded in the text
2. Build a customizable training function.
'''
import torch
import torch.nn as nn

import torch.optim as optim
import pandas as pd

from pytorch_lstm_01intro.model_lstm_tagger import LSTMTagger
from pytorch_lstm_01intro.preprocess import seq_to_embedding, seqs_to_dictionary
from pytorch_lstm_01intro.train import train, test

torch.manual_seed(1)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# read in raw data
training_data_raw = pd.read_csv("./train.csv")
# create mappings


#split texts and tags into training data.
texts = [t.split() for t in training_data_raw["text"].tolist()]
tags_list = [t.split() for t in training_data_raw["tag"].tolist()]

training_data = list(zip(texts, tags_list))
word_to_ix, tag_to_ix = seqs_to_dictionary(training_data)

print(training_data)

# Usually 32 or 64 dim. Keeping them small

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(
    word_to_ix), len(tag_to_ix), is_nll_loss=False)

loss_function = nn.NLLLoss() if model.is_nll_loss else nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
# print(model.parameters)

# get embeddings
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
testing_data = "The dog ate the book"

print("tag_scores before training:")
test(testing_data, model, word_to_ix)

train(model, loss_function, training_data, word_to_ix, tag_to_ix, optimizer, epoch=200)

# Expect: 0,
print("tag_scores after training:")
tag_prob = test(testing_data, model, word_to_ix)
print(tag_prob)


