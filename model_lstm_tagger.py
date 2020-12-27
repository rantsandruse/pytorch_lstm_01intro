'''
This is LSTM POS Tagger
This is almost entirely based on the beginner level tutorial from pytorch.org:
Example: An LSTM for Part-of-speech tagging
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''
from torch import nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
    '''
    Note: here we show two different ways of combining loss functions and results.
    They would result in the same outcome.
    1. NLL Loss + logsoftmax
    2. Cross entropy loss + raw input
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, is_nll_loss=False):
        '''
        embedding_dim: Glove is 300. We are using 6 here.
        hidden_dim: can be anything, usually 32 or 64. We are using 6 here.
        vocab_size: vocabulary size includes an index for padding
        output_size: We need to exclude the index for padding here.
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # In this case, vocab_size is 9, embedding_dim is 6.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_size)
        self.is_nll_loss = is_nll_loss

    def forward(self, sentence):
        # Note: we can implement cross entropy loss in two ways:
        # 1. Use NLL + log softmax:
        # NLLoss = - (input)
        # where input = log_softmax(x)
        # 2. Use Cross entropy directly:
        # CrossEntropyLoss = -log_softmax(input)
        # input = x
        # You can pass raw logits for the latter but need log_softmax for the former.
        embeds = self.word_embeddings(sentence)
        # the dimension should be: seq_len, batch_size, -1
        lstm_out, (h, c) = self.lstm(embeds.view(len(sentence), 1, -1))
        print("lstm out shape:", lstm_out.shape)
        print("hshape:", h.shape)
        print("cshape:", c.shape)
        tag_scores = self.hidden2tag(lstm_out.view(len(sentence), -1))
        if self.is_nll_loss:
            tag_scores = F.log_softmax(tag_scores, dim=1)
        return tag_scores




