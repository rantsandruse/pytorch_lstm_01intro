'''
This is almost entirely based on the beginner level tutorial from pytorch.org:
Example: An LSTM for Part-of-speech tagging
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
The minor modifications are listed below:
1. We used a seqs_to_dictionary function instead of hard code for tag_to_ix dictionary
2. We added an example of inference for non-training data.
3. We added a demonstration of the relationship between NLL loss and cross entropy loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lstm_01intro.model_lstm_tagger import LSTMTagger
from pytorch_lstm_01intro.preprocess import seq_to_embedding, seqs_to_dictionary

def main():
    '''
    Example: An LSTM for Part-of-speech tagging
    Returns: void
    -------
    '''
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    # we remove the hard code of tag_to_ix
    # tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
    word_to_ix, tag_to_ix = seqs_to_dictionary(training_data)

    # Usually 32 or 64 dim. Keeping them small for now
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    # Now we introduce the LSTMTagger model
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = seq_to_embedding(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print("before training:")
        print(tag_scores)

    for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = seq_to_embedding(sentence, word_to_ix)
            targets = seq_to_embedding(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = seq_to_embedding(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)
        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!


main()
