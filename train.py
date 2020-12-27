'''
Training and testing
'''
import torch
import torch.nn.functional as F
from pytorch_lstm_01intro.preprocess import seq_to_embedding

def train(model, loss_fn, training_data, word_to_ix, tag_to_ix, optimizer, epoch=10):
    '''

    Parameters
    ----------
    model: LSTM Tagger model
    loss_fn: Loss function
    training_data: training data
    word_to_ix
    tag_to_ix
    optimizer
    epoch

    Returns
    -------

    '''
    for epoch in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            print(sentence)
            print(tags)
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
            loss = loss_fn(tag_scores, targets)
            print("loss for epoch ", epoch, ":", loss)
            loss.backward()
            optimizer.step()

def test(testing_data, model, word_to_ix):
    '''
    Run inference on testing data
    Parameters
    ----------
    testing_data: test data
    model: tagger model
    word_to_ix: dictionary mapping word to index

    Returns
    -------

    '''
    with torch.no_grad():
        inputs = seq_to_embedding(testing_data.split(), word_to_ix)
        tag_scores = model(inputs)
        # Now evaluate probabilistic output
        # For either NLL loss or cross entropy los
        if model.is_nll_loss:
            # Use NLL loss
            print("Using NLL Loss:")
            tag_prob = tag_scores.exp()
        else:
            # Use cross entropy loss
            print("Using cross entropy loss")
            tag_prob = F.softmax(tag_scores)

        print(tag_prob)
        return tag_prob
