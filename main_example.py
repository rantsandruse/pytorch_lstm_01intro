'''
This is the "quick example", based on:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''
import numpy as np

import torch
import torch.nn as nn

# This is the beginning of the original tutorial
torch.manual_seed(1)
# The first implementation
# Initialize inputs as a list of tensors
# passes each element of the input + hidden state to the next step.
# Think of hidden state as memory
# inputs = [torch.rand(1,3) for _ in range(5)]
# for i in inputs:
#     # Step through the sequence one element at a time.
#     # after each step, hidden contains the hidden state.
#     out, hidden = lstm(i.view(1, 1, -1), hidden)
#
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)

lstm = nn.LSTM(3, 3)
# Alternatively, we can just initialize input as a single tensor instead of a list.
inputs = torch.randn(5, 1, 3)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)

# Additional examples are given for understanding of NLL loss and Cross Entropy loss implementation in pytorch
import torch.nn.functional as F
softmax_prob = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
log_softmax_prob = torch.log(softmax_prob)
print("Log softmax probability:", log_softmax_prob)
target = torch.tensor([0,0])

# Note the NLL loss is the negative log at the correct class
# The real NLL Loss is the sum of negative log of the correct class.
#   NLL loss = -1/n (sum(yi dot log(pi))
# Note: (1/n is the average loss here, which is the default pytorch implementation (reduction=mean). What you usually
# see in textbooks/wikipedia is the sum of all losses (i.e. without 1/n) (reduction=sum in pytorch).
# What is being implemented by pytorch, where xi is the input.
# It is taken for granted that xi = log(pi), i.e. it's already gone through the log_softmax transformation when you
# are feeding it into NLL function in pytorch.
#   Pytorch NLL loss = -1/n (sum(yi dot xi))

# In the example below:
# When target = [0,0], both ground truth classifications below to the first class --> y1 = [1,0], y2 = [1,0]
# y1 = [1,0]; log(p1) = [-0.22, -1.61]
# y2 = [1,0]; log(p2) = [-0.51, -0.91]
# Pytorch NLL loss = -1/n (sum(yi dot xi)) = 1/2 * (-0.22*1 - 0.51*1) = 0.36
nll_loss = F.nll_loss(log_softmax_prob, target)
print("NLL loss is:", nll_loss)



