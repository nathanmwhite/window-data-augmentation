# model.py
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import torch
from torch.nn import Module
from torch.nn import Embedding, Linear, LSTM, MultiheadAttention, Softmax, Transformer

# TODO: implement the following:
# 1. Transformer (with four variants)
# 2. BiLSTM
# 3. GRU
# 4. CNN
# 5. If time: Multilingual BERT

# TODO: finish implementation, or supersede with Luong
class BahdanauAttention(Module):
    def __init__(self, s_dim, h_dim):
        super(BahdanauAttention, self).__init__()
        self.s_dim = s_dim
        self.h_dim = h_dim
        
        # initial hidden state not defined in paper: use random tensor
        # this initial hidden state is essentially s_0
        # the hidden state should also not be a part of the Attention module itself
        #self.hidden_state = torch.rand(s_dim, dtype=torch.float32)
        
        # input is any number of dimensions, with h_dim as the last dimension
        # output is any number of dimensions, with output_dim as the last dimension
        # remember that i is the time points of y, j the time points of x
        self.fnn = Linear(h_dim, h_dim)
        
        # should apply softmax to j values for each i
        self.softmax = Softmax(dim=1)
    
    def forward(self, s_previous, encoder_hidden_state):
        # in an imaginary world, it would be this easy
        # TODO: figure out what Bahdanau intends with an fnn with two inputs, and fix this
        e = self.fnn.forward(s_previous, encoder_hidden_state)
        alpha = self.softmax(e)
        # TODO: continue implementing
        interm_sum = torch.matmul(alpha, encoder_hidden_state)

class BilstmEncoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BilstmEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # TODO:
        # Does embedding layer accommodate sequence length?
        # Is BiLSTM handling input from embedding layer correctly?
        self.embedding_layer = Embedding(input_dim, self.hidden_dim)
        self.bilstm_layer = LSTM(input_size = self.hidden_dim,
                                 hidden_size = self.hidden_dim,
                                 num_layers = 1,
                                 bidirectional=True,
                                 batch_first=True)
        # BiLSTM output is (batch size, sequence length, 2*hidden_dim
        # TODO: review to determine whether attention layer should be moved
        # TODO: how should encoder hidden state be handled?
#        self.dense_layer = Linear(in_features=self.
#                                  out_features=output_dim)

    def forward(self, input_data):
        embeds = self.embedding_layer(input_data)
        # bilstm_output has shape (batch_size, sequence_length, 2*hidden_size)
        bilstm_output, _, _ = self.bilstm_layer(embeds)
        return bilstm_output


class BilstmDecoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BilstmDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.attention = MultiheadAttention(self.hidden_dim, num_attention_heads)
        
        # TODO: implement hidden state, s, and how to output to y
        
