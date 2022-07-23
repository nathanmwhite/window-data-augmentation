# model.py
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import torch
from torch.nn import Module
from torch.nn import Embedding, Linear, LSTM, MultiheadAttention, RNN, Softmax, Transformer

# TODO: implement the following:
# 1. Transformer (with four variants)
# 2. BiLSTM
# 3. GRU
# 4. CNN
# 5. If time: Multilingual BERT

# TODO: finish implementation, or supersede with Luong
# Bahdanau et al. provide a description that is lacking in numerous details,
#  precluding a time-effective implementation here
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
        
        # using RNN as single node, stepping through once in each step
        # TODO: determine input_size and hidden_size
        self.rnn_node = RNN(input_size=???, hidden_size=???, num_layers=1, batch_first=True)
        
        # TODO: determine dimensions
        self.g = Linear(???, ???)
        
        # TODO: determine what the hidden state for the RNN should look like, and initialize with zeros
        self.s_previous = None
        
        # TODO: set to start token once known
        self.y_previous = None
    
    def forward(self, s_previous, encoder_hidden_state):
        # encoder hidden state should be a matrix of j vectors representing each encoder hidden state
        # in an imaginary world, it would be this easy
        # TODO: figure out what Bahdanau intends with an fnn with two inputs, and fix this
        # with an RNN, how are you going to get s_i-1?
        # this is too problematic with a separate RNN layer, too time-consuming if restructured inheriting RNN
        # TODO: determine how to best concatenate these
        e = self.fnn.forward(self.s_previous, encoder_hidden_state)
        alpha = self.softmax(e)
        # interm_product should have one dimension as i, and the other as j
        interm_product = torch.matmul(alpha, encoder_hidden_state)
        
        # c should have one dimension as i, and c sums over j
        # TODO: check dimensionality
        c = torch.sum(interm_product, dim=1)
        
        # Bahdanau paper indicates that y_i-1 serves as an input parameter
        #  to this step; this is impractical, and unlikely
        s_i, last_hidden = self.rnn_node(self.s_previous, c)
        
        # Bahdanau does not actually specify what g is here;
        #  Luong et al. (2015) suggests that Bahdanau uses a deep output and a maxout for g
        # Bahdanau indicates that y_i-1, s_i, c_i are inputs into g; does not indicate how
        #  this would be implemented
        g_out = self.g(s_i)
                
        y_i = torch.max(s_i)
        
        self.s_previous = s_i
        
        return y_i
        

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
        
        self.rnn = LSTM(input_size=???,
                        hidden_size=???,
                        num_layers=1,
                        bidirectional=False,
                        batch_first=True)
        
        # TODO: implement hidden state, s, and how to output to y
        
