# model.py
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from torch.nn import Module
from torch.nn import Embedding, Linear, LSTM, MultiheadAttention, Transformer

# TODO: implement the following:
# 1. Transformer (with four variants)
# 2. BiLSTM
# 3. GRU
# 4. CNN
# 5. If time: Multilingual BERT

# TODO: finish implementation
class BahdanauAttention(Module):
    def __init__(self, s_dim, h_dim):
        super(BahdanauAttention, self).__init__()
        self.s_dim = s_dim
        self.h_dim = h_dim
        
        

class BilstmEncoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_attention_heads):
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
        self.attention = MultiheadAttention(self.hidden_dim, num_attention_heads)
        # TODO: how should encoder hidden state be handled?
#        self.dense_layer = Linear(in_features=self.
#                                  out_features=output_dim)

class BilstmDecoder(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BilstmDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # TODO: implement hidden state, s, and how to output to y
        
