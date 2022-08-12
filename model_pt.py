# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White
# Other code as indicated has copyright held by their respective owners.

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import logging

logging.basicConfig(level=logging.INFO, filename='run_pt.log')

import numpy as np

import torch
from torch.nn import Embedding, Linear, Softmax, Transformer


# code from Tensorflow tutorial website--with final line replaced
# TODO: consider rewriting this with own code
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # angle_rads are the internal parts of the PE equations to which
    #  sine and cosine are applied
    # np.newaxis takes the values and reorganizes them into a new axis
    # if np.newaxis is last, then it puts all elements in their own rows
    # if np.newaxis is first, it puts all elements in a single innermost row
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.FloatTensor(pos_encoding) # changed


# from TensorFlow, completely rewritten for PyTorch
def create_padding_mask(seq):
    seq_ = torch.eq(seq, 0)
  
    return seq_  # (batch_size, seq_len)


# from TensorFlow, completely rewritten for PyTorch
def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)))
    return torch.eq(mask, True)  # (seq_len, seq_len)
# end code from Tensorflow tutorial website



def construct_transformer_model(vocab_size, d_model, encoder_len, decoder_len, *args, **kwargs):
    # note: torch.nn.Transformer allows multiple layers internally, but it has no clf head
    # it also has no imput embedding layer or position encodings
    # it likewise has no means to generate masks
    # TODO: ensure all items are on the same device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class TransformerPredictor(Transformer):
        def __init__(self, vocab_size, d_model, encoder_len, decoder_len, *args, **kwargs):
            super(TransformerPredictor, self).__init__(d_model=d_model, batch_first=True, *args, **kwargs)
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.encoder_len = encoder_len
            self.decoder_len = decoder_len
            self.encoder_embedding_layer = Embedding(num_embeddings=self.vocab_size,
                                                     embedding_dim=self.d_model,
                                                     padding_idx=0)
            self.decoder_embedding_layer = Embedding(num_embeddings=self.vocab_size,
                                                     embedding_dim=self.d_model,
                                                     padding_idx=0)
            self.encoder_pos_encoding = positional_encoding(self.encoder_len, self.d_model).to(device)
            self.decoder_pos_encoding = positional_encoding(self.decoder_len, self.d_model).to(device)
            self.final_layer = Linear(self.d_model, self.vocab_size)
            self.softmax = Softmax(dim=-1)
            
        def forward(self, source, target):
            source_embedded = self.encoder_embedding_layer(source)
            target_embedded = self.decoder_embedding_layer(target)
            # (N, S, E), batch_first=True, 
            # S is the source sequence length, T is the target sequence length, 
            # N is the batch size, E is the feature number
            # add position encoder
            encoder_length = source_embedded.size(dim=1)
            decoder_length = target_embedded.size(dim=1)
            # original has a sqrt step, not implemented here
            # TODO: review whether sqrt step is part of original architecture
            encoder_positional_encoding = self.encoder_pos_encoding[:, :encoder_length, :]
            decoder_positional_encoding = self.decoder_pos_encoding[:, :decoder_length, :]
            source_embedded += encoder_positional_encoding
            target_embedded += decoder_positional_encoding
            
            # generate masks
            # Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked positions. 
            # If a BoolTensor is provided, positions with True are not allowed to attend while False values will be unchanged.
            # TODO: confirm that this should be receiving a maximum combination of the two masks as in the original TensorFlow
            #  version; currently coded to do so
            
            # [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by the attention.
            # If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
            
            # "additive mask" prevents leftward information flow
            # "padding mask" is self-explanatory
            
            
            # Based on logic from Tensorflow tutorial website
            target_lookahead_mask = create_look_ahead_mask(target.size(dim=1)).to(device)
            
            #target_lookahead_mask = self.generate_square_subsequent_mask(target.size(dim=1))
            
            source_padding_mask = create_padding_mask(source).to(device)
            target_padding_mask = create_padding_mask(target).to(device)
                   
            # output is (batch_size, target_seq_len, num_features)
            #logging.info(source_embedded.size())
            #logging.info(target_embedded.size())
            processed = super().forward(source_embedded, 
                                        target_embedded, 
                                        tgt_mask=target_lookahead_mask,
                                        src_key_padding_mask=source_padding_mask,
                                        tgt_key_padding_mask=target_padding_mask)
            # TODO: finish implementation here, check and test all
            out = self.final_layer(processed)
            y_hat = self.softmax(out)
            
            return y_hat
    
    return TransformerPredictor(vocab_size, d_model, encoder_len, decoder_len, *args, **kwargs)
