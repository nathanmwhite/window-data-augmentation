# Text summarization number probing
# Original code Copyright © 2022 Nathan M. White
# Other code as indicated has copyright held by their respective owners.

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

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
    
    return torch.cuda.FloatTensor(pos_encoding) # changed


# from TensorFlow, completely rewritten for PyTorch
def create_padding_mask(seq):
    seq = torch.cuda.FloatTensor(torch.eq(seq, 0))
  
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)
# end code from Tensorflow website

# TODO: implement Transformer model and training support


# TODO: implement
def construct_transformer_model():
    # note: torch.nn.Transformer allows multiple layers internally, but it has no clf head
    # it also has no imput embedding layer or position encodings
    # it likewise has no means to generate masks
    pass
    
