# prepare_data.py
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from tensorflow.keras.layers import Attention, Bidirectional, Dense, Embedding, GRU, Input, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model


def construct_rnn_attention_model(vocab_size, 
                                  hidden_size, 
                                  encoder_seq_len,
                                  decoder_seq_len,
                                  rnn_type='LSTM',
                                  rnn_stack_size=1):
    if rnn_type == 'LSTM':
        RNN_ = LSTM
    elif rnn_type == 'GRU':
        RNN_ = GRU
    else:
        raise NotImplementedError
    
    encoder_input = Input((encoder_seq_len,), name='encoder_input')
    encoder_embedding = Embedding(vocab_size, hidden_size, name='encoder_embedding')
    # Luong et al. only use a stacked unidirectional RNN approach,
    #  in contrast to Bahdanau
    #encoder_bilstm = Bidirectional(RNN_(hidden_size,
    #                                    return_sequences=True,
    #                                    return_states=True),
    #                               name='encoder_bilstm')
#     Luong specifies that a stacking LSTM is used, but does not specify how
#      h and c are passed, if at all
#     The wording suggests that multiple LSTM layers were used, but unclear how
#     Elsewhere, Luong states that an RNN unit is used, which appears to mean
#      a single layer
#     encoder_rnn_list = []
#     for i in range(rnn_stack_size):
#         if i == rnn_stack_size - 1:
#             encoder_rnn = RNN_(hidden_size,
#                                 return_sequences=True,
#                                 return_states=True,
#                                 name='encoder_rnn_'+str(i))
#             encoder_rnn_list.append(encoder_rnn)
#         else:
#             encoder_rnn = RNN_(hidden_size,
#                                 return_sequences=True,
#                                 name='encoder_rnn_'+str(i))
#             encoder_rnn_list.append(encoder_rnn)
    encoder_rnn = RNN_(hidden_size,
                    return_sequences=True,
                    return_states=True,
                    name='encoder_rnn')
    
    decoder_input = Input((decoder_seq_len,), name='decoder_input')
    decoder_embedding = Embedding(vocab_size, hidden_size, name='decoder_embedding')
    decoder_rnn = RNN_(hidden_size,
                        return_sequences=True,
                        return_state=True,
                        name='decoder_rnn')
    decoder_attention = Attention(name='attention_layer')
    decoder_attentional_hidden_state = Dense(hidden_size,
                                             activation='tanh', 
                                             name='attentional_hidden_state')
    decoder_classifier = Dense(vocab_size,
                               activation='softmax', 
                               name='predictive_distribution')
    
    encoder_embed = encoder_embedding(encoder_input)
    #(key_encoded,
    # forward_h,
    # forward_c,
    # backward_h,
    # backward_c) = encoder_bilstm(encoder_embed)
    if rnn_type == 'LSTM':
       (key_encoded,
        encoder_state_h,
        encoder_state_c) = encoder_rnn(encoder_embed)
    else: # GRU
        (key_encoded,
         encoder_state) = encoder_rnn(encoder_embed)
    
    #encoder_state_h = concatenate([forward_h, backward_h])
    #encoder_state_c = concatenate([forward_c, backward_c])
    
    decoder_embed = decoder_embedding(decoder_input)
    
    if rnn_type == 'LSTM':
        (query_encoded,
         decoder_h,
         decoder_c) = decoder_rnn(decoder_embed, initial_state=[encoder_state_h, encoder_state_c])
    else: # GRU
        (query_encoded,
         decoder_state) = decoder_rnn(decoder_embed, initial_state=[encoder_state])
        
    attention_context = decoder_attention([query_encoded, key_encoded])
    attentional_concat = concatenate([query_encoded, attention_context])
    hidden_tilde = decoder_attentional_hidden_state(attentional_concat)
    output = decoder_classifier(hidden_tilde)
    
    model = Model(inputs=[encoder_input, decoder_input], outputs=output)
    
    return model
    
    
# def construct_transformer_model(vocab_size, hidden_size, d_model, h, num_transformer_layers=1):
#     # source: Tensorflow tutorials
#     def positional_encoding(position, d_model):
#         angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                                 np.arange(d_model)[np.newaxis, :],
#                                 d_model)
#         # apply sin to even indices in the array; 2i
#         angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#         # apply cos to odd indices in the array; 2i+1
#         angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#         pos_encoding = angle_rads[np.newaxis, ...]
#         return tf.cast(pos_encoding, dtype=tf.float32)
    
    
    
