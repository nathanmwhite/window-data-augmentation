# prepare_data.py
# Original code Copyright © 2020-2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2020-2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import io

import logging

logging.basicConfig(level=logging.DEBUG, filename='prepare_data.log')

import os

import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.utils.data import Dataset

from sents_util import retrieve_sents

TYPES = ['base', 'LA', 'RA', 'S3', 'S5', 'S7', 'S9', 'S11', 'S13', 'test']


# TODO:
# Where is batch defined with Torch?
# How does decoder_data work? How do you get to the sequential input-outputs in training?
class TorchDataset(Dataset):
    def __init__(self, encoder_data, decoder_data):
        self._encoder_data = encoder_data
        self._decoder_data = decoder_data
        
    def __len__(self):
        return len(self._encoder_data)
    
    def __getitem__(self, idx):
        encoder_input = self._encoder_data[idx]
        decoder_input = self._decoder_data[idx]
        
        return encoder_input, decoder_input


def get_windowed_data(datapath):
    # sliding data could be of many different sizes
    data = []
    left_window_data = []
    right_window_data = []
    sliding_3_data = []
    sliding_5_data = []
    sliding_7_data = []
    sliding_9_data = []
    sliding_11_data = []
    sliding_13_data = []
    test_data = []

    # TODO: rewrite to allow cross-validation
    test_text_indices = [5, 10, 17, 18]
    training_text_indices = [i for i in range(1, 21) if i not in test_text_indices]

    for i in training_text_indices:
        try:
            f = open(os.path.join(datapath, 'text ' + str(i) + ' preprocess.txt'), 'r')
        except OSError as e:
            logging.debug(e)
        item_stream = f.readlines()
        f.close()

        data += retrieve_sents(item_stream, split_comma=True)
        left_window_data += retrieve_sents(item_stream, left_slide=True, split_comma=True)
        right_window_data += retrieve_sents(item_stream, right_slide=True, split_comma=True)
        sliding_3_data += retrieve_sents(item_stream, lim_slide=True, lim=3)
        sliding_5_data += retrieve_sents(item_stream, lim_slide=True, lim=5)
        sliding_7_data += retrieve_sents(item_stream, lim_slide=True, lim=7)
        sliding_9_data += retrieve_sents(item_stream, lim_slide=True, lim=9)
        sliding_11_data += retrieve_sents(item_stream, lim_slide=True, lim=11)
        sliding_13_data += retrieve_sents(item_stream, lim_slide=True, lim=13)

    for i in test_text_indices:
        f = open('text ' + str(i) + ' preprocess.txt', 'r')
        item_stream = f.readlines()
        f.close()

        test_data += retrieve_sents(item_stream, split_comma=True)
        
    return {'base': data,
            'LA': left_window_data,
            'RA': right_window_data,
            'S3': sliding_3_data,
            'S5': sliding_5_data,
            'S7': sliding_7_data,
            'S9': sliding_9_data,
            'S11': sliding_11_data,
            'S13': sliding_13_data,
            'test': test_data}


def get_vocabulary(base_data, path='ywl_vocab.csv', use_char=True):
    vocab_df = pd.read_csv(path, delimiter='\t')
    vocab = vocab_df['word'].dropna().to_numpy().tolist()
    
    # TODO: review for current accuracy and relevance
    charset = set([c for line in base_data[0]+base_data[1] for c in line])
    vocab_letters = list(set([c for line in vocab for c in line]))
    #vocab_letters = vocab
    letterset = list(set([c for line in train_output+test_output for c in line])) + vocab_letters

    complete_set = charset.union(letterset)

    complete_set = ['<pad>'] + list(complete_set) + ['<start>', '<end>']

    total_vocab = {k: i for (i, k) in enumerate(complete_set)}

    inv_total_vocab = {i: k for (k, i) in total_vocab.items()}
    
    vocab_sequences = [[total_vocab[i] for i in word] for word in vocab]
    
    return total_vocab, inv_total_vocab, vocab_sequences


# TODO: review and revise
# add a start and end token to the input and target
def encode(lang1, lang2):
  start_token = total_vocab['<start>']
  end_token = total_vocab['<end>']
  lang1 = [start_token] + [total_vocab[w] for w in lang1] \
           + [end_token]

  lang2 = [start_token] + [total_vocab[w] for w in lang2] \
           + [end_token]
  
  return lang1, lang2


# TODO: review to ensure two different types of combination have not been conflated
def encode_sequences(data):
    sequences = {}
    for type in TYPES:
        sequences[type] = zip(*[encode(a, b) for a, b in zip(data[type][0], data[type][1])])
    
    return sequences


# TODO: finish implementation
def generate_windowed_input_output(data, use_char=True):
    data_out = {}
    for type in TYPES:
        input_, output_ = join_sents(data[type])
        if use_char == True:
            input_tokens = [[c for c in line] for line in input_]
            output_tokens = [[c for c in line] for line in output_]
        else:
            raise NotImplementedError
        tokens = (input_tokens, output_tokens)
        data_out[type] = tokens
        
    vocab, inverse_vocab, vocab_sequences = get_vocabulary()
    
    sequences = encode_sequences(data_out)
    
    in_len, out_len = get_seq_lengths(data_out)
    
    padded_sequences = {}
    for type in TYPES:
        padded_sequences[type][0] = pad_sequences(sequences[type][0],
                                                  maxlen=in_len,
                                                  padding='post',
                                                  value=vocab['<pad>'])
        padded_sequences[type][1] = pad_sequences(sequences[type][1],
                                                  maxlen=out_len,
                                                  padding='post',
                                                  value=vocab['<pad>'])
    
    return padded_sequences


def get_seq_lengths(data):
    input_len = max([len(line) for line in data['base'][0]] + \
                    [len(line) for line in data['test'][0]])
    output_len = max([len(line) for line in data['base'][1]] + \
                     [len(line) for line in data['test'][1]])
    return input_len, output_len

    
def create_tf_dataset(encoder_data, decoder_data):
    enc_numpy = np.asarray(encoder_data, dtype=np.int64)
    enc_dataset = tf.data.Dataset.from_tensor_slices(enc_numpy)

    dec_numpy = np.asarray(decoder_data, dtype=np.int64)
    dec_dataset = tf.data.Dataset.from_tensor_slices(dec_numpy)

    dataset = tf.data.Dataset.zip((enc_dataset, dec_dataset))

    dataset = dataset.cache()
    dataset = dataset.padded_batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def create_pt_dataset(encoder_data, decoder_data):
    pass


def create_train_test_dataset(train_in_padded,
                              train_out_padded,
                              test_in_padded,
                              test_out_padded,
                              tensors='pt'):
    
    if tensors == 'tf':
        # TODO: revisit this: what's going on here?
        train_dataset = create_tf_dataset(train_in_padded, train_out_padded)
        #test_dataset = create_tf_dataset(test_in_padded, test_out_padded)

        enc_test_numpy = np.asarray(test_in_padded, dtype=np.int64)
        enc_test_dataset = tf.data.Dataset.from_tensor_slices(enc_test_numpy)

        dec_test_numpy = np.asarray(test_out_padded, dtype=np.int64)
        dec_test_dataset = tf.data.Dataset.from_tensor_slices(dec_test_numpy)

        test_dataset = tf.data.Dataset.zip((enc_test_dataset, dec_test_dataset))
    elif tensors == 'pt':
        pass
    else:
        raise NotImplementedError
    
    return test_dataset

# TODO: after this, model implementation needs to be done
# LSTM and Transformer have been done previously in Jupyter Notebooks
