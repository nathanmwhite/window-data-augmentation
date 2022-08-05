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

import numpy as np

import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.utils.data import Dataset

from .sents_util import join_sents, retrieve_sents

TYPES = ['base', 'LA', 'RA', 'S3', 'S5', 'S7', 'S9', 'S11', 'S13', 'test']

# Note:
# This code has been tested and works as of 30 July 2022.
# Testing with real data shows that it produces outputs that
#  look and behave exactly as expected.
# TODO:
# Remaining issues:
# 1. The character-based approach
#     still does not support certain tags in the data, such as [name].
# 2. The tf dataset format needs to be tested.
# 3. The new version handling decoder_output must be tested.
# 4. Confirm that the unfinished sequences are adequately handled by both pt and
#     tf implementations.
# 5. Confirm that the dimensionality specified for data slices in dataset are correct.


# Batching is handled in PyTorch by the DataLoader, which is implemented in run_pt.py
class TorchDataset(Dataset):
    def __init__(self, encoder_data, decoder_data):
        self._encoder_data = np.asarray(encoder_data, dtype=np.int64)
        self._decoder_data = np.asarray(decoder_data, dtype=np.int64)
        
    def __len__(self):
        return len(self._encoder_data)
    
    def __getitem__(self, idx):
        encoder_input = self._encoder_data[idx]
        decoder_data = self._decoder_data[idx]
        
        # assumes <end> is embedded in sequence
        decoder_input = decoder_data[:-1]
        decoder_output = decoder_data[1:]
        
        return encoder_input, decoder_input, decoder_output


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
            f = open(os.path.join(datapath, 'text ' + str(i) + ' preprocess.txt'), 'r', encoding='utf8')
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
        f = open(os.path.join(datapath, 'text ' + str(i) + ' preprocess.txt'), 'r', encoding='utf8')
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


def get_vocabulary(base_data, test_data, path='ywl_vocab.csv', use_char=True):
    vocab_df = pd.read_csv(path, delimiter='\t')
    vocab = vocab_df['word'].dropna().to_numpy().tolist()
    
    encoder_charset = set([c for line in base_data[0]+test_data[0] for c in line])
    vocab_letters = list(set([c for line in vocab for c in line]))
    #vocab_letters = vocab
    decoder_charset = list(set([c for line in base_data[1]+test_data[1] for c in line if c != '¤'])) + vocab_letters

    complete_set = encoder_charset.union(decoder_charset)

    complete_set = ['<pad>', '<start>', '<end>', '<unk>'] + list(complete_set)

    total_vocab = {k: i for (i, k) in enumerate(complete_set)}

    inv_total_vocab = {i: k for (k, i) in total_vocab.items()}
    
    vocab_sequences = [[total_vocab[i] for i in word] for word in vocab]
    
    return total_vocab, inv_total_vocab, vocab_sequences


# add a start and end token to the input and target
def encode(encoder_line, decoder_line, total_vocab):
    start_token = total_vocab['<start>']
    end_token = total_vocab['<end>']
    unk_token = total_vocab['<unk>']
    encoded_in = [start_token] + [total_vocab[w] for w in encoder_line] \
                  + [end_token]

    encoded_out = [start_token] + [total_vocab[w] if w != '¤' else unk_token for w in decoder_line] \
                   + [end_token]

    return encoded_in, encoded_out


def encode_sequences(data, total_vocab):
    sequences = {}
    for type in TYPES:
        sequences[type] = tuple(zip(*[encode(a, b, total_vocab) for a, b in zip(data[type][0], data[type][1])]))
    
    return sequences


def generate_windowed_input_output(data, vocab_path, use_char=True):
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
        
    vocab, inverse_vocab, vocab_sequences = get_vocabulary(data_out['base'], data_out['test'], path=vocab_path)
    
    sequences = encode_sequences(data_out, vocab)
    
    in_len, out_len = get_seq_lengths(data_out)
    
    padded_sequences = {}
    for type in TYPES:
        padded_in = pad_sequences(sequences[type][0],
                                                  maxlen=in_len,
                                                  padding='post',
                                                  value=vocab['<pad>'])
        padded_out = pad_sequences(sequences[type][1],
                                                  maxlen=out_len,
                                                  padding='post',
                                                  value=vocab['<pad>'])
        padded_sequences[type] = (padded_in, padded_out)
    
    return padded_sequences, vocab, in_len, out_len


# Note: this is intended to produce seq lengths post-encode
def get_seq_lengths(data):
    input_len = max([len(line) for line in data['base'][0]] + \
                    [len(line) for line in data['test'][0]])
    # should not have -1 to handle offset between decoder input and output
    #  as padding is to shared length, and then sliced inside the dataset
    output_len = max([len(line) for line in data['base'][1]] + \
                     [len(line) for line in data['test'][1]])
    return input_len, output_len

    
def create_tf_dataset(encoder_data, decoder_data):
    enc_numpy = np.asarray(encoder_data, dtype=np.int64)
    enc_dataset = tf.data.Dataset.from_tensor_slices(enc_numpy)

    dec_numpy = np.asarray(decoder_data, dtype=np.int64)
    dec_input_dataset = tf.data.Dataset.from_tensor_slices(dec_numpy[:, :-1])
    dec_output_dataset = tf.data.Dataset.from_tensor_slices(dec_numpy[:, 1:])

    dataset = tf.data.Dataset.zip((enc_dataset, dec_input_dataset, dec_output_dataset))

    dataset = dataset.cache()
    dataset = dataset.padded_batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def create_pt_dataset(encoder_data, decoder_data):
    return TorchDataset(encoder_data, decoder_data)


def create_final_dataset(in_padded,
                         out_padded,
                         tensors='pt'):
    
    if tensors == 'tf':
        dataset = create_tf_dataset(in_padded, out_padded)
    elif tensors == 'pt':
        dataset = create_pt_dataset(in_padded, out_padded)
    else:
        raise NotImplementedError
    
    return dataset


def load_dataset(data_path, vocab_path, window_types=['base'], tensors='pt'):
    """
    load_dataset : Loads the dataset according to the specified tensor type.
    @param data_path (str) : the path to the dataset to load
    @param vocab_path (str) : the path to the saved vocabulary
    @param window_types (List[str]) : a list containing the data windows to return
    @param tensors (str) : whether to return tensors as PyTorch (pt) or TensorFlow (tf)
    returns: 1. the training dataset of the specified tensor type
             2. the test dataset of the specified tensor type
             3. the vocabulary as a dictionary
             4. the maximum length of encoder inputs
             5. the maximum length of decoder inputs
    """
    data = get_windowed_data(data_path)

    padded_sequences, vocab, in_len, out_len = generate_windowed_input_output(data, vocab_path)
        
    encoder_data = np.concatenate(tuple(padded_sequences[type][0] for type in window_types))
    decoder_data = np.concatenate(tuple(padded_sequences[type][1] for type in window_types))
    
    train_dataset = create_final_dataset(encoder_data, decoder_data)
    test_dataset = create_final_dataset(*padded_sequences['test'])

    out_len -= 1 # handles decoder in/out slicing in creating dataset
    #print(out_len)
    #print(len(vocab))
    
    return train_dataset, test_dataset, vocab, in_len, out_len
    
