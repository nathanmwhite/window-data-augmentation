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

from sents_util import retrieve_sents


def prepare_windowed_data(datapath):
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


def get_vocabulary(path='ywl_vocab.csv'):
    vocab_df = pd.read_csv(path, delimiter='\t')
    vocab = vocab_df['word'].dropna().to_numpy().tolist()
    return vocab


# TODO: finish implementation
def generate_windowed_input_output(data, use_char=True):
    data_out = {}
    types = ['base', 'LA', 'RA', 'S3', 'S5', 'S7', 'S9', 'S11', 'S13', 'test']
    for type in types:
        input_, output_ = join_sents(data[type])
        if use_char == True:
            input_tokens = [[c for c in line] for line in input_]
            output_tokens = [[c for c in line] for line in output_]
        else:
            raise NotImplementedError
        tokens = (input_tokens, output_tokens)
        data_out[type] = tokens
        
    vocab = get_vocabulary()
    
    



