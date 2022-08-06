# Window data augmentation
# Original code Copyright © 2022 Nathan M. White
# Other code as indicated has copyright held by their respective owners.

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import argparse

import logging

logging.basicConfig(level=logging.INFO, filename='rnn_experiment.log')

import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from nltk.translate.bleu_score import corpus_bleu

from .model_tf import construct_rnn_attention_model
from .prepare_data import load_dataset
from .util import wer


# From Tensorflow tutorial site
# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
 
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
  
#     return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

   
# class MaskedAccuracy(MeanMetricWrapper):
#     def __init__(self, mask, name='masked_accuracy', dtype=None):
#         super(MaskedAccuracy, self).__init__(accuracy_function, name, dtype=dtype, mask=mask)
# # original Tf-ported function
# def accuracy_function(real, pred):
#     accuracies = tf.equal(real, tf.argmax(pred, axis=2))
  
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     accuracies = tf.math.logical_and(mask, accuracies)
 
#     accuracies = tf.cast(accuracies, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class MaskedCategoricalAccuracy(MeanMetricWrapper):

    def __init__(self, mask_id, name='masked_categorical_accuracy', dtype=None):
        super(MaskedCategoricalAccuracy, self).__init__(
            masked_categorical_accuracy, name, dtype=dtype, mask_id=mask_id)

# still need to remove <end>
def masked_categorical_accuracy(y_true, y_pred, mask_id):
    #true_ids = K.argmax(y_true, axis=-1)
    true_ids = y_true
    pred_ids = K.argmax(y_pred, axis=-1)
    maskBool = K.not_equal(true_ids, mask_id)
    maskInt64 = K.cast(maskBool, 'int64')
    maskFloatX = K.cast(maskBool, K.floatx())

    count = K.sum(maskFloatX)
    equals = K.equal(true_ids * maskInt64,
                     pred_ids * maskInt64)
    sum = K.sum(K.cast(equals, K.floatx()) * maskFloatX)
    return sum / count

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]


# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')


# # modified from original version for LSTM/GRU
# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar):
#     tar_inp = tar[:, :-1]
#     tar_real = tar[:, 1:]
    
#     # Masks are not relevant for LSTM/GRU architecture
# #     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

#     with tf.GradientTape() as tape:
#         # TODO: replace model call with correct one
#         predictions, _ = transformer(inp, tar_inp, 
#                                      True, 
#                                      enc_padding_mask, 
#                                      combined_mask, 
#                                      dec_padding_mask)
#         loss = loss_function(tar_real, predictions)

#     gradients = tape.gradient(loss, transformer.trainable_variables)    
#     optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

#     train_loss(loss)
#     train_accuracy(accuracy_function(tar_real, predictions))
# end Tensorflow tutorial code


# TODO: the evaluate_test code for character-level is different than otherwise
# determine why and replace as necessary
# it may be that this includes my attempt to incorporate Levenshtein head approach
#  if so, need to remove
def evaluate_test(model, test_data, total_vocab, output_len):
    def test_bleu_function(real, pred):
        # real and pred here must be numpy
        bleu_1 = corpus_bleu(real, pred, weights=(1.0,))
        bleu_2 = corpus_bleu(real, pred, weights=(0.5, 0.5))
        bleu_3 = corpus_bleu(real, pred, weights=(0.33, 0.33, 0.33))
        bleu_4 = corpus_bleu(real, pred, weights=(0.25, 0.25, 0.25, 0.25))
        return (bleu_1, bleu_2, bleu_3, bleu_4)
      
    # begin Tensorflow tutorial code  
    def test_accuracy_function(real, pred):
        accuracies = tf.equal(real, pred)

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
    # end Tensorflow tutorial code
    
    accuracies = []
    bleu_real = []
    bleu_pred = []
    wer_scores = []
    for (inp, tar) in test_data:
        input_ = tf.expand_dims(inp, 0) 
        #print(input_)

        decoder_input = [total_vocab['<start>']]
        output = tf.cast(tf.expand_dims(decoder_input, 0), tf.int64)
        #print(output)
        #print(type(output))

        scorable_output = None
        for i in range(output_len):
            # TODO: replace with appropriate code
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                input_, output)
            #print(inp)
            predictions, attention_weights = model((input_, decoder_input),
                                                   training=False)
            # end TODO
            
            #print(type(predictions))
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            #print(predicted_id)

            output = tf.concat([output, predicted_id], axis=-1)

            if predicted_id == total_vocab['<end>']:
              break

        #print(output)
        scorable_output = tf.squeeze(output, axis=0)
        logging.info("Actual: {}".format(' '.join(inv_total_vocab[i] for i in tar.numpy())))
        logging.info("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in scorable_output.numpy())))

        pad_value = total_vocab['<pad>']
        start_value = total_vocab['<start>']
        end_value = total_vocab['<end>']

        target_scorable = np.array([i for i in tar.numpy() if i not in [pad_value, start_value, end_value]])
        #print("target:", target_scorable)
        pred_scorable = np.array([i for i in scorable_output.numpy() if i not in [pad_value, start_value, end_value]])
        #print("predicted:", pred_scorable)

#  # TODO: check especially from here
#         def get_word_sequences(pred_array):
#             word_sequences = []
#             word = []
#             for item in pred_array:
#                 if item == total_vocab[' ']:
#                     word_sequences.append(word)
#                 else:
#                     word.append(item)
#             if len(word) > 0:
#                 word_sequences.append(word)
#             return word_sequences

#         def get_best_words_levenshtein(pred_char_num_sequences):
#             results = []
#             for item in pred_char_num_sequences:
#                 best_idx = np.argmin(np.array([levenshtein(item, i) for i in vocab_sequences]))
#                 results.append(vocab_sequences[best_idx])
#             return results

#         pred_word_sequences = get_word_sequences(pred_scorable)
#         pred_best_levenshtein = get_best_words_levenshtein(pred_word_sequences)

#         pred_scorable = np.array([c for line in pred_best_levenshtein for c in line + [total_vocab[' ']]][:-1])
#  # to here

        logging.info("Actual: {}".format(' '.join(inv_total_vocab[i] for i in target_scorable)))
        logging.info("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in pred_scorable)))

        bleu_real.append([target_scorable.tolist()])
        bleu_pred.append(pred_scorable.tolist())

        # if target and predicted are different lengths, then need to pad here
        if target_scorable.shape[0] != pred_scorable.shape[0]:
            if target_scorable.shape[0] > pred_scorable.shape[0]:
                diff = target_scorable.shape[0] - pred_scorable.shape[0]
                pred_scorable = np.concatenate((pred_scorable, np.zeros((diff,), dtype=np.int32)))
            else:
                diff = pred_scorable.shape[0] - target_scorable.shape[0]
                target_scorable = np.concatenate((target_scorable, np.zeros((diff,), dtype=np.int32)))

        target_scorable = tf.convert_to_tensor(target_scorable)
        pred_scorable = tf.convert_to_tensor(pred_scorable)

        acc = test_accuracy_function(target_scorable, pred_scorable)
        accuracies.append(acc)

        wer_out = wer(target_scorable, pred_scorable)
        wer_scores.append(wer_out/len(target_scorable))

    return np.mean(np.asarray(accuracies)), test_bleu_function(bleu_real, bleu_pred), np.mean(np.asarray(wer_scores))  

   
if __name__ == '__main__':
# TODO: process command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--LA', type=bool, default=False)
    parser.add_argument('--RA', type=bool, default=False)
    parser.add_argument('--S3', type=bool, default=False)
    parser.add_argument('--S5', type=bool, default=False)
    parser.add_argument('--S7', type=bool, default=False)
    parser.add_argument('--S9', type=bool, default=False)
    parser.add_argument('--S11', type=bool, default=False)
    parser.add_argument('--S13', type=bool, default=False)
    args = parser.parse_args()
    
# load data and vocab: train_dataset and test_dataset, get encoder and decoder size
# TODO: ensure train_dataset gives access to sequence correctly
    data_types = ['base']
    if args.LA:
        data_types.append('LA')
    if args.RA:
        data_types.append('RA')
    if args.S3:
        data_types.append('S3')
    if args.S5:
        data_types.append('S5')
    if args.S7:
        data_types.append('S7')
    if args.S9:
        data_types.append('S9')
    if args.S11:
        data_types.append('S11')
    if args.S13:
        data_types.append('S13')
        
    (train_dataset,
     test_dataset,
     total_vocab,
     encoder_seq_len,
     decoder_seq_len) = load_dataset(args.data_path, 
                                     args.vocab_path, 
                                     window_types=data_types, 
                                     tensors='tf', 
                                     batch_size=args.batch_size)
    vocab_size = len(total_vocab)
    
# instantiate model
    model = construct_rnn_attention_model(vocab_size,
                                          args.hidden_size,
                                          encoder_seq_len,
                                          decoder_seq_len,
                                          rnn_type=args.rnn_type,
                                         )

# begin Tensorflow tutorial code
#     for epoch in range(EPOCHS):
#         start = time.time()

#         train_loss.reset_states()
#         train_accuracy.reset_states()

#         # inp -> chars, tar -> words
#         for (batch, (inp, tar)) in enumerate(train_dataset):
#             train_step(inp, tar)

#             if batch % 50 == 0:
#                 print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#                     epoch + 1, batch, train_loss.result(), train_accuracy.result()))

#         if (epoch + 1) % 5 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
#                                                                  ckpt_save_path))

#         print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
#                                                       train_loss.result(), 
#                                                       train_accuracy.result()))
 # end Tensorflow tutorial code
    
# early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=args.patience)

    pad_token = total_vocab['<pad>']
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[MaskedCategoricalAccuracy(pad_token)])

# train model
# TODO: update input and output parameters
# two options: either split the decoder in/out approach into two entries in the dataset
#  or create a custom model with its own built-in train_step
# other considerations:
#  1. The original approach was for Transformer, which uses masks to deal with the different
#      stages of generation
#  2. LSTM and GRU don't have masking as an automatic feature
#      so unclear what happens if you include Transformer-style data without masking
#  3. One recommended approach for LSTM is to use a single sequence-length window to predict
#      the next in sequence; this is not at all the same as the windowed augmentation approach,
#      but predicting the next item based on a fixed length is what is done
# decision: redo dataset approaches such that the data is generated with a sequence followed by
#  the next prediction as a single item, which will result in a dataset with two inputs, one output
# if there is a mask option to implement, it could be useful, but may be too time-consuming
    history = model.fit(train_dataset, batch_size=args.batch_size,
                        epochs=args.epochs, verbose=1, shuffle=False,
                        workers=3, use_multiprocessing=True,
                        callbacks=[early_stopping]
                       )

       # print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
  
    results = evaluate_test(test_dataset)
        
    hyperparam_set = ('rnn_test',
                      args.rnn_type,
                      args.d_model,
                      args.batch_size,
                      args.lr,
                      args.epochs)
    message = f"Model hyperparameters: " + ' | '.join(str(w) for w in hyperparam_set)
    logging.info(message)
    # write results to log file
    logging.info(f'Base accuracy: {results[0]}')
    logging.info(f'BLEU-4: {results[1][3]}')
    logging.info(f'WER: {results[2]}')
