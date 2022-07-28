# Window data augmentation
# Original code Copyright © 2022 Nathan M. White
# Other code as indicated has copyright held by their respective owners.

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"


# TODO: finish imports
import tensorflow as tf

from .util import wer


# From Tensorflow tutorial site
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
 
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
  
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
 
 
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
  
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
 
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
 
  
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))
# end Tensorflow tutorial code


# TODO: the evaluate_test code for character-level is different than otherwise
# determine why and replace as necessary
# it may be that this includes my attempt to incorporate Levenshtein head approach
#  if so, need to remove
def evaluate_test(test_data):
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
        for i in range(OUTPUT_LEN):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                input_, output)
            #print(inp)
            predictions, attention_weights = transformer(input_,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)
            #print(type(predictions))
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)
            #print(predicted_id)

            output = tf.concat([output, predicted_id], axis=-1)

            if predicted_id == total_vocab['<end>']:
              break

        #print(output)
        scorable_output = tf.squeeze(output, axis=0)
        print("Actual: {}".format(' '.join(inv_total_vocab[i] for i in tar.numpy())))
        print("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in scorable_output.numpy())))

        pad_value = total_vocab['<pad>']
        start_value = total_vocab['<start>']
        end_value = total_vocab['<end>']

        target_scorable = np.array([i for i in tar.numpy() if i not in [pad_value, start_value, end_value]])
        #print("target:", target_scorable)
        pred_scorable = np.array([i for i in scorable_output.numpy() if i not in [pad_value, start_value, end_value]])
        #print("predicted:", pred_scorable)

 # TODO: check especially from here
        def get_word_sequences(pred_array):
            word_sequences = []
            word = []
            for item in pred_array:
                if item == total_vocab[' ']:
                    word_sequences.append(word)
                else:
                    word.append(item)
            if len(word) > 0:
                word_sequences.append(word)
            return word_sequences

        def get_best_words_levenshtein(pred_char_num_sequences):
            results = []
            for item in pred_char_num_sequences:
                best_idx = np.argmin(np.array([levenshtein(item, i) for i in vocab_sequences]))
                results.append(vocab_sequences[best_idx])
            return results

        pred_word_sequences = get_word_sequences(pred_scorable)
        pred_best_levenshtein = get_best_words_levenshtein(pred_word_sequences)

        pred_scorable = np.array([c for line in pred_best_levenshtein for c in line + [total_vocab[' ']]][:-1])
 # to here

        print("Actual: {}".format(' '.join(inv_total_vocab[i] for i in target_scorable)))
        print("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in pred_scorable)))

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
# TODO: instantiate model
# TODO: load data: train_dataset and test_dataset
# TODO: ensure train_dataset gives access to sequence correctly
# TODO: restructure to have print to log file

# begin Tensorflow tutorial code
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> chars, tar -> words
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                      train_loss.result(), 
                                                      train_accuracy.result()))
 # end Tensorflow tutorial code

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
  
    results = evaluate_test(test_dataset)
    # TODO: write results to log file
