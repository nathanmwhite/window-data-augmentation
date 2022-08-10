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

logging.basicConfig(level=logging.INFO, filename='run_pt.log')

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

#from torchmetrics import Accuracy

from .model_pt import construct_transformer_model
from .prepare_data import load_dataset
from .util import wer


# train_epoch is from Text Summarization Number Probing repo
#  check to make sure this approach also handles <end> as would be expected
#  also handles the fact that the data has a decoder input and output throughout
def train_epoch(idx, training_data_loader, num_classes, model, loss_function, optimizer, clip_norm):
    batch_loss = 0.0
    continuing_loss = 0.0
    total_loss = 0.0
    
#     accuracy = Accuracy(num_classes=num_classes, mdmc_reduce="global")
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, decoder_in, decoder_out = data_batch
        # This has been superseded by implementation in the training datasets
#         decoder_in = targets[:, :-1]
#         decoder_out = targets[:, 1:]
        
        optimizer.zero_grad()
        
        predictions = model(inputs, decoder_in)
        
        #print(predictions.shape)
        
        # swapaxes as loss_function and torchmetrics require
        #  class dimension to be dim 1
        predictions_axes = torch.swapaxes(predictions, 1, 2)
        
        loss = loss_function(predictions_axes, decoder_out)
        
        loss.backward()
        
        label_int_tensor = decoder_out
        
#         labels_cpu = label_int_tensor.to("cpu")
#         outputs_cpu = predictions_axes.to("cpu")
        
#         batch_accuracy = accuracy(outputs_cpu, labels_cpu)
        
        clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), clip_norm)
                
        optimizer.step()
        # TODO: continue checking and updating for window task here
        continuing_loss += loss.item()
        total_loss += loss.item()
        
        if i % 250 == 249:
            batch_loss = continuing_loss / 250
            n = i + 1
            loss_message = f"-- Batch {n} loss: {batch_loss}"
            print(loss_message)
            logging.info(loss_message)
            continuing_loss = 0.0
            
    return batch_loss, continuing_loss, total_loss #, accuracy.compute()


# Ported from Google Colaboratory-based code and converted to PyTorch
# with modifications
def evaluate(model, device, loss_function, eval_dataloader, total_vocab, output_len):
    def test_bleu_function(real, pred):
        # real and pred here must be numpy
        bleu_1 = corpus_bleu(real, pred, weights=(1.0,))
        bleu_2 = corpus_bleu(real, pred, weights=(0.5, 0.5))
        bleu_3 = corpus_bleu(real, pred, weights=(0.33, 0.33, 0.33))
        bleu_4 = corpus_bleu(real, pred, weights=(0.25, 0.25, 0.25, 0.25))
        return (bleu_1, bleu_2, bleu_3, bleu_4)
    
    def test_accuracy_function(real, pred):
        accuracies = torch.eq(real, pred)
  
        mask = torch.logical_not(torch.eq(real, 0))
        accuracies = torch.logical_and(mask, accuracies)
 
        accuracies = accuracies.to(torch.float32)
        mask = mask.to(torch.float32)
        # TODO: double-check that sum without a dim specification is correct approach
        return torch.sum(accuracies)/torch.sum(mask)
    
    model.eval()
    
    accuracies = []
    bleu_real = []
    bleu_pred = []
    wer_scores = []
   
    pad_idx = total_vocab['<pad>']
    start_idx = total_vocab['<start>']
    end_idx = total_vocab['<end>']

    for i, data_point in enumerate(eval_dataloader):
        inputs, _, targets = data_point
        encoder_in = inputs
        
        decoder_input = [start_idx]
        decoder_input = torch.LongTensor(decoder_input).to(device)
        
        # pad decoder input as output_in
        pad_tensor = torch.zeros((output_len-decoder_input.size(dim=0),), dtype=torch.int64).to(device)
        output_in = torch.cat([decoder_input, pad_tensor], dim=-1)
        output_in = output_in.unsqueeze(0)
        
        scorable_output = None
        # needs access to OUTPUT_LEN
        # TODO: continue here
        for i in range(output_len):
            logging.info(encoder_in.size())
            logging.info(output_in.size())
            predictions = model(encoder_in, output_in)
            
            # TODO: check accuracy of dimensions
            predictions = predictions[:, i:i+1, :]
            
            predicted_id = torch.argmax(predictions, dim=-1).squeeze(0)
            
            decoder_input = torch.cat([decoder_input, predicted_id], dim=-1)
            
            logging.info(predicted_id)
            if predicted_id[0] == end_idx:
                break
        
        scorable_output = decoder_input.squeeze(dim=0)
        
#         print("Actual: {}".format(' '.join(inv_total_vocab[i] for i in targets.numpy())))
#         print("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in scorable_output.numpy())))
   
        # TODO: determine more elegant way to do this
        target_scorable = np.array([i for i in targets.numpy() if i not in [pad_idx, start_idx, end_idx]])
        pred_scorable = np.array([i for i in scorable_output.numpy() if i not in [pad_idx, start_idx, end_idx]])

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

        target_scorable = torch.tensor(target_scorable)
        pred_scorable = torch.tensor(pred_scorable)

        acc = test_accuracy_function(target_scorable, pred_scorable)
        accuracies.append(acc)

        wer_out = wer(target_scorable, pred_scorable)
        wer_scores.append(wer_out/len(target_scorable))

    return np.mean(np.asarray(accuracies)), test_bleu_function(bleu_real, bleu_pred), np.mean(np.asarray(wer_scores))

# Ported from Text Summarization Probing  
class Early_Stopping:
    def __init__(self, min_delta=0.0, patience=10):
        self.num_iterations_elapsed = 0
        self.min_delta = min_delta
        self.patience = patience
        self.early_stopping = False
        self.last_best = None

    def __call__(self, current_loss):
        if self.last_best == None:
            self.last_best = current_loss
        elif self.last_best - current_loss > self.min_delta:
            self.last_best = current_loss
            self.num_iterations_elapsed = 0
        elif self.last_best - current_loss < self.min_delta:
            self.num_iterations_elapsed += 1
            if self.num_iterations_elapsed >= self.patience:
                self.early_stopping = True


# Ported from Tensorflow tutorial code; adapted for PyTorch
def masked_loss_function(pred, real):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    
    mask = torch.logical_not(torch.eq(real, 0))
    loss_result = loss(pred, real)

    mask = mask.to(loss_result.dtype)
    loss_result *= mask

    return torch.sum(loss_result)/torch.sum(mask)
# end ported


# in main loop, create model via construct_transformer_model
# use ported functions to run training and evaluation
if __name__ == '__main__':
# TODO: process command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--early_stopping', type=bool, default=False)
    parser.add_argument('--test_group', type=int, default=0)
    parser.add_argument('--LA', type=bool, default=False)
    parser.add_argument('--RA', type=bool, default=False)
    parser.add_argument('--S3', type=bool, default=False)
    parser.add_argument('--S5', type=bool, default=False)
    parser.add_argument('--S7', type=bool, default=False)
    parser.add_argument('--S9', type=bool, default=False)
    parser.add_argument('--S11', type=bool, default=False)
    parser.add_argument('--S13', type=bool, default=False)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
                                     test_group=args.test_group,
                                     tensors='pt')
    
    # Wrap datasets into DataLoader objects
    train_dataloader = DataLoader(train_dataset, 
                                     batch_size=args.batch_size, 
                                     shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=True)
    
    vocab_size = len(total_vocab)
    
    # instantiate model
    model = construct_transformer_model(vocab_size, args.d_model, encoder_seq_len, decoder_seq_len)
    
    model.to(device)
    
    model.train()
    
    early_stopping = Early_Stopping(patience=args.patience)
    
    # define loss and optimizer
    # from the original Transformer implementations:
    #  one version uses standard categorical_crossentropy,
    #  the other a more complicated one with the mask applied
    # here, I use cross-entropy with masking
    loss_function = masked_loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # training accuracy here considers all positions, including masked positions
    # this ensures that randomly generated elements inside padding penalize
    #  the accuracy metric
    for epoch in range(args.epochs):        
        (batch_loss, 
         continuing_loss,
         total_loss) = train_epoch(epoch, train_dataloader, vocab_size, model, loss_function, optimizer, args.clip_norm)
        
        if epoch % 10 == 0:
            message = f'Epoch {epoch} loss: {total_loss}'
            logging.info(message)
            print(message)
        
        if args.early_stopping:
            early_stopping(total_loss)
            if early_stopping.early_stopping == True:
                message = f'Early stopping of training at epoch {epoch}.'
                logging.info(message)
                break

    model.eval()
    with torch.no_grad():
        results = evaluate(model, device, loss_function, test_dataloader, total_vocab, decoder_seq_len)
        
    hyperparam_set = ('transformer_test',
                      args.d_model,
                      args.batch_size,
                      args.lr,
                      args.epochs)
    message = f"Model hyperparameters: " + ' | '.join(str(w) for w in hyperparam_set)
    logging.info(message)
    message = f"Test raw accuracy: {results[0]}"
    logging.info(message)
    message = f"Test BLEU-4: {results[1][3]}"
    logging.info(message)
    message = f"Test WER: {results[2]}"
    logging.info(message)
