# Window data augmentation
# Original code Copyright © 2022 Nathan M. White
# Other code as indicated has copyright held by their respective owners.

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

from .model_pt import construct_transformer_model
from .util import wer

# TODO: write training and evaluation loop as __main__ loop
# initial training and evaluation content should appear first as module-level functions
# For PyTorch, these should be adapted from Text Summarization Number Probing repo
# evaluation must be modified to handle Window Data Augmentation data and task

# train_epoch is from Text Summarization Number Probing repo
# TODO: train_epoch needs to handle the fact that training steps have to be offset
#  decoder_in = tar[:, :-1], decoder_out = tar[:, 1:]
#  check to make sure this approach also handles <end> as would be expected
#  also has to handle the fact that the data has a decoder input and output throughout
def train_epoch(idx, training_data_loader, model, loss_function, optimizer, clip_norm):
    batch_loss = 0.0
    continuing_loss = 0.0
    total_loss = 0.0
    
    for i, data_batch in enumerate(training_data_loader):
        inputs, labels = data_batch
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # testing only
        #print('Outputs size:', outputs.size())
        #print('Labels size:', labels.size())
        
        # TODO: troubleshoot here
        # Using a target size (torch.Size([64])) that is different
        #     to the input size (torch.Size([64, 2])).
        loss = loss_function(outputs, labels)
        
        loss.backward()
        
        clip_grad_norm_(filter(lambda x: x.requires_grad, model.parameters()), clip_norm)
                
        optimizer.step()
        
        continuing_loss += loss.item()
        total_loss += loss.item()
        
        if i % 250 == 249:
            batch_loss = continuing_loss / 250
            n = i + 1
            loss_message = f"-- Batch {n} loss: {batch_loss}"
            print(loss_message)
            logging.info(loss_message)
            continuing_loss = 0.0
            
    return batch_loss, continuing_loss, total_loss
  

# Ported from Google Colaboratory-based code and converted to PyTorch
def evaluate(model, loss_function, eval_dataloader, total_vocab, output_len):
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
        inputs, targets = data_point
        encoder_in = inputs.unsqueeze(0)
        
        # needs total_vocab index
        decoder_input = [total_vocab['<start>']]
        output_in = decoder_input.unsqueeze(0).to(torch.int64)
        
        scorable_output = None
        # needs access to OUTPUT_LEN
        # TODO: continue here
        for i in range(output_len):
            predictions = model(encoder_in, output_in)
            
            # TODO: check accuracy of dimensions
            predictions = predictions[:, -1:, :]
            
            predicted_id = torch.argmax(predictions, dim=-1)
            
            output_in = torch.cat([output_in, predicted_id], dim=-1)
            
            if predicted_id == total_vocab['<end>']:
                break
        
        scorable_output = output_in.squeeze(dim=0)
        
        print("Actual: {}".format(' '.join(inv_total_vocab[i] for i in targets.numpy())))
        print("Predicted: {}".format(' '.join(inv_total_vocab[i] for i in scorable_output.numpy())))
   
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
    pred_scorable = tf.tensor(pred_scorable)
 
    acc = test_accuracy_function(target_scorable, pred_scorable)
    accuracies.append(acc)
 
    wer_out = wer(target_scorable, pred_scorable)
    wer_scores.append(wer_out/len(target_scorable))
  
  return np.mean(np.asarray(accuracies)), test_bleu_function(bleu_real, bleu_pred), np.mean(np.asarray(wer_scores))


# in main loop, create model via construct_transformer_model
# use ported functions to run training and evaluation
if __name__ == '__main__':
    pass
