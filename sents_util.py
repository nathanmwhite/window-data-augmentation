# sents_util.py
# Original code Copyright © 2020-2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2020-2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

# TODO: check comments below for handling issues and correct
# See line 32

import re


def retrieve_sents(data_stream, 
                   left_slide=False, 
                   right_slide=False, 
                   lim_slide=False, 
                   split_comma=False, 
                   lim=0):
  sents = []
  current_sent = []
  slides = {}
  for i in range(lim):
    slides[i] = []
  for j, item in enumerate(data_stream):
    current_sent.append(item)
    if lim_slide == True:
      if j < lim:
        # j + 1 because otherwise initial sequences would have len > lim
        for i in range(j + 1):
          slides[i].append(item)
      # currently, this doesn't handle the last several instances correctly
      else:
        # check modulo, and add to sents, and clear dict entry
        current_slide = j % lim
        sents.append(slides[current_slide])
        slides[current_slide] = []
        # then add current word to all dict entries
        for i in range(lim):
          slides[i].append(item)
    else:
      if split_comma:
        split = ['.', ',']
      else:
        split = ['.']
      if item[0] in split:
        if len(current_sent) >= 3:
          # handle aligned versions
          if left_slide == True:
            sent_windows = []
            for i in range(len(current_sent) - 1, 1, -1):
              sent_windows.append(current_sent[:i])
            sents += sent_windows
          if right_slide == True:
            sent_windows = []
            for i in range(1, len(current_sent) - 1):
              sent_windows.append(current_sent[i:])
            sents += sent_windows
 
        # handle the basic current_sent in any case
        if len(current_sent) >= 2:
          sents.append(current_sent)
          current_sent = []
        else:  # clear single sentence of comma or period
          current_sent = []
  
  # append whatever is left over at the end of a text (if no final comma or
  #  period); if lim_slide, only include the last complete unit
  if lim_slide == True:
    for i in range(lim):
      if len(slides[i]) == lim:
        sents.append(slides[i])
  else:
    if current_sent != []:
      sents.append(current_sent)
  return sents


def join_sents(sent_list):
  # sents are sequences of this: '.\t.\t[punc]\t100%\n'
  input_sents = []
  output_sents = []
  for sent in sent_list:
    input_sent = []
    output_sent = []
    for word in sent:
      split_word = word.strip('\n').split('\t')
      if len(split_word) > 1:
        output_sent.append(split_word[1])
      input_sent.append(split_word[0])
    input_ = re.sub('&', '', ' '.join(input_sent))
    input_ = re.sub('% ', '', input_)
    input_ = re.sub('%', '', input_)
    input_sents.append(input_)
    output_sents.append(' '.join(output_sent))
  return input_sents, output_sents
