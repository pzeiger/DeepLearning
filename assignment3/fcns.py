# coding: utf-8
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx



def batchify(data, bsz):
    """ Credit for this function:
        https://github.com/pytorch/examples/blob/main/word_language_model/main.py
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, l_seq):
    """ Credit for this function:
        https://github.com/pytorch/examples/blob/main/word_language_model/main.py
    """
    seq_len = min(l_seq, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target



def repackage_hidden(h):
    """ Credit for this function:
        https://github.com/pytorch/examples/blob/main/word_language_model/main.py
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def print_log_head():
    print('-' * 78)
    print('|              |             |       loss        |        perplexity         |')
    print('| end of epoch |     time    |-------------------|---------------------------|')
    print('|              |             |  train  |  valid  |    train    |    valid    |')
    print('-' * 78)
    print('-' * 78)


def load_model(fname):
    model = torch.load(fname)
    
    print_log_head()
    print('|          {:3d} |   {:9.1f}s |   {:5.2f} |   {:5.2f} |    {:8.2f} |    {:8.2f} |'\
          .format(model.epoch, model.time, model.train_loss[-1], model.val_loss[-1], 
                  math.exp(model.train_loss[-1]), math.exp(model.val_loss[-1])))
    print('-' * 78)
    
    return model

