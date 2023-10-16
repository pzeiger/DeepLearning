import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Elman network class
class MyNetwork(nn.Module):
    def __init__(self, embedding_dim=100, hidden_size=100, nlayers=1, vocab_size=10000, rnn_type='RNN'):
        super(MyNetwork, self).__init__()
        
        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define the RNN layer
        if rnn_type == 'RNN':
            # standard Elman RNN
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=nlayers)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=nlayers)
        else:
            raise NotImplementedError()
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.epoch = 0
        self.time = 0
        self.train_loss = []
        self.val_loss = []
    
    
    def forward(self, input, hidden=None):
        # Apply word embedding to input
        embedded = self.embedding(input)

        # Forward pass through the RNN layer
        if hidden is not None:
            output, hidden = self.rnn(embedded, hidden)
        else:
            output, hidden = self.rnn(embedded)

        # Apply the output layer and return the output
        output = self.output_layer(output).view(-1, self.vocab_size)
        
        return output, hidden
    
    
    def init_hidden(self, batch_size, method='zeros'):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            if method == 'zeros':
                return (weight.new_zeros(self.nlayers, batch_size, self.hidden_size),
                        weight.new_zeros(self.nlayers, batch_size, self.hidden_size))
            elif method == 'uniform':
                return (weight.uniform(self.nlayers, batch_size, self.hidden_size),
                        weight.uniform(self.nlayers, batch_size, self.hidden_size))
            else:
                raise NotImplementedError()
        else:
            if method == 'zeros':
                return weight.new_zeros(self.nlayers, batch_size, self.hidden_size)
            elif method == 'uniform':
                return weight.uniform(self.nlayers, batch_size, self.hidden_size)
            else:
                raise NotImplementedError()


