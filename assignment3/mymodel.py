import torch
import torch.nn as nn
import torch.optim as optim

# Define the Elman network class
class MyElmanNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim):
        super(MyElmanNetwork, self).__init__()
        self.hidden_size = hidden_size

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Define the RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Apply word embedding to input
        embedded = self.embedding(input)

        # Forward pass through the RNN layer
        output, _ = self.rnn(embedded)

        # Apply the output layer and return the output
        output = self.output_layer(output)
        return output
