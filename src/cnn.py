import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_size, kernel_width, embeddings, dropout):
        super(CNN, self).__init__()
        self.kernel_width = kernel_width
        self.output_size = output_size

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.conv = nn.Conv1d(embed_dim, output_size, kernel_width, stride=1)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, input):
        # input is of dim: batch x num_samples (1 or 21) x len (60 or 100)
        (batch_size, samples, length) = input.size()
        x = input.view(batch_size * samples, length) # reformat for embedding
        x = self.embedding_layer(x)
        # x is now of dim batch * num_samples x len x 200
        x = torch.transpose(x, 1, 2) # swaps len and 200 to make convolution work
        x = F.tanh(self.dropout(self.conv(x)))
        # x is now of dim batch * num_samples x output_size x (len - kernel_width + 1)
        #x = torch.mean(x, dim = 2)
        #x = torch.squeeze(x, dim = 2)
        #x = x.view(batch_size, samples, self.output_size)
        x = x.view(batch_size, samples, self.output_size, length - self.kernel_width + 1)
        
        return x
