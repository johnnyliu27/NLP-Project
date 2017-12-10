import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, dropout):
        super(LSTM, self).__init__()
        self.output_size = output_size

        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )

        self.lstm = nn.LSTM(embed_dim, output_size // 2, num_layers = 1,
                            bias = True, batch_first = True, dropout = dropout,
                            bidirectional = True)
        self.dropout = nn.Dropout(p = dropout)
                
    def forward(self, input):
        # input is of dim: batch x num_samples (1 or 21) x len (60 or 100)
        (batch_size, samples, length) = input.size()
        x = input.view(batch_size * samples, length) # reformat for embedding
        x = self.embedding_layer(x)
        x = self.dropout(x)
        # x is now of dim batch * num_samples x len x 200
        output, hn = self.lstm(x) # hidden and cells are zero
        # output is of dim batch * num_samples x len x output_size
        x = output.contiguous().view(batch_size, samples, self.output_size, length)
        #x = hn[0].contiguous().view(batch_size, samples, self.output_size)
        return x
