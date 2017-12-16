import torch
import torch.nn as nn
import torch.nn.functional as F

# model used for transfer learning

def normalize(x, dim):
    l2 = torch.norm(x, 2, dim).expand_as(x)
    return x / l2.clamp(min = 1e-8)

class Encoder(nn.Module):
    def __init__(self, output_size, embeddings, dropout):
        super(Encoder, self).__init__()
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


        mask = (input != 0).float()
        lens = torch.sum(mask, 2)
        mask = mask / lens.expand_as(mask).clamp(min = 1)
        # mask has dim: batch x num_samples x len
        
        x = input.view(batch_size * samples, length) # reformat for embedding
        x = self.embedding_layer(x)
        x = self.dropout(x)
        # x is now of dim batch * num_samples x len x 200
        output, hn = self.lstm(x) # hidden and cells are zero
        output = normalize(output, 2)
        
        # output is of dim batch * num_samples x len x output_size
        x = torch.transpose(output, 1, 2)
        x = x.contiguous().view(batch_size, samples, self.output_size, length)
        mask = torch.unsqueeze(mask, 2)

        x = x * mask.expand_as(x) # applies mask from earlier
        x = torch.squeeze(torch.sum(x, 3), 3)
        # x now has mean pooling, dim batch x num_samples x output_size

        return x


class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super(DomainClassifier, self).__init__()
        self.W_hidden = nn.Linear(embed_dim, hidden_size)
        self.W_out = nn.Linear(hidden_size, 2)
        
    def forward(self, input):
        hidden = F.relu( self.W_hidden(input))
        out = self.W_out(hidden)
        return out
