import torch
import torch.nn as nn


class MusicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2, device='cpu'):
        super(MusicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # transform every token in a word embedding of embeddig_dim dimension
        
        self.lstm = nn.LSTM(input_size= embedding_dim, 
                            hidden_size= hidden_dim, 
                            num_layers= num_layers, 
                            dropout= dropout, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, hidden):
        """Converts token in embeddings,
        feed the LSTM model,

        Args:
            x (torch.array): token sequence
            hidden (tuple): hidden_state (h_0, c_0)

        Returns:
            x (torch.array): output, predictions
            hidden (tuple): hidden state for next iteration (h_n_c_n), h_n -> [num_layers, batch_size, hidden_dim], c_n -> [num_layers, batch_size, hidden_dim]
        """
        x = self.embedding(x) # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.contiguous().view(-1, x.size(2))  # Reshape for the fully connected layer
        out = self.fc(x)
        return out, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state for each batch.
        In this case, we reinitialize each batch since we want that batches to not interfere with each other
        In order:
            - take an existing parameter to ensure compatibility
            - creates two tensors of zero
            - returns the tuple (h_0, c_0) for the forward methos

        Args:
            batch_size (int)

        Returns:
            torch.vector: hidden_state (h_0, c_0)
        """
        weight = next(self.parameters()).data
        return (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device=self.device),
                weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device=self.device))