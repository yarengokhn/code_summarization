import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM captures context from both beginning and end of code
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, 
                            dropout=dropout if n_layers > 1 else 0, 
                            bidirectional=True)
        
        # Linear layers to map bidirectional layers to decoder's unidirectional structure
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [n_layers * 2, batch_size, hidden_dim]
        
        # Combine bidirectional layers (per n_layers)
        # Hidden and cell states are converted to [n_layers, batch_size, hidden_dim] format
        h_reshaped = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        c_reshaped = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        
        hidden_cat = torch.cat((h_reshaped[:, 0, :, :], h_reshaped[:, 1, :, :]), dim=2)
        
        cell_cat = torch.cat((c_reshaped[:, 0, :, :], c_reshaped[:, 1, :, :]), dim=2)
        
        new_hidden = torch.tanh(self.fc_hidden(hidden_cat))
        new_cell = torch.tanh(self.fc_cell(cell_cat))
        
        return outputs, new_hidden, new_cell
