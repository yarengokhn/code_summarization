
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM input = embedding + context_vector
        self.lstm = nn.LSTM(enc_hidden_dim * 2 + embedding_dim, 
                            dec_hidden_dim, 
                            num_layers=n_layers, 
                            batch_first=True, 
                            dropout=dropout if n_layers > 1 else 0)
        
        self.fc_out = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        input_step = input_step.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_step))
        
        # Calculate attention (using top layer hidden state)
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        
        # Context vector: [batch_size, 1, enc_hidden_dim * 2]
        context_vector = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        prediction_input = torch.cat((output, context_vector, embedded), dim=2)
        prediction = self.fc_out(prediction_input.squeeze(1))
        
        return prediction, hidden, cell, a.squeeze(1)