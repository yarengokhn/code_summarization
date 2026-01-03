
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        # Attention mechanism: Calculates relationship between decoder's hidden state and encoder outputs
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

