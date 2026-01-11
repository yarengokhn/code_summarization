import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        
        # Get encoder outputs
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # First input token (<SOS>)
        input_step = target[:, 0]
        
        # Use list instead of 'outputs' tensor for memory efficiency
        all_outputs = []

        for t in range(1, target_len):
            output, hidden, cell, _ = self.decoder(input_step, hidden, cell, encoder_outputs)
            all_outputs.append(output.unsqueeze(1))  # [batch, 1, vocab_size]
            
            top1 = output.argmax(1) 
            input_step = target[:, t] if random.random() < teacher_forcing_ratio else top1
            
        return torch.cat(all_outputs, dim=1)  # [batch, target_len-1, vocab_size]