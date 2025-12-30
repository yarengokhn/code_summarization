"""
Training script implementing the full training loop:
data loading, tokenizer initialization, training, and checkpoint saving.
"""
import os
import sys

# Mevcut dosyanın bulunduğu klasörün bir üst dizinini sisteme ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
# Kendi dosyalarından importlar
from models.model import HybridDecoder, HybridEncoder, Seq2Seq
from src.data_loader import (  # data_loader'dan bunları alıyoruz
    tokenized_dataset, tokenizer)
from torch.utils.data import DataLoader

# --- 1. Veriyi Hazırla ---
# Veriyi PyTorch formatına çevir
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
# DataLoader: Veriyi 32'şerli gruplar (batch) halinde modele gönderir
train_iterator = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# --- 2. Hiperparametreler ve Model ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = tokenizer.vocab_size 
OUTPUT_DIM = tokenizer.vocab_size
EMB_DIM = 128 # Bellek dostu olması için biraz düşürdük
HID_DIM = 256

enc = HybridEncoder(INPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
dec = HybridDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
# pad_token_id'yi görmezden gel (hata hesaplarken boşlukları saymasın)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# --- 3. Eğitim Fonksiyonu (Senin yazdığın gibi) ---
def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch['input_ids'].to(DEVICE)
        trg = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# --- 4. Eğitimi Başlat ---
print(f"Eğitim {DEVICE} üzerinde başlıyor...")

for epoch in range(5):
    loss = train_epoch(model, train_iterator, optimizer, criterion)
    print(f"Epoch: {epoch+1} | Loss: {loss:.4f}")

# --- Save trained model (AFTER training) ---
import os
import torch

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pt")
print("Saved: models/model.pt")
