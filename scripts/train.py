import os
import sys
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Encoder, AttentionDecoder, Seq2Seq
from src.data_loader import tokenized_dataset, tokenizer
from src.data_preprocessing import preprocess_code

# ============ COLLATE FUNCTION ============
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, 
                                    padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, 
                                 padding_value=tokenizer.pad_token_id)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

# ============ TRAINING FUNCTION ============
def train_epoch(model, iterator, optimizer, criterion, clip, scaler):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    
    for i, batch in enumerate(iterator):
        src = batch['input_ids'].to(DEVICE)
        trg = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        # Use enabled=True/False based on device to prevent errors on CPU/MPS if not supported
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
            output = model(src, trg, teacher_forcing_ratio=0.5)
            
            # Loss hesaplama
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
        
        # Backward pass
        if scaler and torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        
        epoch_loss += loss.item()
        
        # Progress göstergesi
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_loss = epoch_loss / (i + 1)
            print(f'  Batch {i+1}/{len(iterator)} | Loss: {avg_loss:.3f} | Time: {elapsed:.1f}s')
        
        # Bellek temizliği
        del output, loss
        if (i + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return epoch_loss / len(iterator)

# ============ EVALUATION FUNCTION ============
def evaluate(model, iterator, criterion):
    """Validation/Test fonksiyonu"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['input_ids'].to(DEVICE)
            trg = batch['labels'].to(DEVICE)
            
            # Mixed precision evaluation
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                output = model(src, trg, teacher_forcing_ratio=0)
                
                # Loss hesaplama
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# ============ MAIN TRAINING ============
# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🚀 Training on {DEVICE}...")

# Cleaning memory 
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Hyperparameters
INPUT_DIM = tokenizer.vocab_size
OUTPUT_DIM = tokenizer.vocab_size
EMB_DIM = 128
HID_DIM = 128
CLIP = 1.0

TRAIN_SIZE = 150000
VALID_SIZE = 15000
NUM_EPOCHS = 10

subset_train = tokenized_dataset['train'].select(range(TRAIN_SIZE))  
subset_valid = tokenized_dataset['validation'].select(range(VALID_SIZE)) 

print(f"Train samples: {len(subset_train)}")
print(f"Valid samples: {len(subset_valid)}")


train_iterator = DataLoader(
    subset_train, 
    batch_size=64, 
    shuffle=True, 
    num_workers=2,
    collate_fn=collate_fn 
)
valid_iterator = DataLoader(
    subset_valid, 
    batch_size=64, 
    shuffle=False, 
    num_workers=2,
    collate_fn=collate_fn 
)

print(f"Batches per epoch: {len(train_iterator)}")

# Model setup
enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
dec = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM).to(DEVICE)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# Optimizer & Criterion
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# Scaler is only useful for CUDA
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

# Model save directory
if not os.path.exists('models'):
    os.makedirs('models')

# Training loop
print("\n🎯 Training starting...\n")
best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f'{"="*60}')
    print(f'📈 Epoch {epoch+1}/{NUM_EPOCHS}')
    print(f'{"="*60}')
    
    # Train
    train_loss = train_epoch(model, train_iterator, optimizer, 
                            criterion, CLIP, scaler)
    
    # Validate
    print(f'\n  🔍 Validating...')
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    # Results
    print(f'\n  ✅ Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint = {
            'state_dict': model.state_dict(),
            'hyperparameters': {
                'input_dim': INPUT_DIM,
                'emb_dim': EMB_DIM,
                'hid_dim': HID_DIM,
                'output_dim': OUTPUT_DIM
            }
        }
        torch.save(checkpoint, 'models/model.pkl')
        print(f'  💾 Best model saved! (Val Loss: {valid_loss:.3f})')
    
    print()  # Empty line

print(f'{"="*60}')
print(f'✨ Training complete!')
print(f'📊 Best validation loss: {best_valid_loss:.3f}')
print(f'{"="*60}')

# Quick inference test
print("\n🧪 Testing inference...")
checkpoint = torch.load('models/model.pkl', map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_code = "def add(x, y): return x + y"
print(f"\nInput code: {test_code}")

with torch.no_grad():
    # Preprocess
    cleaned = preprocess_code(test_code, is_code=True)
    tokens = tokenizer.encode(cleaned, return_tensors='pt', 
                             truncation=True, max_length=128).to(DEVICE)
    
    # Encode
    encoder_outputs, hidden, cell = model.encoder(tokens)
    
    # Decode
    input_step = torch.LongTensor([tokenizer.cls_token_id]).to(DEVICE)
    result_tokens = []
    
    for _ in range(30):
        output, hidden, cell, _ = model.decoder(input_step, hidden, cell, encoder_outputs)
        top1 = output.argmax(1)
        
        if top1.item() == tokenizer.sep_token_id:
            break
        
        result_tokens.append(top1.item())
        input_step = top1
    
    summary = tokenizer.decode(result_tokens, skip_special_tokens=True)
    print(f"Generated summary: {summary}")

print("\n✅ All done!")