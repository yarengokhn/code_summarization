# System Architecture & Logic

This document explains the end-to-end logic of the Code Summarization model, from dataset loading to neural network training.

---

## Table of Contents

- [1. Data Pipeline](#1-data-pipeline)
  - [1.1 Data Loading & Preprocessing](#11-data-loading--preprocessing)
  - [1.2 Tokenization](#12-tokenization)
  - [1.3 Batching & Padding](#13-batching--padding)
- [2. Model Architecture](#2-model-architecture)
  - [2.1 Encoder](#21-encoder)
  - [2.2 Attention Mechanism](#22-attention-mechanism)
  - [2.3 Decoder](#23-decoder)
- [3. Training Loop](#3-training-loop)
- [4. Evaluation](#4-evaluation)
- [5. Optimizations](#5-optimizations)
- [6. Model Saving](#6-model-saving)
- [7. Results](#7-results)

---

## 1. Data Pipeline

The process starts with loading the dataset from Hugging Face.

### 1.1 Data Loading & Preprocessing

**Source:** We load the `Nan-Do/code-search-net-python` dataset from Hugging Face, which contains two columns: `code` (input) and `summary` (target).

**Data Splitting:** The dataset is split into three parts:
- **Train (80%):** Used for model training
- **Validation (10%):** Used for performance monitoring during training
- **Test (10%):** Used for final evaluation

**Normalization:** Text is cleaned (whitespace handling, lowercase conversion) to make it easier for the model to process.

```python
def preprocess_code(text, is_code=True):
    if is_code:
        # Remove comments
        text = re.sub(r'#.*', '', text)
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\s+', ' ', text).strip()
        
        # CamelCase splitting: calculateBLUE -> calculate BLUE
        text = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text)
        
        # Snake_case splitting: calculate_blue -> calculate blue
        text = text.replace('_', ' ')
    else:
        # For summaries, just normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()
```

**Dataset Statistics:**
- Total samples: 455,243
- Train: 364,194 samples
- Validation: 45,524 samples
- Test: 45,525 samples

### 1.2 Tokenization

Computers don't understand text, only numbers. We use the **CodeBERT tokenizer** to split text into "tokens" (words or sub-words) and map them to unique **IDs**.

**Example:**
```
Code: def foo():     → ["def", "foo", "(", ")", ":"] → [4, 12, 5, 6, 7]
Summary: Function that foos → ["function", "that", "foos"] → [99, 10, 55]
```

**Tokenizer Configuration:**
- Model: `microsoft/codebert-base`
- Maximum length: 128 tokens
- Padding: Applied to equalize sequence lengths
- Truncation: Enabled for sequences exceeding max length

```python
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    cleaned_codes = [preprocess_code(c, is_code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_code=False) for s in examples["summary"]]
    
    model_inputs = tokenizer(
        cleaned_codes, 
        text_target=cleaned_summaries,
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    return model_inputs
```

### 1.3 Batching & Padding

We train in small groups called "batches" (e.g., 64 samples at a time). Since sequences have different lengths, we pad them to the same length using a special `<pad>` token.

```python
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
```

---

## 2. Model Architecture

We use a **Sequence-to-Sequence (Seq2Seq)** model with **Attention**, which is standard for translation tasks (Code → Natural Language).

```
┌──────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   Code   │ ──► │ Encoder │ ──► │ Context │ ──► │ Decoder │ ──► Summary
└──────────┘     └─────────┘     │ Vector  │     └─────────┘
                                  └─────────┘
                                       │
                                       ▼
                                  Attention
```

### 2.1 Encoder

**Role:** "Reads" the code.

**Mechanism:**
- Uses **Bidirectional LSTM** (processes input in both forward and backward directions)
- Processes code token by token
- 2-layer LSTM with Dropout (0.3)
- Embedding dimension: 128
- Hidden dimension: 128

**Output:** After reading the entire code, it produces a **Context Vector** (hidden state) - a numerical summary of the code's meaning.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM: Captures context from both beginning and end of code
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
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional layers
        h_reshaped = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        c_reshaped = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        
        hidden_cat = torch.cat((h_reshaped[:, 0, :, :], h_reshaped[:, 1, :, :]), dim=2)
        cell_cat = torch.cat((c_reshaped[:, 0, :, :], c_reshaped[:, 1, :, :]), dim=2)
        
        new_hidden = torch.tanh(self.fc_hidden(hidden_cat))
        new_cell = torch.tanh(self.fc_cell(cell_cat))
        
        return outputs, new_hidden, new_cell
```

### 2.2 Attention Mechanism

**Role:** Allows the decoder to focus on **specific parts** of the source code.

**How it works:**
- When generating each word, the decoder looks at all encoder outputs
- It asks: "Which part of the code is important for this word?"
- Example: When generating "add", it focuses on the `return a + b` part

```python
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim * 2]
        
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Convert to attention weights
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
```

### 2.3 Decoder

**Role:** "Writes" the summary.

**Mechanism:**
- Checks the Context Vector and Attention
- Generates the summary word by word (autoregressive)
- Uses **Teacher Forcing (0.5):** During training, 50% of the time uses the real word, 50% uses its own prediction

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
                 n_layers=2, dropout=0.3):
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
        
        self.fc_out = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim + embedding_dim, 
                                vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, hidden, cell, encoder_outputs):
        input_step = input_step.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_step))
        
        # Calculate attention
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        
        # Context vector
        context_vector = torch.bmm(a, encoder_outputs)
        
        # LSTM input
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Prediction
        prediction_input = torch.cat((output, context_vector, embedded), dim=2)
        prediction = self.fc_out(prediction_input.squeeze(1))
        
        return prediction, hidden, cell, a.squeeze(1)
```

---

## 3. Training Loop

### How does it learn?

**1. Forward Pass:**
- Feed a batch of code to the model
- Model predicts the summary

**2. Loss Calculation:**
- Compare **Prediction vs Actual Summary**
- Use **Cross Entropy Loss**
- If model predicted "subtract" but answer was "add", loss is high

**3. Backward Pass (Backpropagation):**
- Calculate gradients (directions to improve)
- **AdamW Optimizer** updates model weights
- **Gradient Clipping (1.0):** Prevents too large updates

**4. Mixed Precision Training:**
- Uses Float16 to save memory
- GradScaler ensures numerical stability

```python
def train_epoch(model, iterator, optimizer, criterion, clip, scaler):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch['input_ids'].to(DEVICE)
        trg = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda'):
            output = model(src, trg, teacher_forcing_ratio=0.5)
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # Progress indicator every 50 batches
        if (i + 1) % 50 == 0:
            avg_loss = epoch_loss / (i + 1)
            print(f'  Batch {i+1}/{len(iterator)} | Loss: {avg_loss:.3f}')
        
        # Memory cleanup every 100 batches
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    return epoch_loss / len(iterator)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAIN_SIZE` | 200,000 | Training samples |
| `VALID_SIZE` | 20,000 | Validation samples |
| `BATCH_SIZE` | 64 | Samples per batch |
| `NUM_EPOCHS` | 8 | Complete passes through data |
| `LEARNING_RATE` | 0.001 | AdamW learning rate |
| `CLIP` | 1.0 | Gradient clipping threshold |
| `EMB_DIM` | 128 | Embedding dimension |
| `HID_DIM` | 128 | Hidden dimension |

### Checkpointing

- After each epoch, validation loss is calculated
- Best model is automatically saved to `models/best-model.pt`
- Progress is shown every 50 batches

---

## 4. Evaluation

After training, the model is evaluated on the test set.

### Metrics

**1. Loss & Perplexity:**
- **Cross-Entropy Loss:** Measures how "surprised" the model is
- **Perplexity:** More interpretable version of loss (lower = better)

**2. BLEU Score:**
- N-gram based metric
- Measures word overlap between generated and reference summaries
- Range: 0-1 (1 = perfect)

**3. ROUGE Scores:**
- **ROUGE-1:** Single word overlap
- **ROUGE-2:** Two-word phrase overlap
- **ROUGE-L:** Longest common subsequence

### Inference (Greedy Decoding)

```python
def greedy_decode_batch(model, src, tokenizer, device, max_length=128):
    batch_size = src.size(0)
    
    # Encoder
    with autocast():
        encoder_outputs, hidden, cell = model.encoder(src)
    
    # Initialize decoder for each sample
    decoder_input = torch.LongTensor([tokenizer.cls_token_id] * batch_size).to(device)
    
    all_predictions = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    
    for t in range(max_length):
        if all(finished):
            break
        
        with autocast():
            output, hidden, cell, _ = model.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
        
        # Select highest probability token
        top_pred = output.argmax(1)
        
        # Add token for each sample
        for i in range(batch_size):
            if not finished[i]:
                token_id = top_pred[i].item()
                
                # Stop if EOS or SEP token
                if token_id in [tokenizer.sep_token_id, tokenizer.eos_token_id]:
                    finished[i] = True
                else:
                    all_predictions[i].append(token_id)
        
        decoder_input = top_pred
    
    # Convert tokens to text
    decoded_texts = []
    for pred_tokens in all_predictions:
        text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(text)
    
    return decoded_texts
```

---

## 5. Optimizations

### Memory Management

```python
# Clear GPU memory every 100 batches
if (i + 1) % 100 == 0:
    torch.cuda.empty_cache()

# PyTorch memory fragmentation prevention
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

### Mixed Precision Training

- Uses **Float16** for 2x speedup
- **GradScaler** protects against overflow/underflow

### DataLoader Optimizations

```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=2,          # Parallel data loading
    shuffle=True,
    collate_fn=collate_fn   # Custom padding
)
```

---

## 6. Model Saving

The trained model is saved as a Kaggle dataset.

**Saved Files:**
- `model.pt` - Model weights
- `config.json` - Model configuration
- `vocab.json`, `merges.txt` - Tokenizer files
- `evaluation_results.json` - Test results
- `README.md` - Usage instructions

**Loading the Model:**

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/your-dataset-name')

# Rebuild architecture
enc = Encoder(vocab_size, emb_dim, hid_dim).to(device)
dec = AttentionDecoder(vocab_size, emb_dim, hid_dim, hid_dim).to(device)
model = Seq2Seq(enc, dec, device).to(device)

# Load weights
model.load_state_dict(torch.load('/kaggle/input/your-dataset-name/model.pt'))
model.eval()

# Run inference
summary = summarize(code, model, tokenizer, device)
```

---

## 7. Results

### Test Set Performance

```
📊 PERFORMANCE METRICS
────────────────────────────────
Cross-Entropy Loss: 3.216
Perplexity: 24.93
BLEU Score: 0.532
ROUGE-1: 0.738
ROUGE-2: 0.626
ROUGE-L: 0.728
```

### Sample Outputs

| Input Code | Generated Summary | Ground Truth |
|------------|-------------------|--------------|
| `def add(x, y): return x + y` | adds x to y | adds two numbers |
| `def factorial(n): result = 1; for i in range(1, n+1): result *= i; return result` | factorial of a sequence | calculates factorial |
| `def reverse_string(s): return s[::-1]` | reverse string | reverses a string |

### Qualitative Examples

**Example 1:**
```python
# Input
def is_even(n):
    return n % 2 == 0

# Generated: returns true if the number is even
# Ground Truth: checks if number is even
```

**Example 2:**
```python
# Input
def count_words(text):
    return len(text.split())

# Generated: count words in text
# Ground Truth: counts words in text
```

---

## Summary Flow Diagram

```
Dataset (Hugging Face)
    ↓
[Preprocessing & Tokenization]
    ↓
[Train/Val/Test Split]
    ↓
[DataLoader with Batching]
    ↓
┌──────────────────────┐
│   Training Loop      │
│  (8 epochs)          │
│                      │
│  Forward Pass →      │
│  Loss Calculation →  │
│  Backpropagation →   │
│  Optimizer Step      │
└──────────────────────┘
    ↓
[Save Best Model]
    ↓
[Evaluation on Test]
    ↓
[Inference Ready!]
```

---

## Future Improvements

1. **Beam Search:** Better decoding strategy than greedy
2. **Larger Model:** Increase hidden dimensions to 256 or 512
3. **More Data:** Use full dataset (364K samples instead of 200K)
4. **Pre-trained Encoder:** Start with CodeBERT weights
5. **Learning Rate Schedule:** Implement cosine annealing
6. **Repetition Penalty:** Penalize repeated words during generation

---

## Technical Stack

- **Framework:** PyTorch 2.0+
- **Tokenizer:** Hugging Face Transformers (CodeBERT)
- **Dataset:** CodeSearchNet Python
- **Hardware:** NVIDIA Tesla T4 GPU (Kaggle)
- **Mixed Precision:** torch.amp
- **Optimizer:** AdamW

---

**Last Updated:** January 2026  
**Model Type:** Seq2Seq with Bidirectional LSTM + Attention  
**License:** MIT