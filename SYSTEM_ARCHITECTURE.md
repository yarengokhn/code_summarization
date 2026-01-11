# System Architecture 

---

## 1. Data Pipeline

### 1.1 Data Loading & Preprocessing

**Source:** We load the `Nan-Do/code-search-net-python` dataset from Hugging Face, which contains two columns: `code` (input) and `summary` (target).

**Data Splitting:** The dataset is split into three parts:
- **Train (80%):** Used for model training
- **Validation (10%):** Used for performance monitoring during training
- **Test (10%):** Used for final evaluation

**Normalization:** Text is cleaned (whitespace handling, lowercase conversion) to make it easier for the model to process.

**Dataset Statistics:**
- Total samples: 455,243
- Train: 364,194 samples
- Validation: 45,524 samples
- Test: 45,525 samples

### 1.2 Tokenization

 We use the **CodeBERT tokenizer** to split text into "tokens" (words or sub-words) and map them to unique **IDs**.


**Tokenizer Configuration:**
- Model: `microsoft/codebert-base`
- Maximum length: 128 tokens
- Padding: Applied to equalize sequence lengths
- Truncation: Enabled for sequences exceeding max length


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

**Mechanism:**
- Uses **Bidirectional LSTM** (processes input in both forward and backward directions)
- Processes code token by token
- 2-layer LSTM with Dropout (0.3)
- Embedding dimension: 128
- Hidden dimension: 128

**Output:** After reading the entire code, it produces a **Context Vector** (hidden state) - a numerical summary of the code's meaning.


### 2.2 Attention Mechanism


**How it works:**
- When generating each word, the decoder looks at all encoder outputs
- It asks: "Which part of the code is important for this word?"
- Example: When generating "add", it focuses on the `return a + b` part


### 2.3 Decoder


**Mechanism:**
- Checks the Context Vector and Attention
- Generates the summary word by word (autoregressive)
- Uses **Teacher Forcing (0.5):** During training, 50% of the time uses the real word, 50% uses its own prediction


## 3. Training Loop

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


### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TRAIN_SIZE` | 150,000 | Training samples |
| `VALID_SIZE` | 15,000 | Validation samples |
| `BATCH_SIZE` | 64 | Samples per batch |
| `NUM_EPOCHS` | 8 | Complete passes through data |
| `LEARNING_RATE` | 0.001 | AdamW learning rate |
| `CLIP` | 1.0 | Gradient clipping threshold |
| `EMB_DIM` | 128 | Embedding dimension |
| `HID_DIM` | 128 | Hidden dimension |


---

## 4. Evaluation

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

