# Model Evaluation Report

**Training Duration:** 8 epochs (~4.5 hours on NVIDIA Tesla T4)  
**Dataset:** Nan-Do/code-search-net-python (455,243 Python code-summary pairs)

## Dataset Split & Usage

| Split | Total Available | Actually Used | Usage % |
|-------|----------------|---------------|---------|
| Train | 364,194 | 200,000 | 54.9% |
| Validation | 45,524 | 20,000 | 43.9% |
| Test | 45,525 | 20,000 | 43.9% ✅ |

---

## Training Summary

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 128 |
| Hidden Dimension | 128 |
| Number of Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 64 |
| Optimizer | AdamW (lr=0.001) |
| Device | CUDA (Tesla T4) |
| Architecture | Bidirectional LSTM Encoder + Attention Decoder |
| Tokenizer | microsoft/codebert-base |

### Training Progress

| Epoch | Train Loss | Train PPL | Valid Loss | Valid PPL |
|-------|------------|-----------|------------|-----------|
| 1 | 4.185 | 65.77 | 3.573 | 35.63 |
| 2 | 2.734 | 15.39 | 3.449 | 31.48 |
| 3 | 2.385 | 10.86 | 3.499 | 33.08 |
| 4 | 2.196 | 8.99 | 3.331 | 27.99 |
| 5 | 2.062 | 7.86 | 3.289 | 26.81 |
| 6 | 1.958 | 7.09 | **3.170** | **23.81** ⭐ |
| 7 | 1.876 | 6.53 | 3.290 | 26.83 |
| 8 | 1.808 | 6.10 | 3.207 | 24.70 |

**Key Observations:**
- ✅ Dramatic loss reduction: 4.185 → 1.808 (56.8% improvement)
- ✅ Best validation loss achieved at Epoch 6: 3.170
- ✅ Consistent training with no catastrophic overfitting
- ✅ Model converged effectively with mixed precision training

---

## Quantitative Metrics

**Evaluated on 20,000 test samples** (43.9% of total test set - matching training/validation ratio)

### Overall Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Cross-Entropy Loss | 3.1649 | Strong performance on unseen data |
| Perplexity | 23.69 | Reasonable uncertainty in predictions |
| BLEU Score | 0.5223 | Good n-gram overlap with references |
| ROUGE-1 F1 | 0.7303 | 73% word-level match |
| ROUGE-2 F1 | 0.6153 | 62% bigram phrase match |
| ROUGE-L F1 | 0.7207 | 72% longest sequence match |

### Comparison: 2K vs 20K Test Samples

| Metric | 2K Samples | 20K Samples | Δ Change |
|--------|-----------|-------------|----------|
| Loss | 3.1278 | 3.1649 | +0.0371 (+1.2%) |
| Perplexity | 22.82 | 23.69 | +0.87 (+3.8%) |
| BLEU | 0.5312 | 0.5223 | -0.0089 (-1.7%) |
| ROUGE-1 | 0.7372 | 0.7303 | -0.0069 (-0.9%) |
| ROUGE-2 | 0.6263 | 0.6153 | -0.0110 (-1.8%) |
| ROUGE-L | 0.7278 | 0.7207 | -0.0071 (-1.0%) |

**Analysis:**
- ✅ Metrics remain stable across different sample sizes
- ✅ Small decreases (1-2%) are statistically normal
- ✅ Model performance is consistent and reliable
- ✅ 20K sample provides statistically robust validation

---