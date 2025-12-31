# Model Evaluation Report: local_run_v1

**Date**: December 31, 2025  
**Model**: local_run_v1  
**Training Duration**: 2 epochs (~8 hours on Mac M-series GPU)  
**Dataset**: data/dataset.csv (57,142 Python code-summary pairs)

---

## Training Summary

### Configuration
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Layers**: 1
- **Dropout**: 0.5
- **Batch Size**: 16
- **Optimizer**: Adam (lr=0.001)
- **Device**: MPS (Metal Performance Shaders)

### Training Progress
| Epoch | Train Loss | Train PPL | Valid Loss | Valid PPL |
|-------|-----------|-----------|------------|-----------|
| 1     | 5.518     | 249.08    | 5.668      | 289.41    |
| 2     | 4.653     | 104.91    | 5.595      | 269.11    |

**Key Observations**:
- Loss decreased significantly from Epoch 1 to 2 (5.52 → 4.65)
- Validation loss also improved (5.67 → 5.60)
- Model is learning and not overfitting

---

## Quantitative Metrics

Evaluated on 100 test samples from the dataset.

### BLEU Score
- **BLEU**: 3.08e-79 (essentially 0)
  - *Interpretation*: Very low n-gram overlap. Model struggles with exact phrase matching.

### ROUGE Scores

| Metric   | Recall | Precision | F1-Score |
|----------|--------|-----------|----------|
| ROUGE-1  | 0.158  | 0.307     | **0.190** |
| ROUGE-2  | 0.016  | 0.044     | **0.022** |
| ROUGE-L  | 0.147  | 0.278     | **0.174** |

**Key Observations**:
- **ROUGE-1 F1 (0.19)**: About 19% of individual words match the reference
- **ROUGE-2 F1 (0.02)**: Very low bigram overlap - model doesn't capture phrase structure well yet
- **ROUGE-L F1 (0.17)**: 17% longest common subsequence match
- Higher precision than recall suggests the model generates shorter, more conservative summaries

---

## Qualitative Analysis

### Sample Predictions

#### Example 1: Simple Function
**Code**:
```python
def add_numbers(a, b): 
    return a + b
```

**Generated Summary**:
> 'return the differences between a b highlighted intelligently the differences of b . '

**Analysis**: ❌ Incorrect - confused "add" with "differences"

---

#### Example 2: File Loading
**Code**:
```python
def _load_file(filename):
    fp = open(filename, 'rb')
    source = (fp.read() + '\n')
    ...
```

**Reference**: 
> 'load a Python source file and compile it to byte-code'

**Generated Summary**:
> 'compile the source file and return a python source file . '

**Analysis**: ⚠️ Partially correct - mentions "compile" and "source file" but misses "byte-code"

---

#### Example 3: Module Loading
**Code**:
```python
def _load_module(filename):
    import magics, marshal
    ...
```

**Reference**:
> 'load a module without importing it'

**Generated Summary**:
> 'return a file containing the version of python file . '

**Analysis**: ❌ Incorrect - hallucinated "version" concept

---

#### Example 4: Decompilation
**Code**:
```python
def uncompyle(version, co, out=None, showasm=0, ...):
    ...
```

**Reference**:
> 'diassembles a given code block 'co''

**Generated Summary**:
> 'create a new of the . '

**Analysis**: ❌ Incorrect - generic/incomplete output

---

## Strengths & Weaknesses

### ✅ Strengths
1. **Generates grammatically plausible English** (not random tokens)
2. **Vocabulary coverage** - uses relevant technical terms like "compile", "file", "source"
3. **No catastrophic failures** - always produces output
4. **Fast inference** on Mac GPU

### ❌ Weaknesses
1. **Semantic accuracy is low** - often misunderstands function purpose
2. **Hallucinations** - generates concepts not present in code (e.g., "version", "differences")
3. **Generic outputs** - sometimes produces vague summaries
4. **Poor n-gram matching** - BLEU score near zero
5. **Limited training** - only 2 epochs completed

---

## Recommendations

### For Immediate Improvement
1. **Continue Training**: Run 3-5 more epochs to reduce loss further
2. **Beam Search**: Currently using greedy decoding; beam search may improve quality
3. **Repetition Penalty**: Add penalty to reduce generic/repetitive outputs

### For Long-term Improvement
1. **More Data**: Current dataset is large but model needs more exposure
2. **Larger Model**: Consider 2-layer LSTM or Transformer architecture
3. **Better Tokenization**: Use BPE or WordPiece instead of simple word-level
4. **Fine-tuning**: Pre-train on general text, then fine-tune on code

---

## Conclusion

The `local_run_v1` model demonstrates **proof of concept** for code summarization but requires additional training to be production-ready. With only 2 epochs, it has learned basic patterns (loss decreased from 10.6 → 4.65) but struggles with semantic accuracy.

**Current Status**: 🟡 **Functional but needs improvement**

**Next Steps**: Continue training for 3-5 more epochs and re-evaluate.

---

## Files Generated
- Model checkpoint: `checkpoints/local_run_v1.pt`
- Vocabularies: `checkpoints/local_run_v1_code_vocab.pkl`, `checkpoints/local_run_v1_summary_vocab.pkl`
- Metrics: `results/metrics.json`
- Training log: `training_log.csv`
