# machine_learning_group_project
## Colab Experiment 

This project was developed and executed in **Google Colab** in order to validate the full
machine learning pipeline in a controlled and reproducible environment.

---

## Project Goal
The goal of this project is to demonstrate a **complete machine learning pipeline**
for the task of **code summarization**.

The focus is placed on:
- clear project structure
- reproducibility
- separation of concerns
- end-to-end workflow (data loading → training → evaluation → inference)

rather than achieving state-of-the-art performance.

---

## Task Description
Given a code snippet as input, the model generates a short natural language
summary describing the functionality of the code.

This task is formulated as a **sequence-to-sequence learning problem** and is
implemented using transformer-based architectures.

---

## Project Structure

---

## Environment
- **Platform:** Google Colab  
- **Hardware:** GPU (T4)  
- **Frameworks:** PyTorch, HuggingFace Transformers  

---

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt

Training
The training script performs:
dataset loading
tokenizer initialization
model training
checkpoint saving
python scripts/train.py
Trained models are stored in the models/ directory.
Evaluation
The evaluation script measures model performance on held-out data.
python scripts/evaluate.py
Evaluation metrics are printed to the console.
Inference
To generate summaries for new code samples:
python scripts/infer.py
Example
Input:
def add(a, b):
    return a + b
Generated summary:
Function that returns the sum of two numbers.
---

## Limitations & Notes

- The goal of this project is educational and architectural rather than achieving
  state-of-the-art performance.
- The dataset used is limited in size and intended for demonstration purposes.
- Training was performed for a small number of epochs to fit within Colab constraints.
- The focus is on clarity, reproducibility, and separation of concerns.

## Configuration

All key model, tokenization, training, and inference parameters are defined in a dedicated configuration file (`config.yaml`).
This design improves reproducibility and allows easy experimentation without modifying the source code.

The configuration includes:
- model architecture parameters
- tokenization limits
- training hyperparameters
- generation settings

## Evaluation

The trained model was evaluated on a held-out test set.

Metric:
- Test Loss (CrossEntropy): **4.54**

The result confirms that the model successfully learned to generate short textual summaries
and that the full training–evaluation–inference pipeline works as expected.

## Inference

The inference script demonstrates qualitative results by comparing:

- input code snippets
- model-predicted summaries
- reference summaries

Example output is produced by running:

```bash
python scripts/infer.py

