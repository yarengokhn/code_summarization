# AI Code Summarization

This project implementsa code summarization tool that automatically generates natural language descriptions for Python code snippets. It uses a Seq2Seq model with Attention mechanism and Pre-trained CodeBERT embeddings, served via a FastAPI backend and a modern web interface.

## 🚀 Features
- **Deep Learning Model**: Custom Seq2Seq architecture with Bi-LSTM Encoder and Attention-based Decoder.
- **CodeBERT Integration**: Uses Microsoft's CodeBERT tokenizer and embeddings for semantic code understanding.
- **REST API**: Fast and asynchronous API built with FastAPI.
- **Interactive UI**: Clean, responsive web interface for easy testing.
- **Command Line Tool**: Scripts for training and quick inference.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd code_summarization
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have PyTorch installed appropriately for your system (CUDA/MPS/CPU).*

## 🏗️ Project Structure
```
code_summarization/
├── api/               # FastAPI backend
│   └── main.py
├── frontend/          # Web interface
│   ├── index.html
│   └── style.css
├── models/            # Directory for saved models
├── scripts/           # Training and utility scripts
│   ├── train.py       # Main training script
│   ├── summarize.py   # CLI inference script
│   └── evaluate.py    # Model evaluation
├── src/               # Core source code
│   ├── data_loader.py
│   └── data_preprocessing.py
└── requirements.txt
```

## 💻 Usage

### 1. Training the Model
To train the model from scratch using the dataset (default: CodeSearchNet):
```bash
python scripts/train.py
```


### 2. Command Line Inference
To summarize code directly from the terminal:
```bash
python scripts/summarize.py
```
*Follow the interactive prompts to enter code snippets.*

### 3. Running the Web Application
The application consists of a backend API and a frontend.

**Start the Backend:**
```bash
uvicorn api.main:app --reload
```
The API will run at `http://localhost:8000`.

**Open the Frontend:**
Simply open `frontend/index.html` in your web browser. You can drag and drop the file into Chrome/Edge or use a live server extension.

## 🧠 Model Architecture
- **Input**: Source code tokens (truncated/padded to 128 tokens).
- **Encoder**: Bidirectional LSTM using CodeBERT embeddings.
- **Decoder**: LSTM with Attention mechanism to generate natural language summaries.
- **Optimization**: AdamW optimizer with CrossEntropyLoss.

## 📝 Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- FastAPI
- Uvicorn