import torch
import sys
import os
from transformers import AutoTokenizer

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Encoder, AttentionDecoder, Seq2Seq
from src.data_preprocessing import preprocess_code

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/best-model.pt'

def load_model():
    """Laws the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Hyperparameters (must match training)
    INPUT_DIM = tokenizer.vocab_size 
    OUTPUT_DIM = tokenizer.vocab_size
    EMB_DIM = 128
    HID_DIM = 128
    
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
    dec = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM).to(DEVICE)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, tokenizer

def summarize(code, model, tokenizer, device, max_len=50):
    """Generates a summary for the given code snippet."""
    cleaned_code = preprocess_code(code, is_code=True)
    
    tokens = tokenizer.encode(cleaned_code, return_tensors='pt', 
                              truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(tokens)
        
        # Start token
        input_step = torch.LongTensor([tokenizer.cls_token_id]).to(device)
        
        summary_tokens = []
        
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_step, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            token_id = top1.item()
            
            if token_id == tokenizer.sep_token_id or token_id == tokenizer.eos_token_id:
                break
                
            summary_tokens.append(token_id)
            input_step = top1
            
    return tokenizer.decode(summary_tokens, skip_special_tokens=True)

# --- Main Execution ---
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            model, tokenizer = load_model()
            print("Model loaded successfully!")
            
            # Example 1: Simple function
            code1 = "def add(x, y): return x + y"
            print(f"\nCode: {code1}")
            print(f"Summary: {summarize(code1, model, tokenizer, DEVICE)}")
            
            # Interactive mode
            print("\nEnter Python code to summarize (type 'q' to quit):")
            while True:
                user_input = input("\n> ")
                if user_input.lower() in ['q', 'quit']:
                    break
                print(f"Summary: {summarize(user_input, model, tokenizer, DEVICE)}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}. Please run train.py first.")