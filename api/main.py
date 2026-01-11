import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Encoder, AttentionDecoder, Seq2Seq
from src.data_preprocessing import preprocess_code

app = FastAPI(title="Code Summarization API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Load model checkpoint
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/model.pkl'))
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Model file not found at {checkpoint_path}. API will fail until model is trained.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        hp = checkpoint['hyperparameters']
        
        # Construct model
        enc = Encoder(hp['input_dim'], hp['emb_dim'], hp['hid_dim']).to(DEVICE)
        dec = AttentionDecoder(hp['output_dim'], hp['emb_dim'], hp['hid_dim'], hp['hid_dim']).to(DEVICE)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("🚀 Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

class SummarizeRequest(BaseModel):
    code: str

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess
        cleaned = preprocess_code(request.code, is_code=True)
        tokens = tokenizer.encode(cleaned, return_tensors='pt', 
                                 truncation=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
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
            return SummarizeResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
