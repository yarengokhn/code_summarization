import torch
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import Tokenizer, load_vocab
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from src.inference import InferenceEngine

def main(args):
    # Device selection with MPS support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load Vocabs
    try:
        code_vocab = load_vocab(f"checkpoints/{args.model_name}_code_vocab.pkl")
        summary_vocab = load_vocab(f"checkpoints/{args.model_name}_summary_vocab.pkl")
    except FileNotFoundError:
        try:
            # Fallback to default
            code_vocab = load_vocab("checkpoints/code_vocab.pkl")
            summary_vocab = load_vocab("checkpoints/summary_vocab.pkl")
        except FileNotFoundError:
            print(f"Vocab files for {args.model_name} not found. Please train the model first.")
            return

    code_tokenizer = Tokenizer(is_code=True)
    
    # Init Model
    attn = Attention(args.hid_dim)
    enc = Encoder(len(code_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
    dec = Decoder(len(summary_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    # Load Weights
    checkpoint_path = f"checkpoints/{args.model_name}.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model: {args.model_name}")
    else:
        print(f"Model weights not found at {checkpoint_path}. Please train the model first.")
        return

    engine = InferenceEngine(model, code_vocab, summary_vocab, code_tokenizer, device)
    
    if args.input:
        summary = engine.summarize(args.input)
        print(f"\nGenerated Summary: {summary}")
    else:
        print("Please provide a python function using --input")

if __name__ == "__main__":
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default="local_run_v1", help="Name of the model to use")
    parser.add_argument("--input", type=str, help="Python function to summarize")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'embedding_dim' in config: args.emb_dim = config['embedding_dim']
            if 'hidden_dim' in config: args.hid_dim = config['hidden_dim']
            if 'n_layers' in config: args.n_layers = config['n_layers']
            if 'dropout' in config: args.dropout = config['dropout']
    
    main(args)
