import torch
import pandas as pd
import os
import argparse
import sys
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import Tokenizer, load_vocab, load_codesearchnet_dataset
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from src.inference import InferenceEngine
from src.metrics import calculate_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    elif os.path.exists("checkpoints/best_model.pt") and args.model_name == "best_model":
        model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    else:
        print(f"Model weights for {args.model_name} not found.")
        return

    # Load test data
    if args.use_codesearchnet:
        df = load_codesearchnet_dataset(split="test", limit=args.csn_limit)
    elif not os.path.exists(args.test_data):
        print(f"Test data {args.test_data} not found.")
        return
    else:
        df = pd.read_csv(args.test_data)
    
    engine = InferenceEngine(model, code_vocab, summary_vocab, code_tokenizer, device)
    
    hypotheses = []
    references = []
    
    print("Evaluating...")
    for i, row in df.iterrows():
        summary = engine.summarize(row['code'])
        hypotheses.append(summary)
        references.append(row['summary'])
        
        if i < 5:
            print(f"\n--- Sample {i+1} ---")
            print(f"Code: {row['code'][:50]}...")
            print(f"Ref: {row['summary']}")
            print(f"Hyp: {summary}")

    metrics = calculate_metrics(hypotheses, references)
    print("\n--- Final Metrics ---")
    print(json.dumps(metrics, indent=4))

    # Save metrics to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to results/metrics.json")

if __name__ == "__main__":
    import yaml
    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default="best_model")
    parser.add_argument("--test_data", type=str, default="data/test.csv")
    parser.add_argument("--use_codesearchnet", action="store_true", help="Use CodeSearchNet dataset")
    parser.add_argument("--csn_limit", type=int, default=100, help="Number of records to load from CodeSearchNet")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            cli_args = sys.argv
            
            def set_if_not_on_cli(attr, key, config_dict=config):
                if f"--{attr.replace('_', '-')}" not in cli_args and key in config_dict:
                    setattr(args, attr, config_dict[key])

            set_if_not_on_cli("model_name", "model_name")
            set_if_not_on_cli("emb_dim", "embedding_dim")
            set_if_not_on_cli("hid_dim", "hidden_dim")
            set_if_not_on_cli("n_layers", "n_layers")
            set_if_not_on_cli("dropout", "dropout")
            
            if 'data' in config:
                d = config['data']
                if "--use-codesearchnet" not in cli_args and 'use_codesearchnet' in d:
                    args.use_codesearchnet = d['use_codesearchnet']
                set_if_not_on_cli("csn_limit", "csn_limit", d)
    
    main(args)
