import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import Tokenizer, Vocabulary, preprocess_data, save_vocab, load_codesearchnet_dataset
from src.dataset import get_dataloader
from models.encoder import Encoder
from models.attention import Attention
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from src.train_loop import train_epoch, validate_epoch, epoch_time
import time

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load data
    if args.use_codesearchnet:
        df = load_codesearchnet_dataset(limit=args.csn_limit)
    elif not os.path.exists(args.data_path):
        print(f"Data path {args.data_path} not found. Creating dummy data for demonstration.")
        df = pd.DataFrame({
            "code": ["def hello(): print('hello')", "def add(a, b): return a + b"] * 50,
            "summary": ["prints hello", "adds two numbers"] * 50
        })
    else:
        df = pd.read_csv(args.data_path)
    
    df = preprocess_data(df)
    
    # Init Tokenizers and Vocabs
    code_tokenizer = Tokenizer(is_code=True)
    summary_tokenizer = Tokenizer(is_code=False)
    
    code_vocab = Vocabulary()
    summary_vocab = Vocabulary()
    
    code_vocab.build_vocabulary(df["code"].apply(code_tokenizer.tokenize))
    summary_vocab.build_vocabulary(df["summary"].apply(summary_tokenizer.tokenize))
    
    print(f"Code Vocab Size: {len(code_vocab)}")
    print(f"Summary Vocab Size: {len(summary_vocab)}")
    
    # Save Vocabs
    os.makedirs("checkpoints", exist_ok=True)
    save_vocab(code_vocab, f"checkpoints/{args.model_name}_code_vocab.pkl")
    save_vocab(summary_vocab, f"checkpoints/{args.model_name}_summary_vocab.pkl")
    
    # Also save as default for ease of use (optional, but keep it for now)
    if args.model_name != "best_model":
        save_vocab(code_vocab, "checkpoints/code_vocab.pkl")
        save_vocab(summary_vocab, "checkpoints/summary_vocab.pkl")
    
    # Dataloaders
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_loader = get_dataloader(train_df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer, batch_size=args.batch_size)
    val_loader = get_dataloader(val_df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer, batch_size=args.batch_size, shuffle=False)
    
    # Model components
    attn = Attention(args.hid_dim)
    enc = Encoder(len(code_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout)
    dec = Decoder(len(summary_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=summary_vocab.stoi["<pad>"])
    
    best_valid_loss = float('inf')
    patience = args.patience
    epochs_no_improve = 0
    
    print(f"Starting training for {args.epochs} epochs (Patience: {patience})...")

    # Initialize CSV log
    log_file = "training_log.csv"
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,train_ppl,valid_loss,valid_ppl\n")

    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip, device)
        valid_loss = validate_epoch(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"checkpoints/{args.model_name}.pt")
            improvement_tag = "*"
        else:
            epochs_no_improve += 1
            improvement_tag = ""
            
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s {improvement_tag}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f} (Best: {best_valid_loss:.3f})')
        
        # Write to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss},{train_ppl},{valid_loss},{valid_ppl}\n")
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

if __name__ == "__main__":
    import math
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model_name", type=str, default="best_model", help="Name for saving model and vocabs")
    parser.add_argument("--data_path", type=str, default="data/dataset.csv")
    parser.add_argument("--use_codesearchnet", action="store_true", help="Use CodeSearchNet dataset")
    parser.add_argument("--csn_limit", type=int, default=1000, help="Number of records to load from CodeSearchNet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=1.0)
    
    args = parser.parse_args() 
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # Only update if NOT passed on command line
            cli_args = sys.argv
            
            def set_if_not_on_cli(attr, key, config_dict=config):
                # Check if the argument was explicitly passed on the command line
                # This is a simplified check and might not cover all argparse nuances (e.g., short flags)
                # but works for the long-form arguments defined.
                if f"--{attr.replace('_', '-')}" not in cli_args and key in config_dict:
                    setattr(args, attr, config_dict[key])

            set_if_not_on_cli("model_name", "model_name")
            set_if_not_on_cli("emb_dim", "embedding_dim")
            set_if_not_on_cli("hid_dim", "hidden_dim")
            set_if_not_on_cli("n_layers", "n_layers")
            set_if_not_on_cli("dropout", "dropout")
            
            # Training parameters
            if 'training' in config:
                t = config['training']
                set_if_not_on_cli("epochs", "epochs", t)
                set_if_not_on_cli("batch_size", "batch_size", t)
                # Special handling for 'lr' as its key in config is 'learning_rate'
                if "--lr" not in cli_args and 'learning_rate' in t: args.lr = t['learning_rate']
                set_if_not_on_cli("clip", "clip", t)
            
            # Data parameters
            if 'data' in config:
                d = config['data']
                # For boolean flags like use_codesearchnet, check for presence of flag
                # and if the config value is different from the default (False for action="store_true")
                # A more robust check might involve checking if the default was overridden.
                if "--use-codesearchnet" not in cli_args and 'use_codesearchnet' in d:
                    args.use_codesearchnet = d['use_codesearchnet']
                set_if_not_on_cli("csn_limit", "csn_limit", d)
                set_if_not_on_cli("data_path", "data_path", d)

    main(args)
