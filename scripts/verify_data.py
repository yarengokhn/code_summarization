import os
import sys
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import Tokenizer, Vocabulary, preprocess_data, load_codesearchnet_dataset
from src.dataset import get_dataloader

def verify_pipeline():
    print("--- 1. Loading Data Sample ---")
    # Load from local CSV
    data_path = "data/dataset.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    # Take a small sample
    df = df.head(10)
    print(f"Loaded {len(df)} samples from {data_path}.")
    print("Sample Output:")
    print(df.head(2))
    
    print("\n--- 2. Preprocessing Data ---")
    df = preprocess_data(df)
    print(f"Post-preprocessing count: {len(df)}")
    print("Sample Preprocessed:")
    print(df.head(2))
    
    print("\n--- 3. Building Vocabulary ---")
    code_tokenizer = Tokenizer(is_code=True)
    summary_tokenizer = Tokenizer(is_code=False)
    
    code_vocab = Vocabulary(min_freq=1) # min_freq=1 to ensure we capture everything for this small test
    summary_vocab = Vocabulary(min_freq=1)
    
    code_vocab.build_vocabulary(df["code"].apply(code_tokenizer.tokenize))
    summary_vocab.build_vocabulary(df["summary"].apply(summary_tokenizer.tokenize))
    
    print(f"Code Vocab Size: {len(code_vocab)}")
    print(f"Summary Vocab Size: {len(summary_vocab)}")
    
    print("\n--- 4. Checking Dataloader ---")
    batch_size = 2
    loader = get_dataloader(df, code_vocab, summary_vocab, code_tokenizer, summary_tokenizer, batch_size=batch_size, shuffle=False)
    
    batch = next(iter(loader))
    
    print("Batch Keys:", batch.keys())
    print("Code IDs Shape:", batch["code_ids"].shape)
    print("Summary IDs Shape:", batch["summary_ids"].shape)
    
    print("\n--- 5. Content Verification ---")
    # Decode first sample in batch
    code_ids = batch["code_ids"][0]
    summary_ids = batch["summary_ids"][0]
    
    decoded_code = " ".join(code_vocab.decode(code_ids.tolist()))
    decoded_summary = " ".join(summary_vocab.decode(summary_ids.tolist()))
    
    print("Decoded Code (with padding/unk):")
    print(decoded_code)
    print("Decoded Summary (with sos/eos/padding):")
    print(decoded_summary)
    
    # Check for <sos> and <eos>
    assert summary_ids[0] == summary_vocab.stoi["<sos>"], "Summary must start with <sos>"
    # We can't guarantee eos is at the very end if there's padding, but it should be present before padding
    # For this small batch with potentially no padding if lengths align, we check existence
    if summary_vocab.stoi["<eos>"] in summary_ids:
        print("Verified <eos> presence.")
    else:
        print("WARNING: <eos> token not found in summary ids (might be truncated?).")

    print("\nVerification Complete!")

if __name__ == "__main__":
    verify_pipeline()
