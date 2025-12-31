import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_results(log_file="training_log.csv", output_dir="results"):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(log_file)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()
    
    # Plot Perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_ppl'], label='Train PPL', marker='s')
    plt.plot(df['epoch'], df['valid_ppl'], label='Valid PPL', marker='s')
    plt.title('Training & Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "ppl_plot.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="training_log.csv")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    plot_results(args.log_file, args.output_dir)
