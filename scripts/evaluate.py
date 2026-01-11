import torch
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.bleu import BLEUScore
from tqdm import tqdm
import os

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ COLLATE FUNCTION ============
def collate_fn(batch):
    """Equalizes sequences of different lengths with padding - REQUIRED!"""
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, 
                                    padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, 
                                 padding_value=tokenizer.pad_token_id)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }


def greedy_decode_batch(model, src, tokenizer, device, max_length=128):
    """Performs greedy decoding for batch"""
    batch_size = src.size(0)
    
    # Encoder
    with autocast():
        encoder_outputs, hidden, cell = model.encoder(src)
    
    # Initialize decoder for each sample
    decoder_input = torch.LongTensor([tokenizer.cls_token_id] * batch_size).to(device)
    
    # Store results
    all_predictions = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    
    for t in range(max_length):
        if all(finished):
            break
        
        with autocast():
            output, hidden, cell, _ = model.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
        
        # Highest probability token
        top_pred = output.argmax(1)
        
        # Add token for each sample
        for i in range(batch_size):
            if not finished[i]:
                token_id = top_pred[i].item()
                
                # Stop if EOS or SEP token
                if token_id == tokenizer.sep_token_id or token_id == tokenizer.eos_token_id:
                    finished[i] = True
                else:
                    all_predictions[i].append(token_id)
        
        # Input for next step
        decoder_input = top_pred
    
    # Convert tokens to text
    decoded_texts = []
    for pred_tokens in all_predictions:
        text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(text)
    
    return decoded_texts


def calculate_metrics(model, iterator, tokenizer, device, max_length=128):
    """Evaluates model performance on test set"""
    model.eval()
    
    # Metric tools
    bleu = BLEUScore()
    rouge = ROUGEScore()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    total_loss = 0
    predictions = []
    targets = []
    
    print("🔍 Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch['input_ids'].to(device)
            trg = batch['labels'].to(device)
            
            # Inference with mixed precision
            with autocast():
                # 1. Calculate loss with teacher forcing
                output = model(src, trg, teacher_forcing_ratio=0)
                
                output_dim = output.shape[-1]
                output_loss = output.reshape(-1, output_dim)
                trg_loss = trg[:, 1:].reshape(-1)
                
                loss = criterion(output_loss, trg_loss)
                total_loss += loss.item()
            
            # 2. Text generation with greedy decoding
            batch_predictions = greedy_decode_batch(model, src, tokenizer, device, max_length)
            
            # 3. Decode target texts
            for i in range(len(trg)):
                pred_text = batch_predictions[i]
                target_text = tokenizer.decode(trg[i][1:], skip_special_tokens=True)
                
                predictions.append(pred_text)
                targets.append(target_text)
    
    # Calculate metrics
    avg_loss = total_loss / len(iterator)
    perplexity = math.exp(min(avg_loss, 100))
    
    bleu_score = bleu(predictions, [[t] for t in targets])
    rouge_results = rouge(predictions, targets)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "bleu": bleu_score.item(),
        "rouge1": rouge_results['rouge1_fmeasure'].item(),
        "rouge2": rouge_results['rouge2_fmeasure'].item(),
        "rougeL": rouge_results['rougeL_fmeasure'].item(),
        "samples": list(zip(targets, predictions))[:10]
    }


def print_results(results):
    """Prints results in formatted style"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS")
    print("="*60)
    
    print(f"\n🎯 Loss & Perplexity:")
    print(f"   • Cross-Entropy Loss: {results['loss']:.4f}")
    print(f"   • Perplexity: {results['perplexity']:.4f}")
    
    print(f"\n📝 Text Generation Metrics:")
    print(f"   • BLEU Score: {results['bleu']:.4f}")
    print(f"   • ROUGE-1: {results['rouge1']:.4f}")
    print(f"   • ROUGE-2: {results['rouge2']:.4f}")
    print(f"   • ROUGE-L: {results['rougeL']:.4f}")
    
    print("\n" + "="*60)
    print("🔎 QUALITATIVE EVALUATION (Sample Summaries)")
    print("="*60)
    
    for i, (tgt, pred) in enumerate(results['samples'], 1):
        print(f"\n📌 Sample {i}:")
        print(f"   Ground Truth: {tgt[:100]}..." if len(tgt) > 100 else f"   Ground Truth: {tgt}")
        print(f"   Model Output: {pred[:100]}..." if len(pred) > 100 else f"   Model Output: {pred}")


# ============ MAIN EXECUTION ============
print("🚀 Starting model evaluation...\n")

# Model check
model_path = 'models/best-model.pt'
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Please train the model first!")
else:
    # Prepare test data
    print("📊 Loading test data...")
    test_subset = tokenized_dataset['test'].select(range(15000))
    
    test_iterator = DataLoader(
        test_subset, 
        batch_size=32, 
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Test samples: {len(test_subset)}")
    print(f"Number of batches: {len(test_iterator)}\n")
    
    # Hyperparameters (same as training!)
    INPUT_DIM = tokenizer.vocab_size
    OUTPUT_DIM = tokenizer.vocab_size
    EMB_DIM = 128
    HID_DIM = 128
    
    # Create model
    print("🏗️  Building model...")
    eval_enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(DEVICE)
    eval_dec = AttentionDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM).to(DEVICE)
    eval_model = Seq2Seq(eval_enc, eval_dec, DEVICE).to(DEVICE)
    
    # Load weights
    print("📥 Loading model weights...")
    eval_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Model loaded!\n")
    
    # Run evaluation
    results = calculate_metrics(eval_model, test_iterator, tokenizer, DEVICE)
    
    # Print results
    print_results(results)
    
    # Save results to file
    results_path = 'models/evaluation_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Cross-Entropy Loss: {results['loss']:.4f}\n")
        f.write(f"Perplexity: {results['perplexity']:.4f}\n")
        f.write(f"BLEU Score: {results['bleu']:.4f}\n")
        f.write(f"ROUGE-1: {results['rouge1']:.4f}\n")
        f.write(f"ROUGE-2: {results['rouge2']:.4f}\n")
        f.write(f"ROUGE-L: {results['rougeL']:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("SAMPLE SUMMARIES\n")
        f.write("="*60 + "\n\n")
        for i, (tgt, pred) in enumerate(results['samples'], 1):
            f.write(f"Sample {i}:\n")
            f.write(f"Ground Truth: {tgt}\n")
            f.write(f"Model Output: {pred}\n\n")
    
    print(f"\n💾 Results saved: {results_path}")
    print("\n✅ Evaluation completed!")