# Disable tokenizers parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from preprocessing import  preprocess_code


# 1. Load the dataset (currently only has 'train' split)
raw_dataset = load_dataset("Nan-Do/code-search-net-python")

# 2. Split the data (Create validation and test sets)
# 80% Train, 20% Temporary (Test+Val)
train_test_split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)

# Split the 20% into two parts (10% Val, 10% Test)
test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

# Combine into a new DatasetDict structure
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': test_valid_split['train'],
    'test': test_valid_split['test']
})

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    """
    Tokenizes code and summary pairs.
    Uses the new 'text_target' parameter instead of 'as_target_tokenizer'.
    """
    cleaned_codes = [preprocess_code(c, is_code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_code=False) for s in examples["summary"]]
    
    # Process encoder input (code) and decoder target (summary) together
    model_inputs = tokenizer(
        cleaned_codes, 
        text_target=cleaned_summaries,  # 'labels' are automatically created
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    return model_inputs

# 4. Apply tokenization to all splits
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Save and print information
tokenized_dataset.save_to_disk("./data/tokenized_dataset")

print(f"Dataset successfully split and tokenized!")
print(f"Train size: {len(tokenized_dataset['train'])}")
print(f"Validation size: {len(tokenized_dataset['validation'])}")
print(f"Test size: {len(tokenized_dataset['test'])}")