import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
try:
    from .data_preprocessing import preprocess_code
except ImportError:
    from data_preprocessing import preprocess_code


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    cleaned_codes = [preprocess_code(c, is_code=True) for c in examples["code"]]
    cleaned_summaries = [preprocess_code(s, is_code=False) for s in examples["summary"]]
    
    model_inputs = tokenizer(
        cleaned_codes, 
        text_target=cleaned_summaries,
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    return model_inputs

TOKENIZED_PATH = "./data/tokenized_dataset"

if os.path.exists(TOKENIZED_PATH):
    from datasets import load_from_disk
    print(f"Loading tokenized dataset from {TOKENIZED_PATH}...")
    tokenized_dataset = load_from_disk(TOKENIZED_PATH)
else:
    print(f"Tokenized dataset not found. Loading raw dataset...")
    try:
        raw_dataset = load_dataset("Nan-Do/code-search-net-python")
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        print("Please ensure you have an internet connection or the dataset is cached.")
        raise

    train_test_split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_valid_split['train'],
        'test': test_valid_split['test']
    })
    
    print(f"Tokenizing dataset and saving to {TOKENIZED_PATH}...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.save_to_disk(TOKENIZED_PATH)

print(f"Dataset ready!")
print(f"Train size: {len(tokenized_dataset['train'])}")
print(f"Validation size: {len(tokenized_dataset['validation'])}")
print(f"Test size: {len(tokenized_dataset['test'])}")