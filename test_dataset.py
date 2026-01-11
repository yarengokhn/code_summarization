from datasets import load_dataset

dataset = load_dataset("Nan-Do/code-search-net-python")

print(dataset)

sample = dataset['train'][0]
print("--- Code ---")
print(sample['code'])
print("\n--- Summary ---")
print(sample['summary'])