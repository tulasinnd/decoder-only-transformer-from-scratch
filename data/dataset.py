import torch
from datasets import load_dataset 
from transformers import GPT2TokenizerFast

# BPE tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
corpus = load_dataset("wikitext", "wikitext-2-raw-v1", download_mode="reuse_cache_if_exists") # dataset contains train, test, validate, text rows

# tokenize and convert text to stream of token IDs
def load_training_data():
    texts = [row["text"] for row in corpus["train"]]
    enc = tokenizer(texts,add_special_tokens=False,return_attention_mask=False )
    tokenized_train_text = torch.tensor([token for seq in enc["input_ids"] for token in seq]) # flattentorch.Size([2403644])

    return tokenized_train_text

def load_validation_data():
    texts = [row["text"] for row in corpus["validation"]] # list of all text rows
    enc = tokenizer(texts,add_special_tokens=False,return_attention_mask=False ) 
    tokenized_validation_text = torch.tensor([token for seq in enc["input_ids"] for token in seq]) # torch.Size([248461])
    
    return tokenized_validation_text

# code for creating batches for training or validation
def get_batch(tokens, batch_size, block_size, device): 

    # randomly choose starting indices
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))

    # input sequences
    x = torch.stack([tokens[i : i + block_size] for i in ix])

    # target sequences (next token)
    y = torch.stack([tokens[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

# total vocab
def get_vocab_size():
    return tokenizer.vocab_size

# dataset information (this function is not part of model architecture)
def get_dataset_info(dataset=corpus, tokenizer=tokenizer):
      
    # Splits
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    
    # Sample counts
    train_size = len(train_data)
    val_size = len(val_data)
    test_size = len(test_data)

    # Token counts
    def count_tokens(split):
        total_tokens = 0
        for example in split:
            tokens = tokenizer.encode(example["text"])
            total_tokens += len(tokens)
        return total_tokens
    
    train_tokens = count_tokens(train_data)
    val_tokens = count_tokens(val_data)
    test_tokens = count_tokens(test_data)
    
    # Vocab info
    vocab_size = tokenizer.vocab_size
    
    # Final dictionary
    info = {       
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
        
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "test_tokens": test_tokens,
        
        "vocab_size": vocab_size,
    } 
    return info