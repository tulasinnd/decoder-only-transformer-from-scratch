import torch
from datasets import load_dataset 
from transformers import GPT2TokenizerFast

# BPE tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# tokenize and convert text to stream of token IDs
def load_training_data():

    corpus = load_dataset("wikitext", "wikitext-2-raw-v1") # dataset contains train, test, validate, text rows
    texts = [row["text"] for row in corpus["train"]] # list of all text rows
    full_train_text = "\n".join(texts) # join all the text

    enc = tokenizer(
        full_train_text,
        add_special_tokens=False
    ) 

    tokenized_train_text = torch.tensor(enc["input_ids"]) # tokenize entire text, torch.Size([2403644])
    
    return tokenized_train_text

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