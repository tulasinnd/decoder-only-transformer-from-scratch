from config import *
import torch
from utils.utils import set_seed
set_seed(seed)

@torch.no_grad()
def generate(model, start_ids, max_new_tokens=10,temperature=temperature):
    model.eval()
    ids = start_ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]  # truncate before feeding

        logits = model(ids)
        next_logits = logits[:, -1, :]      # last token
        
        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)

    return ids

def get_input_ids(tokenizer, device, seq_len):    

    prompt = input("Enter prompt text: ").strip() # Ask the user for a non-empty prompt
    while not prompt:
        prompt = input("Prompt cannot be empty. Enter prompt text: ").strip()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) #  tokenize the prompt
    input_ids = input_ids[:, -seq_len:] # truncate the extra prompt safely.
    
    return input_ids, prompt

def print_generation(tokenizer, output_ids, prompt):
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True) # convert generated ids to text
    print("Prompt:", prompt)
    print("Generated:", text)