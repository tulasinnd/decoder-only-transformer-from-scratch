from config import *
import torch
import torch.nn.functional as F
# from utils.utils import set_seed
# set_seed(seed)

def sample_top_k(logits, top_k=50):
    topk_vals, topk_indices = torch.topk(logits, top_k)
    probs = F.softmax(topk_vals, dim=-1) # ignore all the other tokens except topk

    next_token = torch.multinomial(probs, 1) # pick one from that
    next_token = topk_indices.gather(-1, next_token) # From topk_indices, pick elements at positions specified in next_token, along last dimension. 

    return next_token

def sample_top_p(logits, top_p=0.7):
    probs = F.softmax(logits, dim=-1) # Convert logits → probabilities    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True) # sort probabilities (descending)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # compute cumulative probabilities
    sorted_indices_to_remove = cumulative_probs > top_p # create mask: remove tokens where cumulative prob exceeds top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()     # shift mask right so we always keep at least 1 token
    sorted_indices_to_remove[..., 0] = False
    sorted_probs[sorted_indices_to_remove] = 0.0     # zero out removed probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)     # renormalize remaining probabilities

    next_token = torch.multinomial(sorted_probs, 1)     # sample from filtered distribution
    next_token = sorted_indices.gather(-1, next_token)     # map back to original vocab indices

    return next_token 

@torch.no_grad()
def generate(model, start_ids, max_new_tokens=10,temperature=temperature,top_k=None,top_p=None):
    if top_k is not None and top_p is not None:
        raise ValueError("Choose either top_k or top_p, not both.")
    
    model.eval()
    ids = start_ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]  # truncate before feeding

        logits = model(ids)
        next_logits = logits[:, -1, :]      # last token
        
        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True) # greedy decoding when temp=0

        else:
            scaled_logits = next_logits / temperature # temperature sampling applied

            if top_k is not None:
                next_id = sample_top_k(scaled_logits, top_k) # use either top_k or top_p,

            elif top_p is not None:
                next_id = sample_top_p(scaled_logits, top_p)

            else:
                probs = F.softmax(scaled_logits, dim=-1) # temperature sampling over full vocabulary
                next_id = torch.multinomial(probs, 1)

        ids = torch.cat([ids, next_id], dim=1)

    return ids

def get_input_ids(tokenizer, device, seq_len):    

    prompt = input("Enter prompt text: ").strip() # Ask the user for a non-empty prompt
    while not prompt:
        prompt = input("Prompt cannot be empty. Enter 'quit' to exit or prompt to continue ").strip()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) #  tokenize the prompt
    input_ids = input_ids[:, -seq_len:] # truncate the extra prompt safely.
    
    return input_ids, prompt

def print_generation(tokenizer, output_ids, prompt):
    text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True) # convert generated ids to text
    print("Prompt:", prompt)
    print("Generated:", text)