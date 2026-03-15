from config import *
import torch
torch.manual_seed(42)

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