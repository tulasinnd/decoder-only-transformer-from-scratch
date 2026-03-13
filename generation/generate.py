from config import *
import torch

def generate(model, start_ids, max_new_tokens=10):
    model.eval()
    ids = start_ids.clone()

    for _ in range(max_new_tokens):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]  # truncate before feeding
        logits = model(ids)
        next_logits = logits[:, -1, :]      # last token
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

    return ids