import torch
import math
from config import *
from data.dataset import get_batch

# train the decoder-only transformer on tokenized text data
def train(model, optimizer, criterion, tokenized_train_text, device):
    model.train()
    total_loss = 0.0

    for step in range(1, num_steps + 1):

        # sample a training batch
        X, Y = get_batch(tokenized_train_text, batch_size, seq_len, device=device)

        # forward pass
        logits = model(X)               # (B, S, V)
        B, S, V = logits.shape
        logits = logits.view(B * S, V)  # flatten batch and sequence dimensions for cross-entropy computation
        Y = Y.view(B * S)

        # compute loss
        loss = criterion(logits, Y)

        # backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # training progress logging
        if step % print_every == 0:
            avg_loss = total_loss / print_every
            ppl = math.exp(avg_loss)
            print(f"Step {step} | Avg Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")
            total_loss = 0.0