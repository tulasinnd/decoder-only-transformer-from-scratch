import torch
import math
from config import *
from data.dataset import get_batch

@torch.no_grad()
def evaluate(model, tokenized_validation_text, criterion, device, batch_size, seq_len, eval_iters=100):
    model.eval()  # set model to evaluation mode
    total_loss = 0.0

    for _ in range(eval_iters):
        X, Y = get_batch(tokenized_validation_text, batch_size, seq_len, device=device)

        logits = model(X)          # (B, S, V)
        B, S, V = logits.shape
        logits = logits.view(B * S, V)
        Y = Y.view(B * S)

        loss = criterion(logits, Y)
        total_loss += loss.item()

    model.train()  # back to training mode
    return total_loss / eval_iters

# train the decoder-only transformer on tokenized text data
def train(model, optimizer, criterion, tokenized_train_text, tokenized_validation_text, device,eval_iters):
    model.train()
    total_loss = 0.0
    best_val_loss = float('inf')  # start with “infinite” loss for checkpoints

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
            ppl_train = math.exp(avg_loss)

            loss = evaluate(model, tokenized_validation_text, criterion, device, batch_size, seq_len, eval_iters)
            ppl_eval = math.exp(loss)

            print(f"Step {step}")
            print(f"Train Loss: {avg_loss:.4f} | Train PPL: {ppl_train:.2f}")
            print(f"Valid Loss: {loss:.4f} | Validation PPL: {ppl_eval:.2f}")
            total_loss = 0.0

            # checkpointing the best model with lowest validation loss
            # if loss < best_val_loss:
            #     best_val_loss = loss
                # torch.save(model.state_dict(), "checkpoints/best_model.pt")
                # print(f"Best model updated! at validation loss: {best_val_loss:.4f}")