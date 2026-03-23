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

def get_lr(step, warmup_steps, total_steps, base_lr):

    # warmup
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    # cosine decay
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

# train the decoder-only transformer on tokenized text data
def train(model, optimizer, criterion, tokenized_train_text, tokenized_validation_text, device,
          eval_iters, start_step, steps_per_run, num_steps, best_val_loss, grad_clip=None):
    model.train()
    total_loss = 0.0
    warmup_steps = max(100, int(0.05 * num_steps)) # 5% warmup

    end_step = min(start_step + steps_per_run, num_steps)

    for step in range(start_step, end_step + 1):

        # update learning rate
        lr = get_lr(step, warmup_steps, num_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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

        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        # training progress logging
        if step % print_every == 0:
            avg_loss = total_loss / print_every
            ppl_train = math.exp(avg_loss)

            val_loss = evaluate(model, tokenized_validation_text, criterion, device, batch_size, seq_len, eval_iters)
            ppl_eval = math.exp(val_loss)

            grad_norm_str = f"{grad_norm:.2f}" if grad_clip is not None else "N/A"

            print(
                f"[Step {step:6d}] "
                f"LR={lr:.6f} | "
                f"Grad={grad_norm_str} | "
                f"Train: {avg_loss:.4f} ({ppl_train:.2f}) | "
                f"Val: {val_loss:.4f} ({ppl_eval:.2f})"
            )
            total_loss = 0.0

            # checkpointing the best model with lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": model.state_dict(),
                            "step": step,
                            "val_loss": val_loss}, "checkpoints/best_model.pt")
                print(f"Best model updated! at validation loss: {best_val_loss:.4f}")

            # save resume checkpoint every eval
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss
            }, "checkpoints/resume_checkpoint.pt")

    if end_step == num_steps:
        print("Training completed fully.")