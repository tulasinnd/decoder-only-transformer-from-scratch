import torch
from model.decoder import Decoder
from config import *
from data.dataset import load_training_data,load_validation_data,get_vocab_size,tokenizer
from training.trainer import train
from generation.generation_utils import generate,get_input_ids,print_generation
from utils.utils import set_seed,get_device
set_seed(seed)
device = get_device()

# load dataset
tokenized_train_text = load_training_data()
tokenized_validation_text=load_validation_data()

# build model
vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers)
model = model.to(device)

# TRAINING WITH OPTIONAL RESUME
# ---------------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# defaults
start_step = 1
best_val_loss = float("inf")
# training control (use argparser)
resume_input = input("Resume training? (y/n): ").strip().lower()
resume = resume_input == "y"
steps_per_run = int(input("Enter steps to run today (e.g., 5000): "))

# resume
if resume:
    ckpt = torch.load("checkpoints/resume_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = ckpt["step"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed, Training from step {start_step} to {start_step + steps_per_run}")
else:
    print("No checkpoint found. Starting fresh.")

# training
train(
    model,
    optimizer,
    criterion,
    tokenized_train_text,
    tokenized_validation_text,
    device,
    eval_iters,
    start_step=start_step,
    steps_per_run=steps_per_run,
    num_steps=num_steps,
    best_val_loss=best_val_loss,
    grad_clip=1.0
)

# ------------------------------------------------------------------------------

# generation 
input_ids, prompt= get_input_ids(tokenizer, device, seq_len)
gen_ids = generate(model, input_ids,max_new_tokens, temperature, top_p=0.8) 
print_generation(tokenizer, gen_ids, prompt)