import torch
from model.decoder import Decoder
from data.dataset import load_training_data,load_validation_data,get_vocab_size,tokenizer
from training.trainer import train
from generation.generation_utils import generate,get_input_ids,print_generation
from utils.utils import set_seed,get_device
import config
set_seed(config.seed)
device = get_device()

# load dataset
tokenized_train_text = load_training_data()
tokenized_validation_text=load_validation_data()

# build model
vocab_size = get_vocab_size()
model = Decoder(vocab_size,config)
model = model.to(device)

# TRAINING WITH OPTIONAL RESUME
# ---------------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# defaults
start_step = 1
best_val_loss = float("inf")
# training control (use argparser)
while True:
    resume_input = input("Resume training? (y/n): ").strip().lower() # Validate yes/no input
    if resume_input in ("y", "n"):
        resume = resume_input == "y"
        break
    else:
        print("Invalid input! Please enter 'y' or 'n'.")

while True:
    steps_input = input("Enter steps to run today (e.g., 5000): ").strip() # Validate steps input
    if steps_input.isdigit() and int(steps_input) > 0:
        steps_per_run = int(steps_input)
        break
    else:
        print("Invalid input! Please enter a positive integer.")

# optional resume
if resume:
    ckpt = torch.load("checkpoints/resume_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = ckpt["step"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed, Training from step {start_step} to {start_step - 1 + steps_per_run}")
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
    config.eval_iters,
    start_step=start_step,
    steps_per_run=steps_per_run,
    num_steps=config.num_steps,
    best_val_loss=best_val_loss,
    grad_clip=config.grad_clip
)

# ------------------------------------------------------------------------------
# text generation
model.eval()
while True:
    input_ids, prompt = get_input_ids(tokenizer, device, config.seq_len) # take prompt from user and convert it into ids
    if prompt.lower() == "quit":
        print("Exiting generation.")
        break  # stops the loop

    gen_ids = generate(model, input_ids, config) # generate text intthe form of ids
    print_generation(tokenizer, gen_ids, prompt) # convert generated ids to text and print it 