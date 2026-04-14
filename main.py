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
optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

start_step = 1
best_val_loss = float("inf")

if config.resume:
    ckpt = torch.load("checkpoints/resume_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = ckpt["step"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"Resumed, Training from step {start_step} to {config.num_steps}")
else:
    print("Starting training fresh.")

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
    num_steps=config.num_steps,
    best_val_loss=best_val_loss,
    grad_clip=config.grad_clip,
    warmup_steps= config.warmup_steps
)

# text generation
model.eval()
while True:
    input_ids, prompt = get_input_ids(tokenizer, device, config.seq_len) # take prompt from user and convert it into ids
    if prompt.lower() == "quit":
        print("Exiting generation.")
        break  # stops the loop

    gen_ids = generate(model, input_ids, config) # generate text intthe form of ids
    print_generation(tokenizer, gen_ids, prompt) # convert generated ids to text and print it 