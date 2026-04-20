import torch
import argparse
import importlib
import os

from model.decoder import Decoder
from data.dataset import load_training_data, load_validation_data, get_vocab_size, tokenizer
from training.trainer import train
from generation.generation_utils import generate, get_input_ids, print_generation
from utils.utils import set_seed, get_device

# -------------------------------
# CONFIG LOADER
# -------------------------------
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="default_config",  # default config.py
        help="Config file (default: config.py)"
    )
    args = parser.parse_args()

    config = importlib.import_module(args.config)
    exp_name = args.config.split(".")[-1]

    print("Using config:", args.config)

    return config, exp_name

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    config,exp_name = load_config()

    run_dir = os.path.join("runs", exp_name)
    os.makedirs(run_dir, exist_ok=True)
    print("Experiment results stored in directory: ", run_dir)

    # setup
    set_seed(config.seed)
    device = get_device()

    # load dataset
    tokenized_train_text = load_training_data()
    tokenized_validation_text = load_validation_data()

    # build model
    vocab_size = get_vocab_size()
    model = Decoder(vocab_size, config)
    model = model.to(device)

    # optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # training state
    start_step = 1
    best_val_loss = float("inf")

    # resume option
    ckpt_path = os.path.join(run_dir, "resume_checkpoint.pt")
    if config.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed training from step {start_step} to {config.num_steps}")
    else:
        print("Starting training fresh.")

    # training loop
    train(
        model, 
        optimizer,
        criterion, 
        tokenized_train_text,
        tokenized_validation_text,
        device,
        config,
        run_dir=run_dir,
        start_step=start_step,
        num_steps=config.num_steps,
        best_val_loss=best_val_loss,
        warmup_steps=config.warmup_steps
    )

    # -------------------------------
    # GENERATION LOOP
    # -------------------------------
    model.eval()

    while True:
        input_ids, prompt = get_input_ids(tokenizer, device, config.seq_len)

        if prompt.lower() == "quit":
            print("Exiting generation.")
            break

        gen_ids = generate(model, input_ids, config)
        print_generation(tokenizer, gen_ids, prompt)

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    main()