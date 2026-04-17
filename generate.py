import torch
from model.decoder import Decoder
from data.dataset import tokenizer,get_vocab_size
from default_config import *
from generation.generation_utils import generate,get_input_ids,print_generation
from utils.utils import set_seed,get_device
set_seed(seed)
device = get_device()

# load model from checkpoint
vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers)
ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
if missing: print("Missing:", missing)
if unexpected: print("Unexpected:", unexpected)
model.to(device)

# text generation
max_new_tokens=50
model.eval()
while True:
    input_ids, prompt = get_input_ids(tokenizer, device, seq_len)
    if prompt.lower() == "quit":
        print("Exiting generation.")
        break  # stops the loop

    gen_ids = generate(model, input_ids,max_new_tokens, temperature, top_p=0.8) 
    print_generation(tokenizer, gen_ids, prompt)