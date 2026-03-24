import torch
from model.decoder import Decoder
from data.dataset import tokenizer,get_vocab_size
from config import *
from generation.generation_utils import generate,get_input_ids,print_generation
from utils.utils import set_seed,get_device
set_seed(seed)
device = get_device()

# load model from checkpoint
vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers)
missing, unexpected= model.load_state_dict(torch.load("checkpoints/resume_checkpoint.pt", map_location=device))
if missing: print("Missing:", missing)
if unexpected: print("Unexpected:", unexpected)
model.to(device)

model.eval()
# generate text 
input_ids, prompt= get_input_ids(tokenizer, device, seq_len)
gen_ids = generate(model, input_ids,max_new_tokens, temperature=0.5, top_p=0.9) 
print_generation(tokenizer, gen_ids, prompt)