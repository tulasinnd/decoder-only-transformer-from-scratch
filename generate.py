import torch
from model.decoder import Decoder
from data.dataset import tokenizer,get_vocab_size
from config import *
from generation.generation_utils import generate,get_input_ids,print_generation
from utils.utils import set_seed,get_device
set_seed(seed)
device = get_device()

dropout_config = {
    "embedding": 0.2,
    "attention": 0.2,
    "residual": 0.2,
}

# load model from checkpoint
vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers,dropout=dropout_config)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.to(device)

model.eval()
# generate text 
input_ids, prompt= get_input_ids(tokenizer, device, seq_len)
gen_ids = generate(model, input_ids,max_new_tokens, temperature=0.5, top_p=0.9) 
print_generation(tokenizer, gen_ids, prompt)