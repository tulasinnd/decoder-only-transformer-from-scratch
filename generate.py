import torch
from model.decoder import Decoder
from data.dataset import tokenizer,get_vocab_size
from config import *
from generation.generation_utils import generate
from utils.utils import set_seed,get_device
set_seed(seed)
device = get_device()


vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.to(device)

model.eval()

prompt = input("Enter prompt text: ").strip()
while not prompt: # make sure prompt is not empty
    prompt = input("Prompt cannot be empty. Enter prompt text: ").strip()
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) # tokenize the prompt
input_ids = input_ids[:, -seq_len:]   # safe truncation if prompt size is greater than the sequence length the model trained on
output=generate(model, input_ids, max_new_tokens=10,temperature=0.7)
text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print("Prompt:", prompt)
print("Generated:", text)