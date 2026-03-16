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

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # create optimizer
criterion = torch.nn.CrossEntropyLoss() # create loss function
train(model, optimizer, criterion, tokenized_train_text, tokenized_validation_text, device,eval_iters)  # call training loop

# generation 
input_ids, prompt= get_input_ids(tokenizer, device, seq_len)
gen_ids = generate(model, input_ids,max_new_tokens, temperature, top_p=0.8) 
print_generation(tokenizer, gen_ids, prompt)