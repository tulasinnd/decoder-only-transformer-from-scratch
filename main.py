import torch
from model.decoder import Decoder
from config import *
from data.dataset import load_training_data, get_vocab_size
from training.trainer import train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
tokenized_train_text = load_training_data()

# build model
vocab_size = get_vocab_size()
model = Decoder(vocab_size, max_seq_len, d_model, num_heads, num_layers)
model = model.to(device)

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # create optimizer
criterion = torch.nn.CrossEntropyLoss() # create loss function
train(model, optimizer, criterion, tokenized_train_text, device) # call training loop
