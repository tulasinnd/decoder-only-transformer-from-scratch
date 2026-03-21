# Model Architecture
d_model = 128
num_heads = 4
num_layers = 4
max_seq_len = 128         # for flexible sequence length

# Training Config
batch_size = 16
num_steps = 5000          # training iterations
learning_rate = 3e-4
eval_iters = 500
print_every = 500         # logging frequency
seed = 42
seq_len = 64

# Regularization
embedding_dropout = 0.1       # used after emb+pos 
attention_dropout = 0.1       # used inside MHA
residual_dropout = 0.1        # used after MHA and FFN outputs

# Generation Config
temperature = 1.0
max_new_tokens = 20       # generated text length
