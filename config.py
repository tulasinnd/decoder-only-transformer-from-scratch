# Model Architecture
d_model = 128
num_heads = 4
num_layers = 4
max_seq_len = 128         # for flexible sequence length

# Training Config
batch_size = 16
num_steps = 1500          # training iterations
learning_rate = 3e-4
# grad_clip_value = 1.0     # safe: 0.5–1.0 (prevents exploding gradients)
eval_iters = 100
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
# top_p = 0.8
