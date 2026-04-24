# ================== Exp:1 with BASELINE CONFIG ==================

# Model Architecture
d_model = 128
num_heads = 4
num_layers = 8
max_seq_len = 128
seq_len = 64   # training chunk length

# Training Config
batch_size = 16
num_steps = 5000     
resume= False     
learning_rate = 3e-4
warmup_steps = int(0.1 * num_steps)
grad_clip = 1.0
seed = 42

# Logging / Eval
eval_iters = 100
print_every = 100

# Regularization
embedding_dropout = 0.1
attention_dropout = 0.2
residual_dropout = 0.1

# Generation Config (for comparison only)
temperature = 1.0
max_new_tokens = 20
sampling_strategy = "top_p"
top_p = 0.8
top_k = 50

# Architecture choices
weight_tying = False
post_norm = False   # pre-norm transformer

# ====================================================