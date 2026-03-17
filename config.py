d_model= 128
num_heads=4
num_layers=4

seq_len= 64
max_seq_len=128          # for flexible sequence length

batch_size=16
num_steps = 200         # training iterations
print_every = 200       # logging frequency
learning_rate = 3e-4 
eval_iters=100

temperature= 1.0
max_new_tokens=20
seed = 42

# default dropout values
embedding_dropout = 0.1     # used after emb+pos
attention_dropout = 0.1     # used inside MHA
residual_dropout = 0.1      # used after MHA and FFN outputs
