"""
Decoder-only Transformer (GPT-style) implementation from scratch.

Components implemented:
- Token Embeddings
- Learned Positional Embeddings
- Multi-Head Causal Self-Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Stacked Decoder Layers
"""

import torch
import torch.nn as nn
import math

# IN: x (token IDs)
# OUT: embedding vectors looked up from the vocabulary embedding table
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding= nn.Embedding(vocab_size,d_model)

    def forward(self, x_batch):
        x_emb=self.embedding(x_batch)
        return x_emb

# IN: input tensor used only to infer batch and sequence length
# OUT: learned positional embeddings for each token position
class PositionalEncodings(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__() 
        self.lpe = nn.Embedding(max_seq_len
                                , d_model ) 
 
    def forward(self, input_ids): 
        batch_size, seq_len = input_ids.shape

        # create position indices
        positions = torch.arange(seq_len).to(input_ids.device)  # ex: if input sentence has 5 tokens then [0, 1, 2, 3, 4] 
        positions = positions.unsqueeze(0).expand(batch_size, seq_len) # ex: [[0, 1, 2, 3, 4]] , [ [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]

        # map position indices to learned positional embeddings
        return self.lpe(positions) # Each position index is mapped to a learned embedding vector, final shape: (batch, seq_len, d_model)
    
# IN: input representations X
# OUT: context-aware representations of X using causal self-attention
class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model= d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads

        # create weight matrices 
        self.Wq_projection = nn.Linear(d_model, d_model,bias=False) 
        self.Wk_projection = nn.Linear(d_model, d_model,bias=False)
        self.Wv_projection = nn.Linear(d_model, d_model,bias=False)
        self.Wo_projection = nn.Linear(d_model, d_model,bias=False)

    def forward(self, X,padding_mask=None):
        batch, seq_len, d_model = X.shape
        
        # project input representations into Q, K, V
        Q= self.Wq_projection(X)
        K= self.Wk_projection(X)  
        V= self.Wv_projection(X)

        # split into multiple attention heads
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(Q, K.transpose(-1,-2))/ math.sqrt(self.head_dim) 
        causal_mask= torch.tril(torch.ones(seq_len, seq_len, device = X.device)) # create causal mask to prevent attending to future tokens
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9) 
        if padding_mask is not None: # optional padding mask to ignore padded tokens in attention
            mask = padding_mask.unsqueeze(1).unsqueeze(2).to(X.device)
            scores = scores.masked_fill(mask == 0, -1e9) # mask padded tokens so they neither attend nor receive attention
        weights = torch.softmax(scores, dim= -1) # attention weights after softmax normalization

        out = torch.matmul(weights, V)
        out = out.transpose(1,2).reshape(batch, seq_len,d_model) # concatenate attention heads
        out= self.Wo_projection(out)
        return out      
    
# IN: representations after attention
# OUT: normalized representations for stable and consistent layer-wise computation
class LN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        return self.norm(X)
    
# IN: normalized representations
# OUT: transformed representations with learned non-linear features (shape preserved)
class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4 * d_model, d_model)

    def forward(self, X):
        h = self.relu(self.linear1(X))
        out = self.linear2(h)
        return out 
    
# IN: final context-aware token representations
# OUT: logits over vocabulary for next-token prediction at each position
class Logit(nn.Module): 
    def __init__(self, d_model, vocab_size): 
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
 
    def forward(self, x):
        return self.linear(x)   # logits
    
# create one full decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MHA(d_model, num_heads)
        self.ln1 = LN(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = LN(d_model)

    def forward(self, X, return_states=False,padding_mask=None):
        states = {} if return_states else None

        # self-attention block followed by residual connection and layer normalization
        mha_x = self.mha(X,padding_mask=padding_mask) 
        res1 = mha_x + X
        ln1_x = self.ln1(res1)

        # feed-forward block followed by residual connection and layer normalization
        ffn_x = self.ffn(ln1_x)
        res2 = ln1_x + ffn_x
        ln2_x = self.ln2(res2)

        # optionally return intermediate states for analysis or visualization
        if return_states: 
            states["mha_x"] = mha_x.detach()
            states["res1"] = res1.detach()
            states["ln1_x"] = ln1_x.detach()
            states["ffn_x"] = ffn_x.detach()
            states["res2"] = res2.detach()
            states["ln2_x"] = ln2_x.detach()

            return ln2_x, states

        return ln2_x

# stacking multiple decoder layers
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_layers):
        super().__init__()

        # full decoder architecture with specified number of layers
        self.token_embedding = Embeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncodings(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)]) # stack decoder layers
        self.logit = Logit(d_model, vocab_size)

    def forward(self, input_ids, return_states=False, padding_mask=None):
        states = {} if return_states else None

        emb = self.token_embedding(input_ids)           # lookup token embeddings
        pos = self.positional_encoding(input_ids)       # add positional embeddings
        x= emb+pos

        if return_states:
            states["x_embed"] = x.detach()

        for i, layer in enumerate(self.layers):         # pass representations through stacked decoder layers
            if return_states:
                x, layer_states = layer(x, return_states=True,padding_mask=padding_mask,)
                states[f"layer_{i}"] = layer_states
            else:
                x = layer(x,padding_mask=padding_mask)

        logits = self.logit(x)                          # compute final logits

        if return_states:
            return logits, states

        return logits
