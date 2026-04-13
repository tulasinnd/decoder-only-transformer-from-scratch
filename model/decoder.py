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
        return self.embedding(x_batch)

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
    def __init__(self, d_model, num_heads,attention_dropout):
        super().__init__()
        self.attn_dropout = nn.Dropout(attention_dropout)
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

        scores = torch.matmul(Q, K.transpose(-1,-2))/ math.sqrt(self.head_dim) # prevents large dot-product values
        causal_mask= torch.tril(torch.ones(seq_len, seq_len, device = X.device)) #create causal mask(to buffer)to prevent attending to future tokens
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, -1e9) 
        if padding_mask is not None: # optional padding mask to ignore padded tokens in attention
            mask = padding_mask.unsqueeze(1).unsqueeze(2).to(X.device)
            scores = scores.masked_fill(mask == 0, float('-inf')) # mask padded tokens so they neither attend nor receive attention
        weights = torch.softmax(scores, dim= -1) # attention weights after softmax normalization
        weights = self.attn_dropout(weights) # apply dropout

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
        self.relu = nn.GELU()
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
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
 
    def forward(self, x):
        return self.linear(x)   # logits
    
# create one full decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads,attention_dropout,residual_dropout):
        super().__init__()
        self.mha = MHA(d_model, num_heads,attention_dropout)
        self.ln1 = LN(d_model)
        self.ffn = FFN(d_model)
        self.ln2 = LN(d_model)

        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)

    def forward(self, X, return_states=False,padding_mask=None,post_norm=False):
        states = {} if return_states else None

        if post_norm:
            mha_x = self.mha(X, padding_mask=padding_mask)
            res1 = X + self.dropout1(mha_x)
            ln1_x = self.ln1(res1)

            ffn_x = self.ffn(ln1_x)
            res2 = ln1_x + self.dropout2(ffn_x)
            layer_output = self.ln2(res2)

        else:  # pre-norm
            norm_X1 = self.ln1(X)
            mha_x = self.mha(norm_X1, padding_mask=padding_mask)
            res1 = X + self.dropout1(mha_x)

            norm_X2 = self.ln2(res1)
            ffn_x = self.ffn(norm_X2)
            layer_output = res1 + self.dropout2(ffn_x)

        # optionally return intermediate states for analysis or visualization
        if return_states: 
            states["mha_output"] = mha_x.detach()
            states["ffn_output"] = ffn_x.detach()
            states["layer_output"] = layer_output.detach()

            return layer_output, states

        return layer_output

# stacking multiple decoder layers
class Decoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.d_model = config.d_model
        self.post_norm = config.post_norm
        # load dropout values
        self.embedding_dropout = config.embedding_dropout
        self.attention_dropout = config.attention_dropout
        self.residual_dropout = config.residual_dropout

        # embeddings
        self.token_embedding = Embeddings(vocab_size, config.d_model)
        self.positional_encoding = PositionalEncodings(config.max_seq_len, config.d_model)

        # decoder stack
        self.layers = nn.ModuleList(
            [DecoderLayer(config.d_model, config.num_heads, self.attention_dropout, self.residual_dropout)
             for _ in range(config.num_layers)]
        )

        # output projection
        self.logit = Logit(config.d_model, vocab_size)

        # optional weight tying
        if config.weight_tying:
            self.logit.linear.weight = self.token_embedding.embedding.weight
            print(self.logit.linear.weight.data_ptr() == self.token_embedding.embedding.weight.data_ptr())

        # embedding dropout
        self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout)

    def forward(self, input_ids, return_states=False, padding_mask=None):
        states = {} if return_states else None

        emb = self.token_embedding(input_ids)   # lookup token embeddings and try scaling emb * math.sqrt(self.d_model)
        pos = self.positional_encoding(input_ids)       # add positional embeddings
        x = emb + pos
        x = self.embedding_dropout_layer(x)

        if return_states:
            states["x_embed"] = x.detach()

        for i, layer in enumerate(self.layers):         # pass representations through stacked decoder layers
            if return_states:
                x, layer_states = layer(x, return_states=True,padding_mask=padding_mask,post_norm=self.post_norm)
                states[f"layer_{i}"] = layer_states
            else:
                x = layer(x,padding_mask=padding_mask,post_norm=self.post_norm)

        logits = self.logit(x)                          # compute final logits

        if return_states:
            return logits, states

        return logits
