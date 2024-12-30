import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class InputEmbedding(nn.Module):
    """ Input Embedding for Transformer """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(self, x):
        """ Input Embedding Forward Pass """
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))

class PositionalEncoding(nn.Module):
    """ Positional Encoding for Transformer """
    def __init__(self, d_model: int, seq_len: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)

        #Create a vector of shape (seq_len, 1) and (1, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        #Apply the sin to even postions in the vector
        pe[:, 0::2] = torch.sin(position * div_term)

        #Apply the cos to odd postions in the vector
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """ Positional Encoding Forward Pass """
        x += self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """ Layer Normalization for Transformer """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        """ Layer Normalization Forward Pass"""
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForward(nn.Module):
    """ Feed Forward Network for Transformer """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """ Feed Forward Network Forward Pass """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention for Transformer """
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()

        self.h = h
        self.d_model = d_model
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Query
        self.w_k = nn.Linear(d_model, d_model) #Key
        self.w_v = nn.Linear(d_model, d_model) #Value

        self.w_o = nn.Linear(d_model, d_model) #Output
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """ Scaled Dot-Product Attention """
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = F.softmax(attention_scores, dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask):
        """ Multi-Head Attention Forward Pass """
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        #Split the query, key and value into h heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        #(Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

## Transformer Encoder Layer
