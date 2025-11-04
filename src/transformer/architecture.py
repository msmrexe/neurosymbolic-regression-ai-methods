"""
This script defines the complete Transformer architecture (Cell 37), 
including all sub-modules like Token Embeddings, Positional Encodings, 
Attention, Encoder, and Decoder, with all TODOs completed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

# --- 1. Embedding and Positional Encoding ---

class TokenEmbeddings(nn.Module):
    """
    Converts token indices to dense embedding vectors.
    Embeddings are scaled by sqrt(d_model) as per the "Attention Is All You Need" paper.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of token indices, shape (batch_size, sequence_length)
        Returns:
            Scaled embeddings, shape (batch_size, sequence_length, d_model)
        """
        return self.embedding(x.long()) * math.sqrt(self.d_model)

class PositionalEncodings(nn.Module):
    """
    Injects sinusoidal positional information into the token embeddings.
    """
    def __init__(self, seq_length: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        
        # Term for sine/cosine division
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term) # Even indices
        pe[:, 1::2] = torch.cos(position * div_term) # Odd indices
        
        # Register as buffer (not a model parameter)
        # Add batch dimension: (1, seq_length, d_model)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings, shape (batch_size, seq_length, d_model)
        Returns:
            Embeddings + positional encodings, shape (batch_size, seq_length, d_model)
        """
        # Add positional encoding up to the length of the input sequence
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

# --- 2. Core Attention and MLP Blocks ---

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    """
    def __init__(self, h: int, d_model: int):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h (number of heads)"
        
        self.d_k = d_model // h  # Dimension of each head
        self.h = h
        
        # Use 4 linear layers: 3 for Q, K, V projections, 1 for output
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.p_attn = None # For visualization or debugging

    def forward(self, 
              query: torch.Tensor, 
              key: torch.Tensor, 
              value: torch.Tensor, 
              mask: torch.Tensor = None) -> torch.Tensor:
        
        batch_size = query.size(0)

        # 1) Apply linear projections and split into h heads
        # (batch, seq_len, d_model) -> (batch, h, seq_len, d_k)
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Compute scaled dot-product attention
        # (batch, h, seq_len_q, d_k) @ (batch, h, d_k, seq_len_k) -> (batch, h, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Apply mask (fill with very small number)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        self.p_attn = F.softmax(scores, dim=-1)
        
        # (batch, h, seq_len_q, seq_len_k) @ (batch, h, seq_len_v, d_k) -> (batch, h, seq_len_q, d_k)
        # Note: seq_len_k == seq_len_v
        x = torch.matmul(self.p_attn, value)

        # 3) Concatenate heads and apply final linear layer
        # (batch, h, seq_len_q, d_k) -> (batch, seq_len_q, h, d_k) -> (batch, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        return self.linears[-1](x)

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (Feedforward) block.
    """
    def __init__(self, list_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        for i in range(len(list_dims) - 1):
            layers.append(nn.Linear(list_dims[i], list_dims[i+1]))
            if i < len(list_dims) - 2: # No ReLU or Dropout after final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AddAndNorm(nn.Module):
    """
    A residual connection followed by Layer Normalization.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_input: torch.Tensor, x_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input: The input tensor (from residual connection).
            x_output: The output tensor from the sublayer (e.g., attention).
        """
        return self.norm(x_input + x_output)

# --- 3. Encoder ---

class EncoderLayerMix(nn.Module):
    """
    A custom Encoder layer that mixes MLP and MHA.
    It first applies an MLP to the flattened variable dimension,
    then applies self-attention over the samples.
    """
    def __init__(self, nb_samples: int, max_nb_var: int, d_model: int, h: int, dropout: float):
        super().__init__()
        # MLP to compress (max_nb_var * d_model) features into d_model
        self.first_mlp = MLP([d_model * max_nb_var, d_model, d_model], dropout)
        self.mha = MultiHeadAttention(h, d_model)
        self.add_norm = AddAndNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, nb_samples, max_nb_var, d_model)
        Returns:
            Output tensor, shape (batch_size, nb_samples, d_model)
        """
        batch_size, num_samples, _, _ = x.shape
        
        # Flatten (max_nb_var, d_model) -> (max_nb_var * d_model)
        x_flat = x.view(batch_size, num_samples, -1)
        
        # Apply MLP: (batch, samples, features) -> (batch, samples, d_model)
        x_mlp = self.first_mlp(x_flat)
        
        # Apply MHA on the sample dimension
        x_attn = self.mha(x_mlp, x_mlp, x_mlp) # Self-attention over samples
        
        # Add & Norm
        return self.add_norm(x_mlp, self.dropout(x_attn))

class Encoder(nn.Module):
    """
    The full Encoder, composed of N EncoderLayerMix layers.
    Includes an initial MLP to project input and a final MLP
    before max-pooling for permutation invariance.
    """
    def __init__(self, nb_samples: int, max_nb_var: int, d_model: int, h: int, N: int, dropout: float):
        super().__init__()
        # First MLP projects raw input (y, x1, ...) which has 1 feature
        self.first_mlp = MLP([1, d_model, d_model], dropout)
        
        self.layers = nn.ModuleList(
            [EncoderLayerMix(nb_samples, max_nb_var, d_model, h, dropout) for _ in range(N)]
        )
        
        self.last_mlp = MLP([d_model, d_model, d_model], dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch_size, nb_samples, max_nb_var, 1)
        Returns:
            Encoded representation, shape (batch_size, d_model)
        """
        # 1. Initial projection: (..., 1) -> (..., d_model)
        x = self.first_mlp(x) # (batch, samples, vars, d_model)
        
        # 2. Pass through N encoder layers
        for layer in self.layers:
            x = layer(x) # (batch, samples, d_model)
            
        # 3. Final MLP
        x = self.last_mlp(x)
        
        # 4. Max pooling over sample dimension for permutation invariance
        # (batch, samples, d_model) -> (batch, d_model)
        x_pooled = torch.max(x, dim=1).values 
        
        return x_pooled

# --- 4. Decoder ---

class DecoderLayer(nn.Module):
    """
    A standard Transformer Decoder Layer with two MHA blocks
    (one self-attention, one cross-attention) and one FFN.
    """
    def __init__(self, h: int, d_model: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.cross_attn = MultiHeadAttention(h, d_model)
        
        # Standard feedforward dimension in Transformers
        self.mlp = MLP([d_model, d_model * 4, d_model], dropout)
        
        self.norm1 = AddAndNorm(d_model)
        self.norm2 = AddAndNorm(d_model)
        self.norm3 = AddAndNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
              input_dec: torch.Tensor, 
              mask_dec: torch.Tensor, 
              output_enc: torch.Tensor) -> torch.Tensor:
        
        # 1. Masked Self-Attention
        x = self.norm1(
            input_dec, 
            self.dropout(self.self_attn(input_dec, input_dec, input_dec, mask_dec))
        )
        
        # 2. Cross-Attention
        # output_enc is (batch, d_model). Unsqueeze to (batch, 1, d_model) to act as K, V
        enc_kv = output_enc.unsqueeze(1)
        x = self.norm2(
            x, 
            self.dropout(self.cross_attn(x, enc_kv, enc_kv)) # Q=x, K=V=enc_output
        )
        
        # 3. Feedforward
        x = self.norm3(x, self.dropout(self.mlp(x)))
        
        return x

class Decoder(nn.Module):
    """
    The full Decoder, composed of N DecoderLayers.
    """
    def __init__(self, vocab_size: int, seq_length: int, d_model: int, h: int, N: int, dropout: float):
        super().__init__()
        self.token_emb = TokenEmbeddings(vocab_size, d_model)
        self.pos_enc = PositionalEncodings(seq_length, d_model, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(h, d_model, dropout) for _ in range(N)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
              target_seq: torch.Tensor, 
              mask_dec: torch.Tensor, 
              output_enc: torch.Tensor) -> torch.Tensor:
        
        x = self.token_emb(target_seq)
        x = self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x, mask_dec, output_enc)
            
        return x

# --- 5. Full Transformer Model ---

class TransformerModel(nn.Module):
    """
    The complete Encoder-Decoder Transformer model.
    """
    def __init__(self, 
                 nb_samples: int, 
                 max_nb_var: int, 
                 d_model: int, 
                 vocab_size: int, 
                 seq_length: int, 
                 h: int, 
                 N_enc: int, 
                 N_dec: int, 
                 dropout: float):
        
        super().__init__()
        self.encoder = Encoder(nb_samples, max_nb_var, d_model, h, N_enc, dropout)
        self.decoder = Decoder(vocab_size, seq_length, d_model, h, N_dec, dropout)
        self.final_proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_enc: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training (uses teacher forcing).
        
        Args:
            input_enc: Encoder input, shape (batch, samples, vars, 1)
            target_seq: Decoder input, shape (batch, seq_len)
        
        Returns:
            Logits, shape (batch, seq_len, vocab_size)
        """
        
        # 1. Create decoder mask (combines padding and future mask)
        # Padding mask (ignore 0 tokens)
        pad_mask = (target_seq != 0).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len)
        
        # Future mask
        seq_len = target_seq.size(1)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=target_seq.device), diagonal=1
        ).bool() # (seq_len, seq_len)
        
        # Combined mask (batch, 1, seq_len, seq_len)
        dec_mask = pad_mask & ~future_mask 

        # 2. Encoder processing
        enc_output = self.encoder(input_enc) # (batch, d_model)
        
        # 3. Decoder processing
        dec_output = self.decoder(target_seq, dec_mask, enc_output) # (batch, seq_len, d_model)
        
        # 4. Final projection
        return self.final_proj(dec_output)
