import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class MaskedSelfAttention(nn.Module):
    # Class for the Attention Mechanism
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        

    def forward(self, x):
        k = self.key(x)   
        q = self.query(x) 
        v = self.value(x) 
        
        return F.scaled_dot_product_attention(q,k,v)

class MultiHeadAttention(nn.Module):
    # Class for the MultiHeadAttention
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedSelfAttention(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.num_heads = num_heads
        self.head_size = head_size
    def forward(self, x):
        B, T, _ = x.shape
        out = torch.empty(B,T,self.num_heads * self.head_size, device = x.device,dtype=torch.float32)
        # Concatenation of the outputs of the heads
        for i, head in enumerate(self.heads):
            out[:,:,i * self.head_size:(i+1) * self.head_size] = head(x)
            
            
        out = self.proj(out)
        return out

class OneHeadAttention(nn.Module):
    def __init__(self,  head_size, n_embd, block_size):
        super().__init__()
        self.head = MaskedSelfAttention(head_size, n_embd, block_size)
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self,x):
        x = self.head(x)
        out = self.proj(x)
        return out
class TransformerBlock(nn.Module):
    # Class for the Encoder Block
    def __init__(self, n_embd, n_head, block_size, hidden_size):
        super().__init__()
        head_size = n_embd // n_head
        if n_head > 1 :
            self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        else : 
            self.sa = OneHeadAttention(head_size, n_embd, block_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        
        x = x + self.sa(self.ln1(x))
        print(self.ln2(x))
        print(self.ffwd(self.ln2(x)))
        x = x + self.ffwd(self.ln2(x))
        print(f"ffwd : {x}")
        return x

class TransformerModel(nn.Module):
    # Class for the transformer model
    def __init__(self, num_classes, n_embd, n_head, block_size, hidden_size, n_layers):
        super().__init__()
        #self.positional_embeddings = nn.Parameter(torch.zeros(1, block_size, n_embd))
        #self.pos_embedding = PositionalEncoding(n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, hidden_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)  

    def forward(self, x):
        #x = self.token_embedding(x) + self.positional_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x[:,-1,:])
        return logits


class RawAudioTransformerModel(nn.Module):
    def __init__(self, num_classes, n_embd, n_head, block_size, hidden_size, n_layers):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        #self.batch1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)

        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, hidden_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)  
    

    def forward(self, x):
        #x = self.pool(self.relu(self.batch1(self.conv1(x))))
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.adpool(x)
        print(x)
        x = x.squeeze(-1)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
