import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import math

class Flatten(nn.Module):
    
    def __init__(self, start_dim, end_dim):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    def __call__(self, x):
        return x.flatten(self.start_dim, self.end_dim)
    

# Utilise CNN module to pass over image and produce patches
class PatchEmbedding(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, n_embd=768): # Hyperparameters from paper
        super().__init__()
        
        self.patch_size = patch_size
        # Conv2D layer acts as the learner
        self.patcher = nn.Conv2d(
                            in_channels = in_chans,
                            out_channels = n_embd,
                            kernel_size = patch_size, # kernal of patch size
                            stride = patch_size, # stride ensures patches dont overlap
                            padding = 0
                            )
        
        # flatten dims for transformer
        self.flatten = Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        # asserts suitable patch size is chosen
        img_res = x.shape[-1]
        assert img_res % self.patch_size == 0, f'input image size must be exactly divisable by patch size'
        
        # make and flatten patches for transformer
        x = self.patcher(x)
        x = self.flatten(x)
        
        
        return x.permute(0, 2, 1) # swaps last 2 dims to adhear to (batch, num_patches, Embedding[P^2•C])
    

class Linear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super().__init__()
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn((in_features, out_features)) * (in_features**-0.5)) # in_features**-1 = K
        #use kaiming initialization to restrict weight dist between -sqrt(K) and sqrt(K)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
    def __call__(self, x):
        # x•W + b
        self.out = x @ self.weight
        if self.bias is not None:
              self.out += self.bias
        return self.out
    

class Head(nn.Module):
    
    def __init__(self, head_size, n_embd=768, dropout=0.1):
        super().__init__()
        
        self.head_size = head_size

        # Initialize keys, queries, values, and dropout layers
        # Bias required to be true to match ViT pretrained parameters
            # Bias = False provides better and faster performance
        self.key = Linear(n_embd, head_size, bias=True)
        self.query = Linear(n_embd, head_size, bias=True)
        self.value = Linear(n_embd, head_size, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # apply self-attention as described in Attention is all you need
        
        # create keys and values
        k = self.key(x)
        q = self.query(x)
        
        # Produce weight matrix from q and k communication
        w = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        
        # use value layer to produce outputs
        v = self.value(x)
        out = w @ v
        return out
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size, n_embd=768, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Concat head outputs along final dimention
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        # Apply projection (not in paper but helps performance)
        out = self.dropout(self.proj(out))
        return out
    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
  
    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        
        return self.out
    

def gelu(x):
    # gelu activation function
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):
    
    def __init__(self, n_embd, mlp_size=3072, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            Linear(n_embd, mlp_size),
            nn.GELU(),
            Linear(mlp_size, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        
        return self.net(x)
    

class EncoderBlock(nn.Module):
    
    def __init__(self, n_embd, n_head, mlp_size=3072):
        super().__init__()
        head_size = n_embd // n_head
        # Multi-Head Attention
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # MLP
        self.mlp = MLP(n_embd, mlp_size)
        
        # both layernorms
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
    
    def forward(self, x):
        # residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class vit(nn.Module):
    
    def __init__(self, n_embd = 768, encoder_blocks = 12, output_size=1000, patch_size=16, img_size=224):
        super(vit, self).__init__()
        
        # Calculate number of patches
        n_patches = (img_size**2) // patch_size**2
        
        # CNN Patch Embedding
        self.embedding = PatchEmbedding(in_chans = 3, patch_size = 16, n_embd = 768)
        
        # Token needs to be applied across batch so is expanded in forward
        self.class_emb = nn.Parameter(torch.randn((1, 1, n_embd)), requires_grad=True)
        
        # Positional Embedding (no need for sin cos positional embedding implementation as n_embd is small see Appendix)
        self.pos_emb = nn.Parameter(torch.randn((1, n_patches+1, n_embd)), requires_grad=True)
        
        self.dropout = nn.Dropout(0.1)
        
        # Build encoder stack
        self.encoder = nn.Sequential(*[EncoderBlock(n_embd = 768, n_head = 12, mlp_size = 3072) for i in range(encoder_blocks)])
        
        # Layer Norm + MLP for final prediction
        self.norm = nn.LayerNorm(normalized_shape=n_embd)        
        self.out = nn.Linear(n_embd, output_size)
    
    def forward(self, x):
        
        batch_dim = x.shape[0]
        x = self.embedding(x)
        
        # Expand across batch
        class_tokn = self.class_emb.expand(batch_dim, -1, -1)
        x = torch.cat((x, class_tokn), 1)

        x = self.pos_emb + x

        x = self.dropout(x)
        x = self.encoder(x)

        # pass embedding token slice to final MLP
        x = self.out(self.norm(x[:, 0, :]))

        
        return x