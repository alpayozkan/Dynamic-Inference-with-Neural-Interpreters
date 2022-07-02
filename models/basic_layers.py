import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
  '''
  Given images are linearly embedded via Patch Embedding in order to get tokens.

  Args:
  -----  
    img_size    [int]: Images are assumed to be square
    patch_size  [int]: Images are divided into patches of size `patch size`
    in_channels [int]: Number of input channels of given images
    embed_dim   [int]: Final embedding dimension

  Attributes:
  -----------
    n_patches   [int]:  Number of total patches at the end
    projection  [Conv]: Patch extractor
  '''

  def __init__(self, img_size, patch_size, in_channels, embed_dim):
    super().__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.in_channels = in_channels
    self.embed_dim = embed_dim
    
    self.n_patches = (img_size // patch_size) ** 2
    self.projection = nn.Conv2d(in_channels = in_channels, 
                                out_channels = embed_dim, 
                                kernel_size = patch_size, 
                                stride = patch_size)
  
  def forward(self, x):
    '''
    Args:
    -----
      x [Tensor(B x C x H x W)]: Input images
    
    Returns:
    --------
      projected [Tensor(B x N x E)] where N stands for n_patches & E stands for embed_dim
    '''
    projected = self.projection(x).flatten(2).transpose(1, 2) 
    return projected


  
class MLP(nn.Module):
  '''
  Type Inference MLP module. 
  
  Args:
  ----
    in_features     [int]: Dimension of input features and output features
    hidden_features [int]: Dimension of intermediate features
    out_features    [int]: Dimension of the signature
    
  Returns:
  -------
    t [Tensor()]: Type vector
  '''
  def __init__(self, in_features, hidden_features, out_features):
    super().__init__()
    self.net = nn.Sequential(
              nn.Linear(in_features, hidden_features),
              nn.GELU(),
              nn.Linear(hidden_features, out_features)
              )

  def forward(self, embeddings):
    '''
    Args:
    ----
      embeddings [Tensor(B x N x E)]: 

    Returns:
    --------
      type_vector [Tensor(B x N x S)] where S stands for signature dimension
    '''
    type_vector = self.net(embeddings)
    return type_vector 

  
  
class TypeMatching(nn.Module):
  '''
  Enables the learned routing of input set through functions.
    
    1. Given a set of element x_i, extract its type vector t_i
    2. Compute `Compatibility`
    3. If this compatibility is larger than treshold, permit f_u to access x_i.
  '''
  def __init__(self, in_features, hidden_features, out_features, treshold):
    super().__init__()
    self.treshold = treshold
    self.type_inference = MLP(in_features, hidden_features, out_features)
    self.register_parameter('sigma', nn.Parameter(torch.ones(1)))

  def forward(self, x, s):
    '''
    Args:
    -----
      x [Tensor(B x N x E)]: Embeddings
      s [Tensor(F x S)]
    
    Attributes:
    -----------
      t [Tensor(B x N x S)]
      compatilibity_score [Tensor(B x F x N)]: Parallelized computation score of compatibility score. F stands for # Functions.
      compatilibity_hat   [Tensor(B x F x N)]: Negative exponentiated version of compatibility score
    '''
    t = self.type_inference(x)
    compatibility_hat = self.get_compatilibity_score(t, s)
    
    # Softmax 
    compatibility_norm = compatibility_hat.sum(dim=1).unsqueeze(1) + 1e-5
    compatibility = torch.div(compatibility_hat, compatibility_norm)
    
    return compatibility

  def get_compatilibity_score(self, t, s):
    distance = (1 - t @ s.transpose(0, 1))
    return torch.where(distance > self.treshold, torch.exp(-distance/self.sigma), torch.tensor(0, dtype=torch.float)).transpose(1, 2)

  
class ModLin(nn.Module):
    '''
    Linear Layer conditioned by `code` vector. 
    
    Args:
    ----
      code  [Tensor(dcond x 1)]: Code vector of a `function`.
      dout  [int]: Dimension of the output of the projection.
      din   [int]: Dimension of the input  of the projection.
      dcond [int]: Dimension of the code vector.
    
    Attributes:
    -----------
      W_c [Tensor(din x dcond)]: Projection matrix of condition vector
      b   [Tensor(dout)]:        bias vector 
      W   [Tensor(dout x din)]:  Projection matrix of conditioned vector

    '''
    def __init__(self, code, dout, din, dcond):
      super().__init__()
      self.c = code
      self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
      self.register_parameter('b', nn.Parameter(torch.empty(dout)))
      self.register_parameter('W', nn.Parameter(torch.empty(dout, din)))
      self.norm = torch.nn.LayerNorm(din)

    def forward(self, x):
      '''
      Performs linear projection of embeddings in `din` dimensional space onto
      `dout` dimensional space by fusing [conditioning] embeddings [x] with normalized `code`
      vectors.
      '''  
      out = self.norm(torch.matmul(self.w_c, self.c))
      out = x * out
      out = torch.matmul(out, self.W.transpose(0, 1))+self.b
      return out


class ModLin2D(nn.Module):
    '''
    2D implementation of ModLin: instead of `code` vector, operated on `code` matrix. Used in ModAttention.

    Args:
    ----
      code  [Tensor(dcond x nf)]: Code matrix of a all `function`s.
      dout  [int]: Dimension of the output of the projection.
      din   [int]: Dimension of the input  of the projection.
      dcond [int]: Dimension of the code vector.
    
    Attributes:
    -----------
      W_c [Tensor(din x dcond)]: Projection matrix of condition vector
      b   [Tensor(dout)]:        bias vector 
      W   [Tensor(dout x din)]:  Projection matrix of conditioned vector

    '''
    def __init__(self, code, dout, din, dcond):
      super().__init__()
      self.c = code
      self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
      self.register_parameter('b', nn.Parameter(torch.empty(dout)))
      self.register_parameter('W', nn.Parameter(torch.empty(dout, din)))
      self.norm = nn.LayerNorm(din)

    def forward(self, x):
      '''
      Performs linear projection of embeddings in `din` dimensional space onto
      `dout` dimensional space by fusing [conditioning] embeddings [x] with normalized `code`
      vectors.
      '''  
      out = self.norm(torch.matmul(self.w_c, self.c).T).unsqueeze(1)
      out = x.unsqueeze(1) * out
      out = torch.matmul(out, self.W.transpose(0, 1))+self.b
      return out
   
  
class ModMLP(nn.Module):
  '''
  Combination of ModLin Layers with the GELU activation function.

  Args:
  ----
    n_layers [int]: Number of ModLin layers
    code     [Tensor(dcond x 1)]: Code vector of a `function`.
    dout     [int]: Dimension of the output projection
    din      [int]: Dimension of the input  projection
    dcond    [int]: Dimension of the code vector
    activ    [nn.Module]: Activation function applied after every ModLin Layer
  
  Attributes:
  -----------
    modlin_blocks [List[ModLin]]: Stack of ModLin layers 
  '''
  def __init__(self, n_layers, code, dout, din, dcond, activ=nn.GELU):
      super().__init__()
      self.modlin_blocks = [ModLin(code, dout, din, dcond), activ()]
      
      for i in range(n_layers-1):
        self.modlin_blocks.append(ModLin(code, dout, dout, dcond))
        self.modlin_blocks.append(activ())
     
      self.modlin_blocks = nn.Sequential(*self.modlin_blocks)
  
  def forward(self, x):
      out = self.modlin_blocks(x)
      return out


class ModAttn(nn.Module):
  '''
  Self-Attention Module that uses ModLin2D layer for constructing Q, K, V matrices.
  
  Args:
  ----
    code_matrix [Tensor(dcond x nf)]: Consists of code vectors associated with each `function`
    din         [int]:   Dimension of Embeddings
    dcond       [int]:   Dimension of individual code vectors associated with each function
    n_heads     [int]:   Number of attention heads 
    attn_prob   [float]: Dropout probability 
  '''
  def __init__(self, code_matrix, din, dcond, n_heads, attn_prob = 0.0):
    super().__init__()
    self.C = code_matrix
    self.qkv = ModLin2D(code_matrix, 3 * din, din, dcond)
    self.n_heads = n_heads
    self.head_dim = din // n_heads
    self.scale = self.head_dim ** -0.5

  def forward(self, x):
    B, N, E = x.shape
    # [768, 128, 5, 64]
    qkv = self.qkv(x).permute(3, 0, 1, 2)
    qkv = qkv.view(3, self.n_heads, self.head_dim, B, -1, N) 
    qkv = qkv.permute(0, 3, 1, 4, 5, 2)
    # B x Heads x nf x tokens x token_dim
    q, k, v = qkv[0], qkv[1], qkv[2]
    qk_t = (q @ k.transpose(-2, -1)) * self.scale
    attn = qk_t.softmax(dim=-1)
    # DONT FORGET TO SCALE WITH C_ui . C_uj
    return attn.size()
