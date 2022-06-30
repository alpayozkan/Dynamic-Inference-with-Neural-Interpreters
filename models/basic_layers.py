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
    self.sigma = nn.Parameter(torch.ones(1))

  def forward(self, x, s):
    '''
    Args:
    -----
      x [Tensor(B x N x E)]: Embeddings
      s [Tensor(F x S)]
    
    Attributes:
      t [Tensor(B x N x S)]
      compatilibity_score [Tensor(B x F x N)]: Parallelized computation score of compatibility score. F stands for # Functions.
      compatilibity_hat   [Tensor(B x F x N)]: Negative exponentiated version of compatibility score
    '''
    t = self.type_inference(x)
    compatibility_score = self.get_compatilibity_score(t, s).transpose(1, 2)
    compatibility_hat = torch.exp(-compatibility_score / self.sigma)
    return compatibility_hat

  def get_compatilibity_score(self, t, s):
    distance = (1 - t @ s.transpose(0, 1))
    return torch.where(distance > self.treshold, distance, torch.tensor(0, dtype=torch.float))
