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

  def __init__(self, img_size, patch_size, in_channels, embed_dim, n_cls):
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
    
    self.cls_tokens = nn.Parameter(torch.zeros(1, n_cls, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, n_cls + self.n_patches, embed_dim))
  
  def forward(self, x):
    '''
    Args:
    -----
      x [Tensor(B x C x H x W)]: Input images
    
    Returns:
    --------
      projected [Tensor(B x N + CLS x E)] where N + CLS stands for n_patches + cls tokens & E stands for embed_dim
    '''
    batch_size = x.size(0)
    x = self.projection(x).flatten(2).transpose(1, 2) 
    cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    return x


  
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
  def __init__(self, type_inference, funcSign, threshold):
    super().__init__()
    self.s = funcSign
    self.threshold = threshold
    self.type_inference = type_inference
    self.register_parameter('sigma', nn.Parameter(torch.ones(1)))

  def forward(self, x):
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
    compatibility_hat = self.get_compatilibity_score(t, self.s)
    
    # Softmax 
    compatibility_norm = compatibility_hat.sum(dim=1).unsqueeze(1) + 1e-5
    compatibility = torch.div(compatibility_hat, compatibility_norm)
    
    return compatibility

  def get_compatilibity_score(self, t, s):
    distance = (1 - t @ s.transpose(0, 1))
    M = distance > self.threshold
    out = torch.exp(-distance/self.sigma)*M
    return out.transpose(1, 2)


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
    def __init__(self, code, dout, din, dcond, w_c, W, b):
      super().__init__()
      self.c = code
      
      self.w_c = w_c
      # self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
      
      # interpreter
      self.b = b
      self.W = W
      # self.register_parameter('b', nn.Parameter(torch.empty(dout)))
      # self.register_parameter('W', nn.Parameter(torch.empty(dout, din)))
      
      self.norm = nn.LayerNorm(din)

    def forward(self, x):
      '''
      Performs linear projection of embeddings in `din` dimensional space onto
      `dout` dimensional space by fusing [conditioning] embeddings [x] with normalized `code`
      vectors.
      '''  
      out = self.norm(torch.matmul(self.w_c, self.c).T).unsqueeze(1)
      out = x * out
      out = torch.matmul(out, self.W.transpose(0, 1))+self.b
      return out
   
  
class ModMLP(nn.Module):
  '''
  Combination of ModLin Layers with the GELU activation function.

  Args:
  ----
    mlp_depth [int]: Number of ModLin layers
    code     [Tensor(dcond x 1)]: Code vector of a `function`.
    dout     [int]: Dimension of the output projection
    din      [int]: Dimension of the input  projection
    dcond    [int]: Dimension of the code vector
    activ    [nn.Module]: Activation function applied after every ModLin Layer
  
  Attributes:
  -----------
    modlin_blocks [List[ModLin]]: Stack of ModLin layers 
  '''
  def __init__(self, mlp_depth, code, dout, din, dcond, w_c, W, b, activ=nn.GELU):
      super().__init__()
      self.modlin_blocks = [ModLin2D(code, dout, din, dcond, w_c, W, b), activ()]
      
      for i in range(mlp_depth-1):
        self.modlin_blocks.append(ModLin2D(code, dout, dout, dcond, w_c, W, b))
        self.modlin_blocks.append(activ())
     
      self.modlin_blocks = nn.Sequential(*self.modlin_blocks)
  
  def forward(self, x):
      out = self.modlin_blocks(x)
      return out

  
class ModAttn(nn.Module):
  '''
  Parallelized Self-Attention Layer. 
  
  Args:
  ----
    code_matrix [Tensor(dcond x nf)]: Code matrix of a all `function`s.
    din         [int]: Dimension of the input  projection
    dcond       [int]: Dimension of the code vector
    n_heads     [int]: Number of attention heads
    attn_prob   [float]: Drop-out rate
    proj_prob   [float]: Drop-out rate
  
  Arguments:
  ----------
    qkv:      ModLin Layer to obtain Q, K, V matrices
    head_dim: Dimension of each attention head
    scale:    Scale factor of qk_T
  
  Returns:
  --------
    y: Tensor of size [B x nf x n_token x din]
  '''
  def __init__(self, 
              code_matrix, 
              din, 
              dcond, 
              n_heads, 
              w_c, 
              W, 
              b, 
              W_qkv, 
              b_qkv, 
              attn_prob = 0.0, 
              proj_prob = 0.0):

    super().__init__()
    self.C = code_matrix
    self.qkv = ModLin2D(code_matrix, 3 * din, din, dcond, w_c, W_qkv, b_qkv)
    self.n_heads = n_heads
    self.head_dim = din // n_heads
    self.scale = self.head_dim ** -0.5
    self.proj = ModLin2D(code_matrix, din, din, dcond, w_c, W, b)
    self.attn_drop = nn.Dropout(attn_prob)
    self.proj_drop = nn.Dropout(proj_prob)
    # code, dout, din, dcond

  def forward(self, x, compatibility):
    B, N, E = x.shape
    # [768, 128, 5, 64]
    qkv = self.qkv(x.unsqueeze(1)).permute(3, 0, 1, 2)
    qkv = qkv.view(3, self.n_heads, self.head_dim, B, -1, N) # 3 x 4 x 256 x 128 x 5 x 64
    qkv = qkv.permute(0, 3, 1, 4, 5, 2)
    # B x Heads x nf x tokens x token_dim
    q, k, v = qkv[0], qkv[1], qkv[2]
    qk_t = (q @ k.transpose(-2, -1)) * self.scale
    attn = qk_t.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Create compatibility matrix
    compat_matrix = compatibility.transpose(1, 2) @ compatibility
    W_hat = attn * compat_matrix.unsqueeze(1).unsqueeze(1)
    W = W_hat.softmax(dim = -1)
    y_hat = (W @ v).permute(0, 2, 3, 1, 4) # [B x nf x n_tokens x n_heads x head_dim]
    y_hat = y_hat.flatten(3)  
    # Mix 
    y = self.proj(y_hat).squeeze(1)
    y = self.proj_drop(y)
    return y

class LOC(nn.Module):
  '''
  Line of Code Layer Composed of 1 attention + 1 MLP layers

  Args:
  -----
    code_matrix [Tensor()]:  Code matrix of a all `function`s.
    din         [int]:       Dimension of the input  projection
    dcond       [int]:       Dimension of the code vector
    n_heads     [int]:       Number of attention heads
    mlp_depth   [int]:       Number of MLP depths of LOC layer
    typematch   [nn.Module]: TypeMatching Module
    w_c         [Tensor()]:  Projection Matrix of `code` vector
    W           [Tensor()]:  Projection Matrix of Fusion operation 
    b           [Tensor()]:  Bias vector of Fusion operation
    W_qkv       [Tensor()]:  Weight matrix 
    b_qkv       [Tensor()]:  Bias vector
    attn_prob   [float]:     Drop-out rate
    proj_prob   [float]:     Drop-out rate

  '''
  def __init__( self, 
                code_matrix, 
                din, 
                dcond, 
                n_heads, 
                mlp_depth, 
                typematch, 
                w_c, 
                W, 
                b, 
                W_qkv, 
                b_qkv,
                attn_prob=0, 
                proj_prob=0):

    super().__init__()

    self.typematch = typematch
    self.norm1 = torch.nn.LayerNorm(din)
    self.norm2 = torch.nn.LayerNorm(din)

    self.modattn = ModAttn( code_matrix, 
                            din, 
                            dcond, 
                            n_heads, 
                            w_c, 
                            W, 
                            b, 
                            W_qkv, 
                            b_qkv,
                            attn_prob, 
                            proj_prob)

    self.modmlp = ModMLP( mlp_depth, 
                          code_matrix, 
                          din, 
                          din, 
                          dcond,
                          w_c, 
                          W, 
                          b)

  def forward(self, x):
    compat_matrix = self.typematch(x)
    # x = x.squeeze()
    x_norm = self.norm1(x)
    a_hat = self.modattn(x_norm, compat_matrix)
    
    compat_matrix = compat_matrix.unsqueeze(-1)
    a = x.unsqueeze(1) + compat_matrix*a_hat
    
    b_hat = self.modmlp(self.norm2(a))
    y = a + compat_matrix*b_hat
    
    # pool-LOC => eqn-11
    y = x + torch.sum(compat_matrix*y, dim=1)
    return y


class Script(nn.Module):
  '''
  Script blocks composed of LOC blocks
  
  Assumption:
  -----------
    LOC is composed of 1 layer.
  
  Args:
  -----
    ni            [int]:       Number of function iterations in a script
    nf            [int]:       Number of functions per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    n_heads       [int]:       Number of attention heads
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    typematch     [nn.Module]: TypeMatching Module
    treshold      [int]:       Treshold value for routing 
    code_dim      [int]:       Dimension of `code` vector
    signature_dim [int]:       Dimension of `signature` vector
    W             [Tensor()]:  Projection Matrix of Fusion operation 
    b             [Tensor()]:  Bias vector of Fusion operation
    W_qkv         [Tensor()]:  Weight matrix 
    b_qkv         [Tensor()]:  Bias vector
    attn_prob     [float]:     Drop-out rate
    proj_prob     [float]:     Drop-out rate
  '''
  
  def __init__( self, 
                ni, 
                nf, 
                din, 
                dcond, 
                n_heads, 
                mlp_depth, 
                type_inference, 
                threshold, 
                code_dim, 
                signature_dim,
                W, 
                b, 
                W_qkv, 
                b_qkv,
                attn_prob=0, 
                proj_prob=0):

    super().__init__()
    
    # w_c shared among all functions in a script  
    self.register_parameter('w_c', nn.Parameter(torch.empty(din, dcond)))
    nn.init.xavier_normal_(self.w_c)

    # high-entropy & fixed function signature => avoid mode collapse
    self.register_parameter('funcsign_matrix', nn.Parameter(torch.ones(nf, signature_dim)))
    nn.init.xavier_normal_(self.funcsign_matrix)
    
    self.register_parameter('code_matrix', nn.Parameter(torch.empty(code_dim, nf)))
    nn.init.xavier_normal_(self.code_matrix)

    self.typematch = TypeMatching(type_inference, self.funcsign_matrix, threshold)

    self.locBlocks = []
    for i in range(ni):
      # add LOC layer
      self.loc_blocks.append(LOC(self.
                                code_matrix, 
                                din, 
                                dcond, 
                                n_heads, 
                                mlp_depth, 
                                self.typematch, 
                                self.w_c, 
                                W, 
                                b, 
                                W_qkv, 
                                b_qkv, 
                                attn_prob, 
                                proj_prob))
      
    self.loc_blocks = nn.Sequential(*self.loc_blocks)
  
  def forward(self, x):
    x = self.loc_blocks(x)
    return x
    

class NeuralInterpreter(nn.Module):
  '''
  Neural Interpreter layer that operates the logic

  Args:
  -----
    ns            [int]:       Number of `scripts`
    ni            [int]:       Number of `function iterations` in a script
    nf            [int]:       Number of `functions` per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    n_heads       [int]:       Number of attention heads
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    treshold      [int]:       Treshold value for routing 
    code_dim      [int]:       Dimension of `code` vector
    signature_dim [int]:       Dimension of `signature` vector
    treshold      [int]:       Treshold value for routing
    attn_prob     [float]:     Drop-out rate
    proj_prob     [float]:     Drop-out rate
  '''
  
  def __init__( self, 
                ns, 
                ni, 
                nf, 
                din, 
                dcond, 
                mlp_depth, 
                nheads,
                type_inference_width, 
                signature_dim, 
                threshold,  
                code_dim, 
                attn_prob=0, 
                proj_prob=0, 
              ):

    super().__init__()
    
    self.register_parameter('W', nn.Parameter(torch.empty(din, din)))
    nn.init.xavier_normal_(self.W)
    
    self.register_parameter('b', nn.Parameter(torch.empty(din)))
    nn.init.normal_(self.b)
    
    self.register_parameter('W_qkv', nn.Parameter(torch.empty(3*din, din)))
    nn.init.xavier_normal_(self.W_qkv)
    
    self.register_parameter('b_qkv', nn.Parameter(torch.empty(3*din)))
    nn.init.normal_(self.b_qkv)

    # type inference 
    self.type_inference = MLP(din, type_inference_width, signature_dim)

    self.script_blocks = []
    for i in range(ns):
      # Create Script Blocks
      self.script_blocks.append(Script( ni, 
                                        nf, 
                                        din, 
                                        dcond, 
                                        nheads, 
                                        mlp_depth, 
                                        self.type_inference, 
                                        threshold, 
                                        code_dim, 
                                        signature_dim,
                                        self.W, 
                                        self.b, 
                                        self.W_qkv, 
                                        self.b_qkv,
                                        attn_prob,
                                        proj_prob ))

    self.script_blocks = nn.Sequential(*self.script_blocks)

  def forward(self, x):
    return self.script_blocks(x)

class NeuralInterpreter_vision(nn.Module):
  '''

  Args:
  -----
    ns            [int]:       Number of `scripts`
    ni            [int]:       Number of `function iterations` in a script
    nf            [int]:       Number of `functions` per iteration
    din           [int]:       Dimension of the input  projection
    dcond         [int]:       Dimension of the code vector
    mlp_depth     [int]:       Number of MLP depths of LOC layer
    n_heads       [int]:       Number of attention heads
    signature_dim [int]:       Dimension of `signature` vector
    treshold      [int]:       Treshold value for routing
    code_dim      [int]:       Dimension of `code` vector
    n_classes     [int]:       Number of classification heads
    attn_prob     [float]:     Drop-out rate
    proj_prob     [float]:     Drop-out rate
  '''
  def __init__( self, 
                ns, 
                ni, 
                nf, 
                din, 
                dcond, 
                mlp_depth, 
                nheads,
                type_inference_width, 
                signature_dim, 
                threshold, 
                code_dim, 
                n_classes=10,
                attn_prob=0, 
                proj_prob=0
              ) 

    super().__init__()
    self.ni_model = NeuralInterpreter( ns, 
                                       ni, 
                                       nf, 
                                       din, 
                                       dcond, 
                                       mlp_depth, 
                                       nheads,
                                       type_inference_width, 
                                       signature_dim, 
                                       threshold,  
                                       code_dim, 
                                       attn_prob=0, 
                                       proj_prob=0)

    self.cls_head = nn.Linear(din, n_classes)
  
  def forward(self, x):
    x = self.ni_model(x)
    # first cls taken
    x = x[:,0,:] 
    x = self.cls_head(x)
    return x
