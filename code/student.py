"""MP1 submission model: a small GPT modernized for the WikiText-2 / BPE-2048 benchmark.

Final frozen architecture (config: configs/final.json; see RUN_LOG.csv and the report):
  width 192, depth 8, heads 6 (head_dim 32), context 256, vocab 2048,
  GELU MLP hidden 768 (parameter-matched to SwiGLU-512), dropout 0.15,
  tied embedding, 3,935,936 parameters.

Mechanisms (each validated by paired ablation on the validation split):
  RMSNorm      pre-norm everywhere; statistics computed in fp32 for bf16 safety
  RoPE         rotary positions on q/k, rotate-half convention (no position
               table); use_rope=False falls back to a learned absolute table
  QK-Norm      per-head RMSNorm(head_dim) on q and k before attention
  MLP switch   mlp='swiglu' (default): down(silu(gate(x)) * up(x)), hidden=mlp_hidden
               mlp='gelu':               down(gelu(up(x))), hidden=mlp_hidden*3//2
               (parameter-matched). The FINAL model uses 'gelu': at this scale it
               beat SwiGLU by 0.012 val BPB -- an ablation-driven design reversal.
  dropout      SDPA attention dropout + both residual-branch outputs; train-only
               (nn.Dropout and the SDPA flag are inactive under model.eval())
  init         Linear/Embedding ~ N(0, 0.02); residual outputs (proj/down) re-drawn
               at 0.02/sqrt(2*depth); head tied to the embedding table last

Config keys: vocab/width/heads/depth/context (required). Optional with defaults:
  mlp_hidden=4*width, mlp='swiglu', use_rope=True, use_qk_norm=True, dropout=0.0.
  tests/test_contract.py builds the model without the optional keys, so the
  defaults must always produce a working full model.

Harness interfaces (checked by tests/test_contract.py and evaluate.score):
  build_model(config) -> nn.Module with attribute .context == 256
  forward(ids)           -> unnormalized logits [B, T, 2048]
  predict_log_probs(ids) -> fp32 normalized log-probs [B, T, 2048]; no state is
                            carried across calls, examples or windows
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean centering, no bias).

    y = x / sqrt(mean(x^2) + eps) * weight, with the statistics computed in
    fp32 and the result cast back to the input dtype (bf16-safe).
    """

    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def rope_tables(context, head_dim, base=10000.0):
    """Precompute RoPE cos/sin tables, each of shape [context, head_dim].

    Channel pair (j, j + head_dim/2) is rotated by angle t * theta_j at
    position t, where theta_j = base^(-2j/head_dim) spans high-to-low
    frequencies. Each angle is duplicated into both halves of the table to
    match the rotate-half pairing used by apply_rope.
    """
    i = torch.arange(0, head_dim, 2).float()
    thetas = base ** (-i / head_dim)
    t = torch.arange(context).float()
    angles = t[:, None] * thetas[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate q/k per position: out = x*cos + rotate_half(x)*sin.

    x: [B, heads, T, head_dim]; cos/sin: [T, head_dim], already sliced to T
    and cast to x.dtype by the caller. The rotation preserves vector norms
    and makes q.k depend only on the relative position between tokens.
    """
    d = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + x_rot * sin


class Block(nn.Module):
    """Pre-norm transformer block: QK-Norm + RoPE attention, then MLP.

    forward flow (x: [B, T, width]):
      h  = norm1(x) -> qkv -> per-head q/k: QK-Norm, then RoPE if enabled
      x  = x + drop(proj(SDPA(q, k, v, causal)))
      h2 = norm2(x)
      x  = x + drop(down(silu(gate(h2)) * up(h2)))   [mlp='swiglu']
      x  = x + drop(down(gelu(up(h2))))              [mlp='gelu']

    All Linear layers are bias-free. RoPE tables are non-persistent buffers:
    excluded from state_dict, but they follow the module across .to(device).
    """

    def __init__(self, width, heads, mlp_hidden, context,
                 use_rope=True, mlp='swiglu', use_qk_norm=True, dropout=0.1):
        # Constructor defaults are only used if Block is built directly;
        # StudentGPT always passes explicit values parsed from the config.
        super().__init__()
        self.heads = heads
        self.use_rope, self.mlp_kind, self.dropout = use_rope, mlp, dropout
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        head_dim = width // heads
        self.q_norm = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if use_qk_norm else nn.Identity()
        if mlp == 'swiglu':
            self.gate, self.up = nn.Linear(width, mlp_hidden, bias=False), nn.Linear(width, mlp_hidden, bias=False)
            self.down = nn.Linear(mlp_hidden, width, bias=False)
        else:
            self.up = nn.Linear(width, mlp_hidden * 3 // 2, bias=False)
            self.down = nn.Linear(mlp_hidden * 3 // 2, width, bias=False)
        if use_rope:
            cos, sin = rope_tables(context, head_dim)
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3,
                        self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q = apply_rope(self.q_norm(q), self.cos[:length].to(q.dtype), self.sin[:length].to(q.dtype))
            k = apply_rope(self.k_norm(k), self.cos[:length].to(k.dtype), self.sin[:length].to(k.dtype))
        else:
            q, k = self.q_norm(q), self.k_norm(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(batch, length, width)))
        h = self.norm2(x)
        if self.mlp_kind == 'swiglu':
            return x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x + self.drop(self.down(F.gelu(self.up(h))))


class StudentGPT(nn.Module):
    """Assembly: token embedding (+ optional position table) -> Blocks -> RMSNorm -> tied head.

    Initialization order matters and is deliberate:
      1. all Linear/Embedding weights ~ N(0, 0.02)
      2. residual outputs (block.proj, block.down) re-drawn at 0.02/sqrt(2*depth)
         so every branch starts near-transparent (GPT-2 scaled initialization)
      3. head.weight tied to token.weight LAST -- tying earlier would let any
         later re-initialization of the head silently rewrite the shared table
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        use_rope = config.get("use_rope", True)
        mlp = config.get("mlp", 'swiglu')
        use_qk_norm = config.get("use_qk_norm", True)
        dropout = config.get("dropout", 0.0)
        width, depth = config['width'], config['depth']
        mlp_hidden = config.get('mlp_hidden', 4 * width)
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = None if use_rope else nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([Block(width, config["heads"], mlp_hidden, self.context,
             use_rope=use_rope, mlp=mlp, use_qk_norm=use_qk_norm, dropout=dropout) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)

        self.apply(self.initialize)
        residual_std = 0.02 / math.sqrt(2 * depth)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.down.weight, std=residual_std)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [B, T, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: fp32 normalized log-probs; stateless per call."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
