from dataclasses import dataclass
from typing import assert_type

import math
import os

import torch
from torch._prims_common import DeviceLikeType
from torch.distributed import is_available
import torch.nn as nn
from torch.nn import functional as F

import modal

app = modal.App("gpt2-initial-prompts")
image = modal.Image.debian_slim("3.11.9").pip_install_from_requirements(
    "./requirements.txt"
)


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # holds K, Q, V_down (each n_embd x n_embd) over columns
        # V_down to go from token-space into attention-space for values
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # holds V_up to go from attention-space back to token-space
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # this is the attention mask for preventing future tokens
        # called "bias" for historical reasons
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # Assume x is ([B]atch x [T]okens x [C]hannels)
        # where:
        #   channels -> dimensionality of embedding
        B, T, C = x.size()

        # efficient way of multiplying x against W_q, W_k, W_vdown
        qkv = self.c_attn(x)

        # dim=2 means to split among columns (0,1,[2])
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Convert all to (B, n_head, T, n_embd)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # (B, n_head, T, n_embd) x (B, n_head, n_embd, T) = (B, n_head, T, T)
        attn = q @ k.transpose(-2, -1)
        # Scale by 1 / sqrt(num_embd)
        attn = attn * (1.0 / math.sqrt(k.size(-1)))
        # create mask for preventing future tokens from leaking in
        # Use -inf here to remove instead of zero as next step is softmax
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # Softmax over each column
        attn = F.softmax(attn, dim=-1)
        # (B, num_head, T, T) x (B, num_head, T, dim_attn)
        # = (B, num_head, T, dim_attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
        # Re-assemble all attention heads
        # Map it back from dim_attn to dim_embd


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # context length
    vocab_size: int = 50257  # 50K BPE, 256 byte tokens, 1 Special (EOT)
    n_layer: int = 12  # how many consecutive transformer blocks
    n_head: int = 12  # how many self-attention heads per transformer
    n_embd: int = 768  # dimentionality of embeddings


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is only {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # Static position embeddings irrespective of inputs
        # position embedding (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # token embeddings (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        # Broadcast pos_emb and add it to each prompt in B
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            # Successively apply transformer blocks
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def _generate_next_token(model: nn.Module, x: torch.Tensor):
    __import__("ipdb").set_trace()
    # (B, T, vocab_size)
    logits, _ = model(x)
    # (B, vocab_size) using last token
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    # filter out only top-50 to "steer" model
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

    # predict next token using the top ones
    ix = torch.multinomial(topk_probs, 1)  # (B, 1)

    # (B, 1)
    xcol = torch.gather(topk_indices, -1, ix)

    x = torch.cat((x, xcol), dim=1)


@app.function(gpu="L40S", image=image)
def run_model():
    NUM_SEQUENCES = 5
    MAX_LENGTH = 30

    if not os.path.exists("input.txt"):
        import requests

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open("input.txt", "w") as f:
            f.write(requests.get(url).text)
            print("Downloaded tiny shakespeare dataset")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    with open("input.txt", "r") as f:
        text = f.read()

    text = text[:1000]
    tokens = enc.encode(text)

    B, T = 4, 32
    buf = torch.tensor(tokens[: B * T + 1]).to(device)
    x = buf[:-1].view(B, T).to(device)
    y = buf[1:].view(B, T).to(device)
    logits, loss = model(x, targets=y)
    print(logits.shape)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(50):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")

    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(NUM_SEQUENCES, 1)
    x = tokens.to(device)

    while x.size(1) < MAX_LENGTH:
        with torch.no_grad():
            _generate_next_token(model, x)

    for i in range(NUM_SEQUENCES):
        tokens = x[i, :MAX_LENGTH].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


@app.local_entrypoint()
def trigger_on_modal():
    run_model.remote()


if __name__ == "__main__":
    run_model.local()
