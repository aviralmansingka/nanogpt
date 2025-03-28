from dataclasses import dataclass
from typing import assert_type

import inspect
import math
import os
import time

import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule
import torch.nn as nn
from torch.nn import functional as F

import modal

app = modal.App("gpt2-initial-prompts")
image = modal.Image.debian_slim("3.11.9").pip_install_from_requirements(
    "./requirements.txt"
)
traces = modal.Volume.from_name("traces", create_if_missing=True)


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # # (B, n_head, T, n_embd) x (B, n_head, n_embd, T) = (B, n_head, T, T)
        # with record_function("attn:head"):
        #     attn = q @ k.transpose(-2, -1)
        #     # Scale by 1 / sqrt(num_embd)
        #     attn = attn * (1.0 / math.sqrt(k.size(-1)))
        #     # create mask for preventing future tokens from leaking in
        #     # Use -inf here to remove instead of zero as next step is softmax
        #     attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        #     # Softmax over each column
        #     attn = F.softmax(attn, dim=-1)
        #     # (B, num_head, T, T) x (B, num_head, T, dim_attn)
        #     # = (B, num_head, T, dim_attn)
        #     y = attn @ v

        with record_function("attn:head"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble all attention heads
        # Map it back from dim_attn to dim_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        with record_function("Block::attn"):
            x = x + self.attn(self.ln_1(x))
        with record_function("Block::mlp"):
            x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # context length
    vocab_size: int = 50304  # 50K BPE, 256 byte tokens, 1 Special (EOT)
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
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is only {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        with record_function("GPT::embeddings"):
            # Static position embeddings irrespective of inputs
            # position embedding (T, n_embd)
            pos_emb = self.transformer.wpe(pos)
            # token embeddings (B, T, n_embd)
            tok_emb = self.transformer.wte(idx)
            # Broadcast pos_emb and add it to each prompt in B
            x = tok_emb + pos_emb

        with record_function("GPT::transformers"):
            for block in self.transformer.h:
                # Successively apply transformer blocks
                x = block(x)

            # forward the final layernorm and the classifier
            x = self.transformer.ln_f(x)

        with record_function("GPT::lm_head"):
            # (B, T, vocab_size)
            logits = self.lm_head(x)

        with record_function("GPT::loss"):
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the candidate params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters with 2D will be decayed
        # i.e. all biases and all layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed params: {len(decay_params)}, params: {num_decay_params}")
        print(
            f"num nodecayed params: {len(nodecay_params)}, params: {num_nodecay_params}"
        )
        # Use kernel fusion for loss calculation
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_group, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)}")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


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


MAX_LR = 3e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 50


def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    # 2) If it > lr_decay_iters, return min learning rate
    if it > MAX_STEPS:
        return MIN_LR
    # 3) In between, use cosine decay down to min_lr
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


@app.function(gpu="A100", image=image, volumes={"/mnt/traces": traces})
def run_model():
    if not os.path.exists("input.txt"):
        import requests

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open("input.txt", "w") as f:
            f.write(requests.get(url).text)
            print("Downloaded tiny shakespeare dataset")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        torch.mps.manual_seed(1337)
    elif torch.backends.cuda.is_built():
        device = "cuda"
        torch.cuda.manual_seed(1337)
    else:
        device = "cpu"

    train_loader = DataLoaderLite(B=16, T=1024)

    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    model: GPT = torch.compile(model)

    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_modules=True,
        with_flops=True,
        record_shapes=True,
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
    ) as prof:
        prof.start()
        optimizer.zero_grad()
        for step in range(10):
            t0 = time.time()
            with record_function("load_data"):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
            with record_function("next_token_pred"):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)

            with record_function("grad_update"):
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            with record_function("cleanup"):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif torch.mps.is_available():
                    torch.mps.synchronize()
                t1 = time.time()
                dt = (t1 - t0) * 1000
                tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
                print(
                    f"step {step} | loss: {loss.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
                )
                prof.step()

    prof.export_chrome_trace(f"/mnt/traces/trace_{time.strftime('%Y%m%d_%H%M%S')}.json")


@app.local_entrypoint()
def trigger_on_modal():
    run_model.remote()


if __name__ == "__main__":
    run_model.local()
