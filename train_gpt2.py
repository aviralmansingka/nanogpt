from dataclasses import dataclass

import argparse
import inspect
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.nn import functional as F
import torch.multiprocessing as mp

import modal

app = modal.App("gpt2-initial-prompts")
image = modal.Image.debian_slim("3.11.9").pip_install_from_requirements(
    "./requirements.txt"
)

# Create a volume to store the FineWeb dataset
fineweb_volume = modal.Volume.from_name(
    "fineweb-volume", create_if_missing=True, size=80
)


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
        # attn = q @ k.transpose(-2, -1)
        # # Scale by 1 / sqrt(num_embd)
        # attn = attn * (1.0 / math.sqrt(k.size(-1)))
        # # create mask for preventing future tokens from leaking in
        # # Use -inf here to remove instead of zero as next step is softmax
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # # Softmax over each column
        # attn = F.softmax(attn, dim=-1)
        # # (B, num_head, T, T) x (B, num_head, T, dim_attn)
        # # = (B, num_head, T, dim_attn)
        # y = attn @ v

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
        x = x + self.attn(self.ln_1(x))
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


class FineWebDataLoader:
    def __init__(self, B, T, rank=0, world_size=1, data_dir=None):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.data_dir = data_dir

        # Check if we're in a testing environment
        local_debug = os.environ.get("LOCAL_TEST", "0") == "1"

        if local_debug:
            # Create a simple random token buffer for local testing
            print(f"Rank {rank}: Using synthetic data for local testing")
            self.token_buffer = torch.randint(0, 50257, (B * T * 20 + 1,))
            self.current_position = 0
        elif self.data_dir and os.path.exists(os.path.join(self.data_dir, "tokens.pt")):
            # Load pre-tokenized data from volume
            print(f"Rank {rank}: Loading tokenized data from volume at {self.data_dir}")
            token_path = os.path.join(self.data_dir, f"tokens_shard_{rank}.pt")
            if not os.path.exists(token_path):
                token_path = os.path.join(self.data_dir, "tokens.pt")

            try:
                self.token_buffer = torch.load(token_path)
                self.current_position = 0
                print(
                    f"Rank {rank}: Loaded {len(self.token_buffer)} tokens from volume"
                )
                print(
                    f"Rank {rank}: 1 epoch estimated at {len(self.token_buffer) // (B * T)} batches"
                )
            except Exception as e:
                print(f"Error loading tokenized data: {e}")
                # Fall back to streaming mode
                self._setup_streaming()
        else:
            self._setup_streaming()

            # If we have a data directory, save tokenized data for future use
            if self.data_dir:
                self._save_tokens_to_volume()

    def _setup_streaming(self):
        """Setup streaming mode from Hugging Face"""
        # Import datasets here to ensure it's available
        from datasets import load_dataset
        import tiktoken

        print(f"Rank {self.rank}: Setting up streaming from HuggingFace")
        # Load a subset of FineWeb based on rank to distribute data loading
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb", split="train", streaming=True
        )

        # Tokenize text with tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

        # Buffer to store processed tokens
        self.token_buffer = []
        self.min_buffer_size = self.B * self.T * 10  # Keep buffer 10x batch size
        self.current_position = 0

        # Pre-fill the buffer
        self._fill_buffer()

        print(f"Rank {self.rank}: initialized FineWeb dataloader")
        print(f"Rank {self.rank}: buffer contains {len(self.token_buffer)} tokens")
        print(
            f"Rank {self.rank}: 1 epoch estimated at {len(self.token_buffer) // (self.B * self.T)} batches"
        )

    def _save_tokens_to_volume(self):
        """Save tokenized data to volume for future use"""
        if not self.data_dir:
            return

        try:
            os.makedirs(self.data_dir, exist_ok=True)
            token_path = os.path.join(self.data_dir, f"tokens_shard_{self.rank}.pt")
            torch.save(self.token_buffer, token_path)
            print(f"Rank {self.rank}: Saved tokenized data to {token_path}")
        except Exception as e:
            print(f"Error saving tokenized data: {e}")

    def _fill_buffer(self):
        """Fill the token buffer with more data from FineWeb"""
        if len(self.token_buffer) > self.min_buffer_size:
            return

        # Process examples until we have enough tokens
        for example in self.dataset:
            if len(self.token_buffer) > self.min_buffer_size:
                break

            if "text" in example:
                # Tokenize and add to buffer
                tokens = self.enc.encode(example["text"])
                self.token_buffer.extend(tokens)

        # Convert to tensor for efficient processing
        self.token_buffer = torch.tensor(self.token_buffer)

    def next_batch(self):
        """Get the next batch of data"""
        B, T = self.B, self.T

        # Check if we're using synthetic data (LOCAL_TEST)
        local_debug = os.environ.get("LOCAL_TEST", "0") == "1"

        if local_debug:
            # Simple circular buffer for synthetic data
            if self.current_position + (B * T + 1) > len(self.token_buffer):
                self.current_position = 0

            buf = self.token_buffer[
                self.current_position : self.current_position + B * T + 1
            ]
        else:
            # Normal data loading with buffer refilling
            # Check if we need to refill the buffer
            if self.current_position + (B * T + 1) > len(self.token_buffer):
                # If we're using streaming mode, refill buffer
                if hasattr(self, "dataset"):
                    self._fill_buffer()
                # Otherwise, cycle through the buffer
                self.current_position = 0

            # Get current batch
            buf = self.token_buffer[
                self.current_position : self.current_position + B * T + 1
            ]

            # If we don't have enough tokens, refill and retry
            if len(buf) < B * T + 1:
                if hasattr(self, "dataset"):
                    self._fill_buffer()
                self.current_position = 0
                buf = self.token_buffer[
                    self.current_position : self.current_position + B * T + 1
                ]

        # Create inputs and targets
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Update position
        self.current_position += B * T

        return x, y


def _generate_next_token(model: nn.Module, x: torch.Tensor):
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
DEBUG_STEPS = 1  # Absolute minimum steps for quick testing


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


def setup_distributed(rank, world_size):
    """Initialize distributed training process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    print(f"Initialized process group: rank {rank} of {world_size}")


def train_worker(rank, world_size, debug=False):
    """Worker function for distributed training"""
    # Setup distributed process group
    setup_distributed(rank, world_size)

    # Set appropriate seeds for reproducibility
    torch.manual_seed(1337 + rank)
    torch.cuda.manual_seed(1337 + rank)

    # Create dataloader for this rank with volume path
    train_loader = FineWebDataLoader(
        B=4, T=1024, rank=rank, world_size=world_size, data_dir="/data/fineweb"
    )

    # Create model and move to device
    model = GPT(GPTConfig())
    model.to(rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Compile model
    model = torch.compile(model)

    # Setup optimizer
    optimizer = model.module.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=f"cuda:{rank}"
    )

    # Determine number of steps based on debug flag
    steps = DEBUG_STEPS if debug else MAX_STEPS

    if rank == 0 and debug:
        print(f"DEBUG MODE: Running only {steps} steps")

    # Training loop
    for step in range(steps):
        t0 = time.time()

        # Get batch for this rank
        x, y = train_loader.next_batch()
        x, y = x.to(rank), y.to(rank)

        # Forward and backward pass
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Adjust learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        # Synchronize for timing
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

        # Account for all GPUs' tokens in the calculation
        global_tokens_per_sec = tokens_per_sec * world_size

        if rank == 0:  # Only print from rank 0
            print(
                f"step {step} | loss: {loss.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | global tok/sec: {global_tokens_per_sec:.2f}"
            )

    # Clean up
    dist.destroy_process_group()


def run_model_distributed(num_gpus=4, debug=False):
    """Launch distributed training with the specified number of GPUs"""
    # Set appropriate PyTorch settings
    torch.set_float32_matmul_precision("high")

    # Launch processes
    mp.spawn(train_worker, args=(num_gpus, debug), nprocs=num_gpus, join=True)


@app.function(
    gpu="A100",
    memory=16384,
    cpu=4,
    image=image,
    volumes={"/data/fineweb": fineweb_volume},
)
def run_model_single(debug=False):
    """Run training on a single GPU"""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloader with volume path
    train_loader = FineWebDataLoader(B=16, T=1024, data_dir="/data/fineweb")

    # Create model
    model = GPT(GPTConfig())
    model.to(device)
    model = torch.compile(model)

    # Setup optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    # Determine number of steps based on debug flag
    steps = DEBUG_STEPS if debug else MAX_STEPS

    if debug:
        print(f"DEBUG MODE: Running only {steps} steps")

    # Training loop
    for step in range(steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        # In single GPU mode, global tokens/sec equals local tokens/sec
        global_tokens_per_sec = tokens_per_sec

        print(
            f"step {step} | loss: {loss.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f} | global tok/sec: {global_tokens_per_sec:.2f}"
        )


@app.function(gpu="H100:8", image=image, volumes={"/data/fineweb": fineweb_volume})
def run_model_multi(debug=False):
    """Multi-GPU Modal function - will launch a cluster with distributed training"""
    # Number of GPUs to use
    num_gpus = 8
    run_model_distributed(num_gpus, debug)


@app.function(
    gpu="A100",
    memory=16384,
    cpu=4,
    image=image,
    volumes={"/data/fineweb": fineweb_volume},
)
def prepare_fineweb_volume():
    """Download, tokenize and save FineWeb dataset to Modal volume"""
    print("Preparing FineWeb dataset in Modal volume at /data/fineweb")
    os.makedirs("/data/fineweb", exist_ok=True)

    # Import necessary libraries
    from datasets import load_dataset
    import tiktoken

    # Create tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load dataset
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

    # Process dataset
    print("Processing and tokenizing dataset...")
    token_buffer = []
    buffer_size_limit = 10_000_000  # Aim for ~10M tokens
    sample_count = 0

    for example in dataset:
        if len(token_buffer) > buffer_size_limit:
            break

        if "text" in example:
            # Tokenize and add to buffer
            tokens = enc.encode(example["text"])
            token_buffer.extend(tokens)
            sample_count += 1

            if sample_count % 1000 == 0:
                print(
                    f"Processed {sample_count} samples, collected {len(token_buffer)} tokens"
                )

    # Convert to tensor
    token_buffer = torch.tensor(token_buffer)

    # Save to volume
    print(f"Saving {len(token_buffer)} tokens to volume...")
    torch.save(token_buffer, "/data/fineweb/tokens.pt")

    # Create additional sharded versions for distributed training
    shard_count = 8  # Create 8 shards for distributed training
    shard_size = len(token_buffer) // shard_count

    for i in range(shard_count):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < shard_count - 1 else len(token_buffer)
        shard = token_buffer[start_idx:end_idx]
        torch.save(shard, f"/data/fineweb/tokens_shard_{i}.pt")
        print(f"Created shard {i} with {len(shard)} tokens")

    print("Dataset preparation complete!")


@app.local_entrypoint()
def trigger_on_modal():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train GPT-2 on Modal with FineWeb dataset"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer steps"
    )
    parser.add_argument(
        "--prepare_volume",
        action="store_true",
        help="Only download and tokenize FineWeb data to volume without training",
    )
    args = parser.parse_args()

    if args.debug:
        print("Running in DEBUG mode with reduced steps to save credits")

    if args.prepare_volume:
        print("Preparing FineWeb dataset in volume...")
        prepare_fineweb_volume.remote()
    else:
        # Run with debug flag if specified
        run_model_multi.remote(debug=args.debug)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train GPT-2 locally with FineWeb dataset"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer steps"
    )
    parser.add_argument(
        "--prepare_volume",
        action="store_true",
        help="Only download and tokenize FineWeb data to volume without training",
    )
    args = parser.parse_args()

    if args.prepare_volume:
        print("Preparing FineWeb dataset in volume locally...")
        prepare_fineweb_volume.local()
    else:
        # Default to single GPU locally with debug flag if specified
        run_model_single.local(debug=args.debug)
