import torch
import triton
import triton.language as tl

import modal

app = modal.App("softmax-analysis")
image = modal.Image.debian_slim("3.11.9").pip_install_from_requirements(
    "./requirements.txt"
)


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1, keepdim=True)[0]
    safe_x = x - x_max
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1, keepdim=True)[0]
    softmax_out = numerator / denominator
    return softmax_out


@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    output_row_stride,
    input_ptr,
    input_row_stride,
    cols,
    block_size: tl.constexpr,
):
    row_ix = tl.program_id(0)

    row_start_ix = input_ptr + row_ix * input_row_stride
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ix + col_offsets
    mask = col_offsets < cols

    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    row_max = tl.max(row, axis=0)
    safe_row = row - row_max

    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)

    sm_output = numerator / denominator

    output_row_ptr = output_ptr + row_ix * output_row_stride
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, sm_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}"

    block_size = triton.next_power_of_2(cols)

    num_warps = 4
    if block_size == 2048:
        num_warps = 8
    elif block_size == 4096:
        num_warps = 16

    grid = (rows,)

    sm_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps,
    )

    return sm_out


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        safe_max = row - tl.max(row, axis=0)
        numerator = tl.exp(safe_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


@triton.testing.perf_report(
    benchmarks=triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


@app.function(gpu="A100", image=image)
def trigger():
    benchmark.run(print_data=True)
