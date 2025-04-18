import torch
import triton
import triton.language as tl

import modal

app = modal.App("flash-attention")
image = modal.Image.debian_slim("3.12.9").pip_install_from_requirements(
    "./requirements.txt"
)


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_KV,
    STAGE,
    offs_q,
    offs_kv,
    SEQ_LEN,
):
    if STAGE == 1:
        # all elements left of diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # elements in the diagonal (needs masking)
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # all elements (non-causal)
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # compute KV
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        P_block = tl.math.exp(QK_block)

        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return (
        O_block,
        l_i,
    )


@app.function(gpu="A100", image=image)
def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    @triton.jit
    def _attn_fwd(
        Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        K,
        softmax_scale,
        V,
        M,
        O: tl.tensor,
        stride_batch,
        stride_head,
        stride_seq,
        stride_dim,
        BATCH_SIZE,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        STAGE: tl.constexpr,
        BLOCK_SIZE_Q=4,  # these will be passed in during tuning
        BLOCK_SIZE_KV=4,
    ):
        block_index_q = tl.program_id(0)

        index_batch_head = tl.program_id(1)

        index_batch = index_batch_head // NUM_HEADS
        index_head = index_batch_head % NUM_HEADS

        qvk_offset = (
            index_batch.to(tl.int64) * stride_batch
            + index_head.to(tl.int64) * stride_head
        )

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,  # Q[index_batch, index_head, :, :]
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_seq, stride_dim),
            offsets=(block_index_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_dim, stride_seq),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_seq, stride_dim),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            base=O + qvk_offset,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_seq, stride_dim),
            offsets=(block_index_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
            order=(1, 0),
        )

        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)

        # Stores the running max. One per query
        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float(
            "inf"
        )  # pyright:ignore[reportUnreachable]

        # Stores the running norm. One per query
        # Adding 1 for more stability when computing natural log
        l_i = (
            tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
        )  # pyright:ignore[reportUnreachable]

        # Stores the accumulated output
        O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

        Q_block = tl.load(Q_block_ptr)

        # 3 if causal else 1
        if STAGE == 1 or STAGE == 3:
            # This step runs for non-causal attention
            # Runs for all elems left of diagonal
            O_block, l_i, m_i = _attn_fwd_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                4 - STAGE,
                offs_q,
                offs_kv,
                SEQ_LEN,
            )

        if STAGE == 3:
            # This step runs for both
            # Compute all elements right of diagonal
            O_block, l_i, m_i = _attn_fwd_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                2,
                offs_q,
                offs_kv,
                SEQ_LEN,
            )

        m_i += tl.math.log(l_i)

        O_block = O_block / l_i[:, None]
        m_ptrs = M + index_batch_head * SEQ_LEN + offs_q

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, O_block.to(O.type.element_ty))

    class TritonAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, causal, softmax_scale) -> torch.Tensor:
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

            O = torch.empty_like(Q)

            grid = lambda args: (
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * NUM_HEADS,
                1,
            )

            stage = 3 if causal else 1

            # this will store the logsumexp for the backward pass
            M = torch.empty(
                (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
            )

            _attn_fwd[grid](
                Q=Q,
                K=K,
                V=V,
                softmax_scale=softmax_scale,
                M=M,
                O=O,
                stride_batch=Q.stride(0),
                stride_head=Q.stride(1),
                stride_seq=Q.stride(2),
                stride_dim=Q.stride(3),
                BATCH_SIZE=BATCH_SIZE,
                NUM_HEADS=NUM_HEADS,
                SEQ_LEN=SEQ_LEN,
                HEAD_DIM=HEAD_DIM,
                STAGE=stage,
            )

            ctx.save_for_backward(Q, K, V, O, M)
            ctx.grid = grid
            ctx.softmax_scale = softmax_scale
            ctx.HEAD_DIM = HEAD_DIM
            ctx.causal = causal

            return O

    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)
    d0 = torch.randn_like(Q)

    # lower triangular mask
    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device="cuda"))
    # P = Q * K^T
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        #
        P[:, :, MASK == 0] = float("-inf")

    P = torch.softmax(P.float(), dim=-1).half()
    ref_0 = torch.matmul(P, V)

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()

    print("ref_0", ref_0[0, :, 10, :])
    print("tri_out", tri_out[0, :, 10, :])
    print(torch.allclose(tri_out, ref_0, atol=0.01, rtol=0.0))


@app.local_entrypoint()
def trigger_on_modal():
    test_op.remote(256, 12, 1024, 64, 3)


if __name__ == "__main__":
    test_op.local(256, 12, 1024, 64, 3)
