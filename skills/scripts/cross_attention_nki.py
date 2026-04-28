import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


def get_tile_size(
    context_len: int,
    action_seq: int,
    head_dim: int,
    batch_heads: int,
    dtype_bytes: int = 2,
) -> int:
    SBUF_BYTES = 24 * 1024 * 1024
    q_bytes = batch_heads * action_seq * head_dim * dtype_bytes
    remaining = (SBUF_BYTES - q_bytes) // 3
    tile_elements = remaining // (batch_heads * head_dim * dtype_bytes)
    tile = 1
    while tile * 2 <= tile_elements:
        tile *= 2
    return min(tile, context_len)


@nki.jit
def cross_attention_kernel[tile_size](q, k, v, scale, outputdest):
    batch_heads, action_seq, head_dim = q.shape
    _, context_len, _ = k.shape

    i_bh = nl.arange(batch_heads)[:, None, None]
    i_q  = nl.arange(action_seq)[None, :, None]
    i_t  = nl.arange(tile_size)[None, :, None]
    i_d  = nl.arange(head_dim)[None, None, :]

    q_imp = nl.load(q[i_bh, i_q, i_d])

    scores = nl.zeros((batch_heads, action_seq, context_len), dtype=nl.float32)
    for chunk in nl.affine_range(0, context_len, tile_size):
        i_kv = chunk + i_t
        k_chunk = nl.load(k[i_bh, i_kv, i_d])
        scores[i_bh, i_q, i_kv] = nl.matmul(q_imp, k_chunk, transpose_y=True) * scale

    weights = nl.softmax(scores, axis=2)

    output = nl.zeros((batch_heads, action_seq, head_dim), dtype=nl.float32)
    for chunk in nl.affine_range(0, context_len, tile_size):
        i_kv = chunk + i_t
        v_chunk = nl.load(v[i_bh, i_kv, i_d])
        output += nl.matmul(weights[i_bh, i_q, i_kv], v_chunk)

    nl.store(outputdest[i_bh, i_q, i_d], output)
