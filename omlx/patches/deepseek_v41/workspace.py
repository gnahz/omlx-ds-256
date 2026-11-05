"""Pure CPU geometry shared by index dispatch and prefill memory admission."""

def index_radix_query_group(key_tokens):
    """The actual score group planner, also used by memory admission."""
    return min(1024, max(1, (64 << 20) // (4 * max(1, key_tokens))))


def index_scan_plan(length, width, *, block_count=0, block_size=0,
                    query_chunk_size=None, chunk_size=None,
                    mid_index=True, decode_single_tile=True):
    """Plan unchanged source scans without allocating tensors or GPU work."""
    if (mid_index and block_count and query_chunk_size is None
            and chunk_size is None and 8 < length <= 128):
        query_chunk_size, chunk_size = length, 262144
    if query_chunk_size is None:
        query_chunk_size = min(512, max(16, 1048576 // max(1, width)))
    if length <= 0 or width < 0 or block_count < 0 or query_chunk_size <= 0:
        raise ValueError("Invalid index selection budget")
    if chunk_size is None:
        chunk_size = min(524288 if length == 1 else 262144,
                         1048576 // max(1, min(length, query_chunk_size)))
        if decode_single_tile and length <= 8:
            chunk_size = max(chunk_size, (width + 63) // 64 * 64)
    if chunk_size <= 0 or (block_count and block_size <= 0):
        raise ValueError("Invalid index selection budget")
    if block_count:
        chunk_size = max(block_size, chunk_size // block_size * block_size)
    return query_chunk_size, chunk_size


def packed_index_workspace_bytes(query_tokens, key_tokens, count, *,
                                 block_count=0, block_size=0):
    """Conservative live score/selection/ID buffers from the dispatch planners."""
    if query_tokens <= 0 or key_tokens <= 0:
        return 0
    if count < 0 or block_count < 0 or (block_count and block_size <= 0):
        raise ValueError("Invalid index selection budget")
    selected = min(count, key_tokens)
    radix_bytes = 0
    if (query_tokens > 8 and selected and block_count <= 2048
            and key_tokens >= max(4096, selected, block_count * block_size)):
        group = min(query_tokens, index_radix_query_group(key_tokens))
        # One group executes while its successor is constructed.
        scores = 2 * group * key_tokens * 4
        selection = 6 * query_tokens * (selected + block_count) * 4
        # Candidate-source radix scans also select per-block maxima.
        if block_count:
            keys_per_block = (key_tokens + block_size - 1) // block_size
            selection += 3 * group * keys_per_block * 8
        radix_bytes = scores + selection
        if not block_count:
            return radix_bytes
    queries, keys = index_scan_plan(query_tokens, key_tokens,
                                   block_count=block_count, block_size=block_size)
    queries, keys = min(queries, query_tokens), min(keys, key_tokens)
    scores = 2 * queries * keys * 4
    tile_limit = max(1024, 2 * selected)
    tile = max(32, 1 << (max(1, min(keys, tile_limit)) - 1).bit_length())
    tiles = (keys + tile - 1) // tile
    # Initial FP32/int32 winner pairs, merge output and a second pending scan.
    winners = 3 * queries * tiles * selected * 8
    block_winners = 0
    if block_count:
        block_keys = (keys + block_size - 1) // block_size
        selected_blocks = min(block_count, block_keys)
        block_tile_limit = max(1024, 2 * selected_blocks)
        block_tile = max(32, 1 << (max(1, min(block_keys, block_tile_limit)) - 1).bit_length())
        block_tiles = (block_keys + block_tile - 1) // block_tile
        # Source-token top-k and candidate-block maxima have independent
        # FP32/int32 winner and merge buffers inside the same score scan.
        block_winners = 3 * queries * block_tiles * selected_blocks * 8
    ids = 6 * query_tokens * (selected + block_count) * 4
    # Keep the custom scan allowance as a floor while the upstream radix
    # candidate route remains shape/device dependent. Admission must cover both.
    return max(radix_bytes, scores + winners + block_winners + ids)
