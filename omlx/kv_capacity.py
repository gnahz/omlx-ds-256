"""Allocation geometry shared by the cache writer and CPU memory planner."""

def growth_capacity(end, rows):
    if rows > 64:
        return (max(end, 256) + 255) // 256 * 256
    return (end + max(4096, end // 16) + 255) // 256 * 256
