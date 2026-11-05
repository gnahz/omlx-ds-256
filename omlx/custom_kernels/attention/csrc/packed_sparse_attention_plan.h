#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace omlx::attention {

struct PackedSparseAttentionWorkspacePlan {
  bool unpack_kv;
  size_t unique_kv_bytes;
  size_t scratch_bytes;
};

// This CPU-only geometry planner is shared by allocation and admission.
// Q/output storage is charged separately by the model's memory profile.
inline PackedSparseAttentionWorkspacePlan packed_sparse_attention_workspace(
    int64_t batch, int64_t queries, int64_t local_rows, int64_t pooled_rows) {
  if (batch <= 0 || queries <= 0 || local_rows < queries || pooled_rows < 0) {
    throw std::invalid_argument("Invalid packed attention workspace geometry");
  }
  constexpr size_t row_bytes = 512 * sizeof(uint16_t);
  constexpr size_t limit = 80 * 1024 * 1024;
  const auto local = static_cast<size_t>(local_rows);
  const auto pooled = static_cast<size_t>(std::max<int64_t>(pooled_rows, 1));
  const auto batches = static_cast<size_t>(batch);
  if (local > std::numeric_limits<size_t>::max() - pooled ||
      batches > std::numeric_limits<size_t>::max() / row_bytes /
          (local + pooled)) {
    throw std::overflow_error("Packed attention workspace exceeds address space");
  }
  const size_t bytes = batches * (local + pooled) * row_bytes;
  const bool unpack = queries >= 256 && bytes <= limit;
  return {unpack, bytes, unpack ? bytes : size_t(0)};
}

} // namespace omlx::attention
