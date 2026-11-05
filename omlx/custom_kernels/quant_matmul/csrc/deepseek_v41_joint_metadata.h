#pragma once
#include <optional>
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"
namespace omlx::quant_matmul {
// Internal expert-sorted LSQ3/G128 BF16 prefill. rhs_indices must be sorted
// valid expert IDs; lhs_indices, when provided, map each route to a valid x
// source row. With lhs omitted x contains one row per sorted route.
// Shape/dtype/layout admission is synchronous, index data remain on device.
mlx::core::array deepseek_v41_joint_metadata_gather_qmm(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& joint_words,
    const mlx::core::array& joint_dictionary,
    const std::optional<mlx::core::array>& lhs_indices,
    const mlx::core::array& rhs_indices,
    mlx::core::StreamOrDevice stream = {});
}
