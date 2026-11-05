// SPDX-License-Identifier: MIT
// Expert-sorted NAX geometry follows Apple's MLX 0.32.2 Metal quantized.cpp.
#include "deepseek_v41_joint_metadata.h"
#include "omlx/binary_dir.h"

#include <algorithm>
#include <limits>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"

namespace omlx::quant_matmul {
namespace {
using namespace mlx::core;

// Mirror MLX's NAX hardware/OS admission using exported Device accessors.
// is_nax_available itself is private in the installed MLX shared library.
bool nax_supported(const Stream& stream) {
  if (__builtin_available(macOS 26.2, *)) {
    auto& device = metal::device(stream.device);
    const auto& architecture = device.get_architecture();
    return !architecture.empty() &&
        device.get_architecture_gen() >= (architecture.back() == 'p' ? 18 : 17);
  }
  return false;
}

class JointMetadataGatherQMMPrimitive : public Primitive {
 public:
  JointMetadataGatherQMMPrimitive(Stream stream, bool gather_x)
      : Primitive(stream), gather_x_(gather_x) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DeepSeek joint-metadata NAX QMM has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs,
                std::vector<array>& outputs) override {
    const auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& words = inputs[2];
    const auto& dictionary = inputs[3];
    const auto& indices = inputs[4];
    // Empty optional entries are never handed to this primitive; the last
    // input is either the true source-row mapping or the already-normalized
    // expert indices (unused in the contiguous routed-down specialization).
    const auto& rows = gather_x_ ? inputs[5] : indices;
    if (!weight.flags().row_contiguous || !words.flags().row_contiguous ||
        !dictionary.flags().row_contiguous) {
      throw std::invalid_argument("Resident joint-metadata banks must remain row-contiguous.");
    }
    if (!x.flags().row_contiguous || !indices.flags().row_contiguous ||
        !rows.flags().row_contiguous) {
      throw std::runtime_error("Joint-metadata activation/routing normalization failed.");
    }
    auto& output = outputs[0];
    const int M = static_cast<int>(indices.size());
    const int N = weight.shape(1);
    const int K = x.shape(2);
    const int E = weight.shape(0);
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int WM = 4;
    constexpr int WN = 1;
    const int max_tiles = (M + BM - 1) / BM + std::min(E, M);
    const uint32_t dictionary_size = static_cast<uint32_t>(dictionary.size());
    const uint32_t source_rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t expert_count = static_cast<uint32_t>(E);

    // Resolve both pipelines before the first dispatch. If a JIT/pipeline
    // lookup throws, no tile kernel has been enqueued with inputs awaiting
    // gpu::eval's normal completion-owner callback.
    auto library = device.get_library(
        "omlx_quant_matmul_kernels_nax", omlx::binary_dir());
    auto tile_kernel = device.get_kernel("ds41_joint_metadata_rhs_tiles_64", library);
    const bool align_M = false;
    const bool align_N = true;
    const bool align_K = true;
    const bool use_tiles = true;
    const bool gather_x = gather_x_;
    metal::MTLFCList constants = {
        {&align_M, MTL::DataType::DataTypeBool, 200},
        {&align_N, MTL::DataType::DataTypeBool, 201},
        {&align_K, MTL::DataType::DataTypeBool, 202},
        {&use_tiles, MTL::DataType::DataTypeBool, 203},
        {&gather_x, MTL::DataType::DataTypeBool, 204},
    };
    const std::string function = "ds41_joint_metadata_gather_qmm_nax_bf16";
    auto kernel = device.get_kernel(
        function, library, function + (gather_x ? "_source_rows" : "_routed_rows"), constants);
    // Register the tile descriptor owner before any encoding. Metal's
    // command buffers use unretained references; it must survive until the
    // encoder fence even if subsequent allocation/encoding throws.
    array tiles({max_tiles, 2}, int32, nullptr, {});
    tiles.set_data(allocator::malloc(tiles.nbytes()));
    auto& encoder = metal::get_command_encoder(stream);
    encoder.add_temporary(tiles);
    output.set_data(allocator::malloc(output.nbytes()));

    encoder.set_compute_pipeline_state(tile_kernel);
    encoder.set_input_array(indices, 0);
    encoder.set_output_array(tiles, 1);
    encoder.set_bytes(M, 2);
    encoder.set_bytes(max_tiles, 3);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(words, 2);
    encoder.set_input_array(dictionary, 3);
    encoder.set_input_array(indices, 4);
    encoder.set_output_array(output, 5);
    encoder.set_bytes(M, 6);
    encoder.set_bytes(N, 7);
    encoder.set_bytes(K, 8);
    encoder.set_input_array(tiles, 9);
    if (gather_x) {
      encoder.set_input_array(rows, 10);
    }
    encoder.set_bytes(dictionary_size, 11);
    encoder.set_bytes(source_rows, 12);
    encoder.set_bytes(expert_count, 13);
    encoder.dispatch_threadgroups(
        MTL::Size(N / BN, max_tiles, 1), MTL::Size(32, WN, WM));
  }

  DEFINE_NAME(OMLXDeepSeekV41JointMetadataGatherQMM)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    return gather_x_ == static_cast<const JointMetadataGatherQMMPrimitive&>(other).gather_x_;
  }
  auto state() const { return std::make_tuple(nullptr, gather_x_); }

 private:
  bool gather_x_;
};
} // namespace

mlx::core::array deepseek_v41_joint_metadata_gather_qmm(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& joint_words,
    const mlx::core::array& joint_dictionary,
    const std::optional<mlx::core::array>& lhs_indices,
    const mlx::core::array& rhs_indices,
    mlx::core::StreamOrDevice stream_or_device) {
  using namespace mlx::core;
  const auto stream = to_stream(stream_or_device);
  if (stream.device == Device::cpu || !metal::is_available() || !nax_supported(stream)) {
    throw std::invalid_argument("DeepSeek joint-metadata QMM requires Metal NAX.");
  }
  if (x.ndim() != 3 || x.shape(0) <= 0 || x.shape(1) != 1 ||
      x.shape(2) <= 0 || x.dtype() != bfloat16 || weight.ndim() != 3 ||
      weight.dtype() != uint32 || !weight.flags().row_contiguous ||
      weight.shape(0) <= 0 || weight.shape(0) > 384 || weight.shape(1) <= 0 ||
      weight.shape(2) <= 0 || joint_words.ndim() != 1 ||
      joint_words.dtype() != uint32 || !joint_words.flags().row_contiguous ||
      joint_dictionary.ndim() != 1 || joint_dictionary.dtype() != uint32 ||
      !joint_dictionary.flags().row_contiguous || joint_dictionary.size() < 1 ||
      joint_dictionary.size() > 65536 || rhs_indices.ndim() != 1 ||
      rhs_indices.dtype() != uint32 || rhs_indices.size() < 16 ||
      rhs_indices.size() > std::numeric_limits<int>::max() - 1024) {
    throw std::invalid_argument("Unsupported DeepSeek joint-metadata QMM layout.");
  }
  const uint64_t K = static_cast<uint64_t>(x.shape(2));
  const uint64_t N = static_cast<uint64_t>(weight.shape(1));
  const uint64_t E = static_cast<uint64_t>(weight.shape(0));
  const uint64_t routes = rhs_indices.size();
  // E is capped384 and MLX shape extents are signed32-bit. This product is
  // therefore bounded below UINT64_MAX; arithmetic must never narrow to int.
  const uint64_t metadata_count = E * N * (K / 128);
  const uint64_t packed_bank_bytes = E * N * K * 3 / 8;
  // V1 keeps original signed-int tile strides with an explicit safe bound:
  // max N*K*3/8 = 1,610,612,736, below INT_MAX. Current checkpoint dimensions
  // 5120/2304 are well inside it. Do not silently admit unqualified huge banks.
  // The retained shader also uses 32-bit expert*stride offsets. Bound the
  // whole bank, not only each expert: current banks are 1,698,693,120 bytes.
  if (packed_bank_bytes > std::numeric_limits<int>::max() ||
      K > 65536 || N > 65536 || K % 128 != 0 || N % 64 != 0 || routes / E < 4 ||
      static_cast<uint64_t>(weight.shape(2)) != K * 3 / 32 ||
      metadata_count % 2 != 0 || joint_words.size() != metadata_count / 2 ||
      (lhs_indices && (lhs_indices->ndim() != 1 || lhs_indices->dtype() != uint32 ||
                       lhs_indices->size() != routes)) ||
      (!lhs_indices && static_cast<uint64_t>(x.shape(0)) != routes)) {
    throw std::invalid_argument("Invalid DeepSeek joint-metadata QMM geometry.");
  }
  // Always create the public Contiguous graph node. Lazy input flags may
  // change during evaluation, so a factory-time flag shortcut is unsafe.
  // Its evaluator shares an already-contiguous input without a device copy;
  // strided activations/routing get a bounded copy with normal fence ownership.
  // Resident weights and metadata are never normalized or copied here.
  auto normalize = [&](const array& value) {
    return contiguous(value, false, stream);
  };
  std::vector<array> inputs{normalize(x), weight, joint_words, joint_dictionary, normalize(rhs_indices)};
  if (lhs_indices) inputs.push_back(normalize(*lhs_indices));
  return array({static_cast<int>(routes), 1, static_cast<int>(N)}, bfloat16,
               std::make_shared<JointMetadataGatherQMMPrimitive>(stream, lhs_indices.has_value()),
               std::move(inputs));
}
} // namespace omlx::quant_matmul
