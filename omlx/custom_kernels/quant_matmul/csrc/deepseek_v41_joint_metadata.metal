// SPDX-License-Identifier: MIT
// Dictionary-addressed variant of MLX's affine3/G128 expert-sorted NAX GEMM.
// Copyright (c) 2023-2026 Apple Inc. Original MLX arithmetic retained.
// Metadata dictionaries store original BF16 scale/bias bit patterns; no requantization.
// K and N alignment are admission requirements; inputs must contain valid sorted experts.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mlx::steel;
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
constant bool use_tiles [[function_constant(203)]];
constant bool gather_x [[function_constant(204)]];
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;
template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename U, int N, int bits>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  const float s = float(scale);
  const float b = float(bias);

  if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = static_cast<U>(sc[0] * (w[i] & 0x03) + b);
      w_local[4 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x0c) + b);
      w_local[4 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x30) + b);
      w_local[4 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0xc0) + b);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = static_cast<U>((w[0] & 0x7) * s + b);
      w_local[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
      w_local[2] =
          static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
      w_local[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
      w_local[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
      w_local[5] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
      w_local[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
      w_local[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
    }
  }

  else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct JointMetadataBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device ushort* codes;
  const device uint* dictionary;
  const uint dictionary_size;

  JointMetadataBlockLoader(
      const device uint8_t* src_,
      const device ushort* codes_,
      const device uint* dictionary_,
      const uint dictionary_size_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        codes(codes_ + bi * src_ld / group_size),
        dictionary(dictionary_),
        dictionary_size(dictionary_size_) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    const uint code = *codes;
    const uint pair = code < dictionary_size ? dictionary[code] : 0;
    T scale = as_type<T>(ushort(pair & 65535));
    T bias = as_type<T>(ushort(pair >> 16));
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    const uint code = *codes;
    const uint pair = code < dictionary_size ? dictionary[code] : 0;
    T scale = as_type<T>(ushort(pair & 65535));
    T bias = as_type<T>(ushort(pair >> 16));
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          codes++;
        }
      } else {
        codes++;
      }
    } else {
      codes += group_stride;
    }
  }
};

template <
    typename T,
    int group_size,
    const int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void ds41_joint_metadata_gather_qmm_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint32_t* joint_words,
    const device uint32_t* dictionary,
    const device uint32_t* indices,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    const device int2* tiles [[buffer(9), function_constant(use_tiles)]],
    const device uint32_t* x_rows [[buffer(10), function_constant(gather_x)]],
    const constant uint& dictionary_size [[buffer(11)]],
    const constant uint& source_rows [[buffer(12)]],
    const constant uint& expert_count [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  // With tiles, each threadgroup owns rows [x, y) of one expert.
  int2 tile = use_tiles ? tiles[tid.y] : int2(tid.y * BM, M);
  if (use_tiles && tile.x >= tile.y) {
    return;
  }
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = JointMetadataBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tile.x;
  const int m_end = tile.y;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, m_end - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block. With gather_x, row r of the
  // product reads row x_rows[r] of x.
  auto wl = (const device uint8_t*)w;
  if (!gather_x) {
    x += y_row_long * K;
  }
  y += y_row_long * N + y_col_long;
  // Internal callers supply valid expert-sorted routing. Guard expert bounds
  // as a defensive device-side fallback without synchronizing routing to CPU.
  if (indices[y_row] >= expert_count) {
    const uint linear_tid = simd_group_id * 32 + simd_lane_id;
    for (uint i = linear_tid; i < uint(tgp_bm) * uint(tgp_bn); i += WM * WN * 32) {
      y[(i / uint(tgp_bn)) * N + i % uint(tgp_bn)] = T(0);
    }
    return;
  }
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  auto codes = reinterpret_cast<const device ushort*>(joint_words);
  codes += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm =
      align_M ? SM : min(int(SM), max(0, (m_end - (y_row + tm))));
  const short sgp_sn = align_N ? SN : min(int(SN), max(0, (N - (y_col + tn))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    // With gather_x, the two rows of each A fragment that this thread loads.
    // Rows past the tile repeat its last row; their results are not stored.
    const device T* xr[TM][2];
    bool xr_valid[TM][2];
    if (gather_x) {
      const short2 sc = BaseNAXFrag::get_coord();
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; r++) {
          const int row = min(y_row + tm + i * 16 + sc.y + r * 8, m_end - 1);
          const uint source = x_rows[row];
          xr_valid[i][r] = source < source_rows;
          xr[i][r] = x + size_t(xr_valid[i][r] ? source : 0) * K + sc.x;
        }
      }
    }

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        codes + index * stride_s,
        dictionary,
        dictionary_size,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if (gather_x) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < TM; i++) {
                  STEEL_PRAGMA_UNROLL
                  for (short tk = 0; tk < TK; tk++) {
                    thread auto& fr = Atile.frag_at(i, tk);
                    STEEL_PRAGMA_UNROLL
                    for (short r = 0; r < 2; r++) {
                      STEEL_PRAGMA_UNROLL
                      for (short j = 0; j < 4; j++) {
                        fr[r * 4 + j] = xr_valid[i][r] ? xr[i][r][kk1 + tk * 16 + j] : T(0);
                      }
                    }
                  }
                }
              } else if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }

          xn += BK;
          if (gather_x) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM; i++) {
              xr[i][0] += BK;
              xr[i][1] += BK;
            }
          }
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (k_remain - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(
                    Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(
                    Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}

template <int BM>
[[kernel]] void ds41_joint_metadata_rhs_tiles(
    const device uint32_t* indices [[buffer(0)]],
    device int2* tiles [[buffer(1)]],
    const constant int& M [[buffer(2)]],
    const constant int& max_tiles [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup int sg_sums[32];
  threadgroup int total;

  const int per = (M + int(tsize) - 1) / int(tsize);
  const int r0 = min(M, int(tid) * per);
  const int r1 = min(M, r0 + per);

  auto segment_end = [&](int i) {
    int j = i + 1;
    while (j < M && indices[j] == indices[i]) {
      j++;
    }
    return j;
  };

  int count = 0;
  for (int i = r0; i < r1; i++) {
    if (i == 0 || indices[i] != indices[i - 1]) {
      count += (segment_end(i) - i + BM - 1) / BM;
    }
  }

  // Exclusive prefix sum of the counts over the threadgroup
  if (simd_group_id == 0) {
    sg_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int incl = simd_prefix_inclusive_sum(count);
  if (simd_lane_id == 31) {
    sg_sums[simd_group_id] = incl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    int v = sg_sums[simd_lane_id];
    int v_incl = simd_prefix_inclusive_sum(v);
    sg_sums[simd_lane_id] = v_incl - v;
    if (simd_lane_id == 31) {
      total = v_incl;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int offset = sg_sums[simd_group_id] + incl - count;
  for (int i = r0; i < r1; i++) {
    if (i == 0 || indices[i] != indices[i - 1]) {
      int end = segment_end(i);
      for (int s = i; s < end; s += BM) {
        // Bound writes even if a caller violates the sorted-valid-ID contract.
        if (offset < max_tiles) { tiles[offset] = int2(s, min(s + BM, end)); }
        offset++;
      }
    }
  }
  for (int t = total + int(tid); t < max_tiles; t += int(tsize)) {
    tiles[t] = int2(0, 0);
  }
}
instantiate_kernel("ds41_joint_metadata_rhs_tiles_64", ds41_joint_metadata_rhs_tiles, 64)
instantiate_kernel("ds41_joint_metadata_gather_qmm_nax_bf16", ds41_joint_metadata_gather_qmm_nax,
                   bfloat16_t, 128, 3, 64, 64, 128, 4, 1, true)
