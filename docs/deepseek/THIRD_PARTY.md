# Third-party notices

This fork preserves the upstream oMLX project and its [Apache-2.0 license](../../LICENSE).

The DeepSeek V4.1 implementation retains the [DeepSeek MIT license](../../omlx/patches/deepseek_v41/LICENSE), including its copyright notice. Per-file SPDX notices remain in place.

MLX source and MLX-derived native kernel code are distributed under the Apple MIT license. The notices are included in [attention/csrc/MLX_LICENSE.txt](../../omlx/custom_kernels/attention/csrc/MLX_LICENSE.txt) and [quant_matmul/csrc/MLX_LICENSE.txt](../../omlx/custom_kernels/quant_matmul/csrc/MLX_LICENSE.txt). The embedded JIT source in `single_decode_jit_source.h` retains its Apple copyright notices and source references. The separately applied [MLX patch](../../deploy/deepseek/patches/mlx-custom.patch) includes the full MIT notice and identifies its exact upstream commit in the dependency lock.

Native attention also retains the [mlx-serve license notice](../../omlx/custom_kernels/attention/csrc/MLX_SERVE_LICENSE.txt). No existing attribution has been removed.

Model files are not included in this Git repository. Downloading the separately published [1017 model](https://huggingface.co/gnahzhang/DeepSeek-V4.1-Flash-1017-LSQ3-G128-MLX) is subject to the model repository's terms. Python and build dependencies retain their own licenses.
