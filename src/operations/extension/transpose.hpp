/***************************************************************************
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename transpose.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TRANSPOSE_HPP
#define SYCL_BLAS_EXTENSION_TRANSPOSE_HPP

#include "operations/extension/transpose.h"

namespace blas {

// Transpose
template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE bool
Transpose<in_place, Tile_size, local_memory, in_t, out_t,
          element_t>::valid_thread(cl::sycl::nd_item<1> item) const {
  // Valid threads are established by ::eval()
  return true;
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::bind(cl::sycl::handler &cgh) {
  A_.bind(cgh);
  At_.bind(cgh);
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE typename in_t::index_t
Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>::get_size()
    const {
  return (M_pad_ * N_pad_);
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::adjust_access_displacement() {
  A_.adjust_access_displacement();
  At_.adjust_access_displacement();
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::eval(cl::sycl::nd_item<1> id) {
  auto idx = id.get_global_linear_id();

  auto j = idx / M_pad_;
  auto i = idx - j * M_pad_;

  if (i < M_ && j < N_) {
    auto A = A_.get_data().get_pointer();
    auto At = At_.get_data().get_pointer();

    auto in_index = i + j * lda_;
    auto out_index = i * ldat_ + j;

    At[out_index] = alpha_ * A[in_index];
  }
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
template <typename index_t>
SYCL_BLAS_INLINE void
Transpose<in_place, Tile_size, local_memory, in_t, out_t,
          element_t>::compute_trans_indices(cl::sycl::nd_item<1> id,
                                            index_t &in_idx, index_t &in_idc,
                                            index_t &out_idx, index_t &out_idc,
                                            bool &valid_index_in,
                                            bool &valid_index_out) {
  index_t idg = id.get_group(0);
  index_t idc = id.get_local_id();

  const index_t jg = idg / tile_count_m_;
  const index_t ig = idg - jg * tile_count_m_;

  const index_t jl = idc / Tile_size;
  const index_t il = idc - jl * Tile_size;

  const index_t i_block_start = ig * Tile_size;
  const index_t j_block_start = jg * Tile_size;

  valid_index_in = (i_block_start + il < M_ && j_block_start + jl < N_);
  valid_index_out = (i_block_start + jl < M_ && j_block_start + il < N_);

  in_idx = i_block_start * stridea_ + j_block_start * lda_ + il * stridea_ +
           jl * lda_;
  in_idc = jl * (Tile_size + 1) + il;

  out_idx = i_block_start * ldat_ + j_block_start * strideat_ + il * strideat_ +
            jl * ldat_;
  out_idc = il * (Tile_size + 1) + jl;
}

template <bool both_trans, int Tile_size, typename index_t>
SYCL_BLAS_INLINE void compute_trans_add_indices(
    cl::sycl::nd_item<1> id, const index_t &row_tiles, const index_t &lda,
    const index_t &ldb, const index_t &ldc, index_t &in_a_idx,
    index_t &in_b_idx, index_t &in_local_idx, index_t &out_idx,
    index_t &out_local_idx) {
  index_t idx = id.get_global_linear_id();
  index_t idg = id.get_group(0);
  index_t idc = id.get_local_id();

  const index_t jg = idg / row_tiles;
  const index_t ig = idg - jg * row_tiles;

  const index_t jl = idc / Tile_size;
  const index_t il = idc - jl * Tile_size;

  if constexpr (both_trans) {
    in_a_idx = ig * Tile_size + jg * Tile_size * lda + il + jl * lda;
    out_idx = ig * Tile_size * ldc + jg * Tile_size + il + jl * ldc;

  } else {
    in_a_idx = jg * Tile_size + ig * Tile_size * lda + il + jl * lda;
    out_idx = ig * Tile_size + jg * Tile_size * ldc + il + jl * ldc;
  }

  in_b_idx = ig * Tile_size + jg * Tile_size * ldb + il + jl * ldb;

  in_local_idx = jl * (Tile_size + 1) + il;

  out_local_idx = il * (Tile_size + 1) + jl;
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::eval(local_memory_t local_mem,
                                                 cl::sycl::nd_item<1> id) {
  index_t idx = id.get_global_linear_id();

  if (idx < get_size()) {
    value_t *local = local_mem.localAcc.get_pointer();

    auto A = A_.get_data().get_pointer();
    auto At = At_.get_data().get_pointer();

    index_t in_index, in_local_id, out_index, out_local_id;
    bool valid_index_in, valid_index_out;

    compute_trans_indices(id, in_index, in_local_id, out_index, out_local_id,
                          valid_index_in, valid_index_out);

    // Copy input to local memory
    if (valid_index_in) {
      local[in_local_id] = alpha_ * A[in_index];
    }

    id.barrier(sycl::access::fence_space::local_space);

    // Copy output from local memory
    if (valid_index_out) {
      At[out_index] = local[out_local_id];
    }
  }
}

// Transpose-Add
template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
SYCL_BLAS_INLINE bool
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::valid_thread(cl::sycl::nd_item<1> item) const {
  // Valid threads are established by ::eval()
  return true;
}

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
SYCL_BLAS_INLINE void
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::bind(cl::sycl::handler &cgh) {
  A_.bind(cgh);
  B_.bind(cgh);
  C_.bind(cgh);
}

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
SYCL_BLAS_INLINE typename in1_t::index_t
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::get_size() const {
  return C_.get_size();
}

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
SYCL_BLAS_INLINE void
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::adjust_access_displacement() {
  A_.adjust_access_displacement();
  B_.adjust_access_displacement();
  C_.adjust_access_displacement();
}

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
SYCL_BLAS_INLINE void
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::eval(cl::sycl::nd_item<1> id) {
  auto idx = id.get_global_linear_id();

  if (idx < get_size()) {
    auto A = A_.get_data().get_pointer();
    auto B = B_.get_data().get_pointer();
    auto C = C_.get_data().get_pointer();

    if constexpr (both_trans) {
      // Compute sum & then transpose
      auto j = idx / N_;
      auto i = idx - j * N_;

      auto in_index_a = i + j * lda_;
      auto in_index_b = i + j * ldb_;

      auto temp_sum = alpha_ * A[in_index_a] + beta_ * B[in_index_b];

      auto out_index_c = i * ldc_ + j;

      C[out_index_c] = temp_sum;

    } else {
      // Transpose A then compute sum (Applies to B as well)
      auto j = idx / M_;
      auto i = idx - j * M_;

      auto in_index_at = j + i * lda_;
      auto in_index_b = i + j * ldb_;

      auto temp_sum = alpha_ * A[in_index_at] + beta_ * B[in_index_b];

      auto out_index_c = i + j * ldc_;

      C[out_index_c] = temp_sum;
    }
  }
}

template <bool both_trans, int Tile_size, bool local_memory, typename in1_t,
          typename in2_t, typename out_t, typename element_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void
TransposeAdd<both_trans, Tile_size, local_memory, in1_t, in2_t, out_t,
             element_t>::eval(local_memory_t local_mem,
                              cl::sycl::nd_item<1> id) {
  auto idx = id.get_global_linear_id();

  if (idx < get_size()) {
    value_t *local = local_mem.localAcc.get_pointer();

    auto A = A_.get_data().get_pointer();
    auto B = B_.get_data().get_pointer();
    auto C = C_.get_data().get_pointer();

    index_t in_a_idx, in_b_idx, in_local_id, out_idx, out_local_id;

    if constexpr (both_trans) {
      compute_trans_add_indices<both_trans, Tile_size>(
          id, tile_count_n_, lda_, ldb_, ldc_, in_a_idx, in_b_idx, in_local_id,
          out_idx, out_local_id);

      // Compute & Copy sum/scaled input to local memory (before transpose)
      local[in_local_id] = alpha_ * A[in_a_idx] + beta_ * B[in_b_idx];

      id.barrier(sycl::access::fence_space::local_space);

      // Copy transposed output from local memory
      C[out_idx] = local[out_local_id];

    } else {
      compute_trans_add_indices<both_trans, Tile_size>(
          id, tile_count_m_, lda_, ldb_, ldc_, in_a_idx, in_b_idx, in_local_id,
          out_idx, out_local_id);

      // Compute transposed-scaled A & copy to local memory
      local[in_local_id] = alpha_ * A[in_a_idx];

      id.barrier(sycl::access::fence_space::local_space);

      // Compute & Copy output from local & global memory to global memory
      C[out_idx] = local[out_local_id] + beta_ * B[in_b_idx];
    }
  }
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_HPP