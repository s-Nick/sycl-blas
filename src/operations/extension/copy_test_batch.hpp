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
 *  @filename matcopy.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_COPY_TEST_BATCH_HPP
#define SYCL_BLAS_EXTENSION_COPY_TEST_BATCH_HPP

#include "operations/extension/copy_test_batch.h"
#include "operations/extension/matcopy.h"
#include <type_traits>

namespace blas {

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
Copytest_batch<op, ClSize, trans_rhs_1, lhs_t, rhs_t>::Copytest_batch(
    lhs_t lhs, rhs_t rhs_1, rhs_t rhs_2, typename lhs_t::value_t alpha,
    typename lhs_t::value_t beta, typename rhs_t::index_t m,
    typename rhs_t::index_t n, typename rhs_t::index_t lhs_ld,
    typename rhs_t::index_t rhs_ld, typename rhs_t::index_t rhs_2_ld,
    typename rhs_t::index_t lhs_stride, typename rhs_t::index_t rhs_stride,
    typename rhs_t::index_t rhs_2_stride, typename rhs_t::index_t batch_size)
    : lhs_(lhs),
      rhs_1_(rhs_1),
      rhs_2_(rhs_2),
      alpha_(alpha),
      beta_(beta),
      m_(m),
      n_(n),
      lhs_ld_(lhs_ld),
      rhs_1_ld_(rhs_ld),
      rhs_2_ld_(rhs_2_ld),
      lhs_stride_(lhs_stride),
      rhs_1_stride_(rhs_stride),
      rhs_2_stride_(rhs_2_stride),
      batch_size_(batch_size) {}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
typename lhs_t::value_t
Copytest_batch<op, ClSize, trans_rhs_1, lhs_t, rhs_t>::eval(index_t i) {}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
typename lhs_t::value_t
Copytest_batch<op, ClSize, trans_rhs_1, lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t m{m_};
  const index_t n{n_};
  constexpr index_t workgroup_cluster =
      64;  // set to 32 to test warp size on nvidia gpu

  const index_t required_tile =
      (m * n % workgroup_cluster == 0 && !(m % workgroup_cluster) &&
       !(n % workgroup_cluster))
          ? (((m * n - 1) / (workgroup_cluster * workgroup_cluster)) + 1)
          : (((m * n - 1) / (workgroup_cluster * workgroup_cluster)) + 1) +
                (m / workgroup_cluster) + (n / workgroup_cluster) + 1;  // +1;

  const index_t wg_batch_id = ndItem.get_group(0) / required_tile;

  // This will disable all workgroups that dont have any batch to work on
  if (wg_batch_id >= batch_size_) {
    return 0;
  }

  const index_t batch_stride = ndItem.get_group_range(0) / required_tile;

  const index_t l_size = m_ * lhs_ld_;
  const index_t r_size = m_ * rhs_1_ld_;

  const index_t l_lhs_stride = lhs_stride_;
  const index_t l_rhs_stride = rhs_1_stride_;

  const index_t number_of_block_per_row = ((m_ - 1) / workgroup_cluster) + 1;
  // const index_t number_of_block_per_col = ((n_ - 1) / workgroup_cluster ) +
  // 1;
  // const index_t y_dim_split = workgroup_cluster / workgroup_cluster;

  const index_t wg_id = ndItem.get_group(0) % required_tile;
  /*
  const index_t wg_id =
      ndItem.get_group(0) %
      ((m_ * n_ - 1) / (workgroup_cluster * workgroup_cluster) + 1);
      */

  /* row tile id  per work group */
  const index_t tile_id_row = wg_id % number_of_block_per_row;
  /* column tile id per work group */
  const index_t tile_id_col = wg_id / number_of_block_per_row;
  /* the start position of the tile-row per work group */
  const index_t wg_row = tile_id_row * workgroup_cluster;
  /* the start position of the tile-column per work group */
  const index_t wg_col = tile_id_col * workgroup_cluster;

  const index_t item_id = ndItem.get_local_id(0);

  /*
  index_t tile_offset_col =
      (ndItem.get_group(0) %
       ((m_ * n_ - 1) / (workgroup_cluster * workgroup_cluster) + 1)) *
      (workgroup_cluster * rhs_1_ld_);
  index_t tile_offset_row =
      (ndItem.get_group(0) %
       ((m_ * n_ - 1) / (workgroup_cluster * workgroup_cluster) + 1)) *
      workgroup_cluster;

  index_t item_id =
      (ndItem.get_group(0) % workgroup_cluster) * (ndItem.get_local_range(0)) +
      ndItem.get_local_id(0);
  if (item_id >= m_ * n_) {
    return 0;
  }
  */

  auto orig_lhs = lhs_.get_pointer() + (wg_batch_id * l_lhs_stride);
  auto orig_rhs = rhs_1_.get_pointer() + (wg_batch_id * l_rhs_stride);

  // const index_t row = item_id % m_;
  // const index_t col = item_id / m_;
  orig_lhs = orig_lhs + wg_row + wg_col * lhs_ld_ + item_id;
  orig_rhs = orig_rhs + wg_row + wg_col * rhs_1_ld_ + item_id;

  // orig_lhs = orig_lhs + row + col * lhs_ld_;
  // orig_rhs = orig_rhs + row + col * rhs_1_ld_;
  value_t reg_rhs[workgroup_cluster];
  const index_t alpha = alpha_;

  const bool is_internal_block =
      (m - wg_row >= workgroup_cluster) && (n - wg_col >= workgroup_cluster);

  // check for short&large
  const bool valid_index = (item_id > m || (item_id >= (m-wg_row)) ) ? false : true;
  if (!valid_index) return 0;

  if (is_internal_block) {
    do {
      auto A = orig_rhs;
      auto B = orig_lhs;

#pragma unroll
      for (int i = 0; i < workgroup_cluster; ++i) {
        // if(ndItem.get_local_id(0) == 0)
        // if (item_id + i * rhs_1_ld_ > r_size) break;
        reg_rhs[i] = A[i * rhs_1_ld_];
        /*
        B[ndItem.get_local_id(0) + i * lhs_ld_] =  // wg_id;
            alpha_ * A[ndItem.get_local_id(0) + i * rhs_1_ld_];
            */
      }
#pragma unroll
      for (int i = 0; i < workgroup_cluster; ++i) {
        // if (ndItem.get_local_id(0) + i * lhs_ld_ > l_size) break;
        // if(ndItem.get_local_id(0) == 0)
        B[i * lhs_ld_] = alpha * reg_rhs[i];
      }
      // B[0] = alpha_ * A[0];

      orig_lhs += (lhs_stride_ * batch_stride);
      orig_rhs += (rhs_1_stride_ * batch_stride);

      batch_size_ -= batch_stride;
    } while (batch_size_ > wg_batch_id);
  } else {
    const auto limit_m = m - wg_row;
    const auto limit_n = n - wg_col;
    //const auto limit_n = n - ;
    // if (limit_m == 0 || limit_n ==0 ) return 0;
    do {
      auto A = orig_rhs;
      auto B = orig_lhs;

      for (int i = 0; i < workgroup_cluster; ++i) {
        // if(ndItem.get_local_id(0) == 0)
        // if (item_id + i * rhs_1_ld_ > r_size) break;
        if ( i >= limit_n ) break;
        reg_rhs[i] = A[i * rhs_1_ld_];
        /*
        B[ndItem.get_local_id(0) + i * lhs_ld_] =  // wg_id;
            alpha_ * A[ndItem.get_local_id(0) + i * rhs_1_ld_];
            */
      }
      for (int i = 0; i < workgroup_cluster; ++i) {
        // if(ndItem.get_local_id(0) == 0)
        // if (item_id + i * lhs_ld_ >= limit_n || item_id + i >= limit_m )
        // break;
        if ( i >= limit_n ) break;
        B[i * lhs_ld_] =  alpha * reg_rhs[i];
        // B[0] = alpha * reg_rhs[i];
      }
      // B[0] = alpha_ * A[0];

      orig_lhs += (lhs_stride_ * batch_stride);
      orig_rhs += (rhs_1_stride_ * batch_stride);

      batch_size_ -= batch_stride;
    } while (batch_size_ > wg_batch_id);
  }

  return 0;
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void Copytest_batch<op, ClSize, trans_rhs_1, lhs_t,
                                     rhs_t>::bind(cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void Copytest_batch<op, ClSize, trans_rhs_1, lhs_t,
                                     rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE typename rhs_t::index_t
Copytest_batch<op, ClSize, trans_rhs_1, lhs_t, rhs_t>::get_size() const {
  return m_ * n_;
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE bool
Copytest_batch<op, ClSize, trans_rhs_1, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_MATCOPY_HPP
