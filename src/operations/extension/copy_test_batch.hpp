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
  const index_t workgroup_cluster =
      ((m_ * n_ - 1) / ndItem.get_local_range(0) + 1);
  const index_t wg_batch_id = ndItem.get_group(0) / workgroup_cluster;

  // This will disable all workgroups that dont have any batch to work on
  if (wg_batch_id >= batch_size_) {
    return 0;
  }

  const index_t batch_stride = ndItem.get_group_range(0) / workgroup_cluster;

  const index_t l_size = m_ * lhs_ld_;
  const index_t r_size = m_ * rhs_1_ld_;

  const index_t l_lhs_stride = lhs_stride_;
  const index_t l_rhs_stride = rhs_1_stride_;

  index_t item_id =
      (ndItem.get_group(0) % workgroup_cluster) * (ndItem.get_local_range(0)) +
      ndItem.get_local_id(0);
  if (item_id >= m_ * n_) {
    return 0;
  }
  auto orig_lhs = lhs_.get_pointer() + (wg_batch_id * l_lhs_stride);
  auto orig_rhs = rhs_1_.get_pointer() + (wg_batch_id * l_rhs_stride);

  const index_t row = item_id % m_;
  const index_t col = item_id / m_;

  orig_lhs = orig_lhs + row + col * lhs_ld_;
  orig_rhs = orig_rhs + row + col * rhs_1_ld_;

  // omatadd operator has another matrix as input and the computation is 
  // slighlty different.
  if constexpr (op == matcopy_op::outplaceadd) {
    auto orig_rhs_2 = rhs_2_.get_pointer() + (wg_batch_id * rhs_2_stride_);
    orig_rhs_2 = orig_rhs_2 + row + col * rhs_2_ld_;
    do {
      auto A = orig_rhs;
      auto B = orig_rhs_2;
      auto C = orig_lhs;

      C[0] = alpha_ * A[0] + beta_ * B[0];

      orig_lhs += (lhs_stride_ * batch_stride);
      orig_rhs += (rhs_1_stride_ * batch_stride);
      orig_rhs_2 += (rhs_2_stride_ * batch_stride);

      batch_size_ -= batch_stride;

    } while (batch_size_ > wg_batch_id);
  } else {
    do {
      auto A = orig_rhs;
      auto B = orig_lhs;

      B[0] = alpha_ * A[0];

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
                                     rhs_t>::bind(cl::sycl::handler &h) {
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
