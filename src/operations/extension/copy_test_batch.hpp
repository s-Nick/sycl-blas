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
    lhs_t lhs, rhs_t rhs_1, typename lhs_t::value_t alpha,
    typename lhs_t::value_t beta, typename rhs_t::index_t m,
    typename rhs_t::index_t n, typename rhs_t::index_t lhs_ld,
    typename rhs_t::index_t rhs_ld, typename rhs_t::index_t lhs_stride,
    typename rhs_t::index_t rhs_stride, typename rhs_t::index_t batch_size)
    : lhs_(lhs),
      rhs_1_(rhs_1),
      alpha_(alpha),
      beta_(beta),
      m_(m),
      n_(n),
      lhs_ld_(lhs_ld),
      rhs_1_ld_(rhs_ld),
      lhs_stride_(lhs_stride),
      rhs_1_stride_(rhs_stride),
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
  const index_t local_id = ndItem.get_local_id(0);
  const index_t group_id = ndItem.get_group(0);
  const index_t group_range = ndItem.get_group_range(0);

  // const index_t wg_batch_id = ndItem.get_group(0) / ((m_ * n_ - 1) / 8);

  if (group_id >= batch_size_) {
    return 0;
  }
  const index_t l_rhs_stride = rhs_1_stride_;
  const index_t l_lhs_stride = lhs_stride_;

  auto orig_lhs = lhs_.get_pointer();    // + (group_id * l_lhs_stride);
  auto orig_rhs = rhs_1_.get_pointer();  // + (group_id * l_rhs_stride);

  index_t item_id =
      ndItem.get_group(0) * ndItem.get_group_range(0) + ndItem.get_local_id(0);

  // if (local_id >= m_ * n_) {
  // return 0;
  //}
  // const index_t row = item_id % m_;
  // const index_t col = item_id / m_;

  index_t row, col;

  index_t batch_stride_ = batch_size_;

  // auto A = orig_rhs;
  // auto B = orig_lhs;

  /*
  B += item_id;
  *B = item_id;
  auto A = orig_rhs;
  auto B = orig_lhs;
  for (int i = local_id; i < 128; i += lhs_stride_) {
    *B = 2;  // alpha_ * (*A);
    A = A + i%rhs_1_stride_;
    B = B + i%lhs_stride_;
  }
  const index_t loop_stride = (ndItem.get_local_range(0) < m_ * n_)
                                  ? ndItem.get_local_range(0)
                                  : m_ * n_;
  const index_t bigger_matrix =
      (l_lhs_stride < l_rhs_stride) ? l_lhs_stride : l_rhs_stride;
    */
  do {
    auto A = orig_rhs;
    auto B = orig_lhs;

    for (int i = local_id, c = 1; i < m_ * n_;
         i += ndItem.get_local_range(0), ++c) {
      row = i % m_;
      col = i / m_;
      A = A + row + col * rhs_1_ld_;
      B = B + row + col * lhs_ld_;
      // B[0] = i;//batch_stride_;//ndItem.get_local_range(0);
      B[0] = alpha_ * A[0];
      A = orig_rhs;
      B = orig_lhs;
    }
    /*
    A = A + item_id;
    B = B + item_id + lhs_ld_;
    */
    orig_rhs += l_rhs_stride;
    orig_lhs += l_lhs_stride;
    batch_stride_ -= 1;
  } while (batch_stride_ > 0);

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
