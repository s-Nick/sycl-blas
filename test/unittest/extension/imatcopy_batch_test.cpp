/***************************************************************************
 *
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
 *  @filename imatcopy_batch_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<char, int64_t, int64_t, scalar_t, int64_t, int64_t, int64_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans;
  int64_t m, n, ld_in, ld_out, batch_size;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, ld_in, ld_out, batch_size) = combi;

  // bail out early if the leading dimensions are not correct
  if (ld_in < m || ld_out < (trans == 't' ? n : m)) return;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  int64_t stride_a = ld_in * n;
  int64_t stride_b = (trans == 't') ? ld_out * m : ld_out * n;

  int64_t size = std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n);
  std::vector<scalar_t> A(size * batch_size);

  fill_random(A);

  std::vector<scalar_t> A_ref = A;

  // Reference implementation does not provide batch version,
  // the loop overcome this limitation
  for (int i = 0; i < batch_size; ++i) {
    reference_blas::imatcopy(trans, m, n, alpha, A_ref.data() + (i * stride_a),
                             ld_in, ld_out);
  }

  auto matrix_in_out =
      blas::make_sycl_iterator_buffer<scalar_t>(A, size * batch_size);

  blas::extension::_imatcopy_batch(sb_handle, trans, m, n, alpha, matrix_in_out,
                                   ld_in, ld_out, stride_a, batch_size);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_in_out, A.data(), size * batch_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(A, A_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values<char>('n'),
                                      ::testing::Values<int64_t>(2,3),
                                      ::testing::Values<int64_t>(2,3),
                                      ::testing::Values<scalar_t>(0,1,2),
                                      ::testing::Values<int64_t>(2,3),
                                      ::testing::Values<int64_t>(2,3),
                                      ::testing::Values<int64_t>(1, 2 ,4 ,8));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  int64_t m, n, ld_in, ld_out, batch_size;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, ld_in, ld_out, batch_size);
}

BLAS_REGISTER_TEST_ALL(ImatCopy_batch, combination_t, combi, generate_name);
