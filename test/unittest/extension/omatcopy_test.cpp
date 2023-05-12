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
 *  @filename omatcopy_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<char, index_t, index_t, scalar_t, index_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans;
  index_t m, n, ld_in, ld_out;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, ld_in, ld_out) = combi;

  // bail out early if the leading dimensions are not correct
  if (ld_in < m || ld_out < (trans == 't' ? n : m)) return;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  //index_t size = std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n);
  index_t m_a_size = std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n); 
  index_t m_b_size = std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n); 
  std::vector<scalar_t> A(m_a_size);
  std::vector<scalar_t> B(m_b_size);

  fill_random(A);
  //fill_random(B);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;

  // Reference implementation
  reference_blas::omatcopy(trans, m, n, alpha, A_ref.data(), ld_in,
                           B_ref.data(), ld_out);

  auto matrix_in = blas::make_sycl_iterator_buffer<scalar_t>(A, m_a_size);
  auto matrix_out = blas::make_sycl_iterator_buffer<scalar_t>(B, m_b_size);

  blas::extension::_omatcopy(sb_handle, trans, m, n, alpha, matrix_in, ld_in,
                             matrix_out, ld_out);
  auto event = blas::helper::copy_to_host<scalar_t>(sb_handle.get_queue(),
                                                    matrix_out, B.data(), m_b_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values<char>('n'), ::testing::Values<index_t>(64, 128, 256, 512),
    ::testing::Values<index_t>(64, 128, 256, 512),
    ::testing::Values<scalar_t>(0, 1, 2),
    ::testing::Values<index_t>(64, 128, 256, 512),
    ::testing::Values<index_t>(64, 128, 256, 512));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  index_t m, n, ld_in, ld_out;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, ld_in, ld_out);
}

BLAS_REGISTER_TEST_ALL(OmatCopy, combination_t, combi, generate_name);
