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
 *  @filename omatadd_batch_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

namespace reference_blas {

// blas-like extension omatAdd used as wrapper around omatcopy
template <typename scalar_t>
void omatadd(const char trans_a, const char trans_b, const index_t m,
             const index_t n, const scalar_t alpha, std::vector<scalar_t> &a,
             const index_t lda_m, const scalar_t beta, std::vector<scalar_t> &b,
             const index_t ldb_m, std::vector<scalar_t> &c,
             const index_t ldc_m) {
  const index_t a_rows = trans_a == 't' ? n : m;
  const index_t a_cols = trans_a == 't' ? m : n;
  const index_t b_rows = trans_b == 't' ? n : m;
  const index_t b_cols = trans_b == 't' ? m : n;

  index_t ldc = ldc_m * m;

  // Temp Matrix 1 for computing a -> alpha * op(A)
  std::vector<scalar_t> TempMatrix1(ldc * n, 0);
  omatcopy(trans_a, a_rows, a_cols, alpha, a.data(), lda_m * a_rows,
           TempMatrix1.data(), ldc);
  // Temp Matrix 2 for computing b -> beta * op(B)
  std::vector<scalar_t> TempMatrix2(ldc * n, 0);
  omatcopy(trans_b, b_rows, b_cols, beta, b.data(), ldb_m * b_rows,
           TempMatrix2.data(), ldc);

  // Compute Sum of Temp matrices -> c
  for (index_t j = 0; j < n; j++) {
    for (index_t i = 0; i < m; i++) {
      c.at(i + j * ldc) =
          TempMatrix1.at(i + j * ldc) + TempMatrix2.at(i + j * ldc);
    }
  }
}

}  // namespace reference_blas

// Parameters : trans_a, trans_b, m, n, alpha, beta, lda_m, ldb_m, ldc_m, batch_size
template <typename scalar_t>
using combination_t = std::tuple<char, char, index_t, index_t, scalar_t,
                                 scalar_t, index_t, index_t, index_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul, batch_size;
  scalar_t alpha, beta;

  std::tie(trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul, ld_c_mul,
           batch_size) = combi;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  index_t base_size = m * n;

  std::vector<scalar_t> A(base_size * ld_a_mul * batch_size);
  std::vector<scalar_t> B(base_size * ld_b_mul * batch_size);
  std::vector<scalar_t> C(base_size * ld_c_mul * batch_size, (scalar_t)0);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> C_ref = C;

  const index_t lda = (trans_a == 'n') ? m * ld_a_mul : n * ld_a_mul;
  const index_t ldb = (trans_b == 'n') ? m * ld_b_mul : n * ld_b_mul;
  const index_t ldc = m * ld_c_mul;

  const index_t stride_a = lda * n;
  const index_t stride_b = ldb * n;
  const index_t stride_c = ldc * n;

  // Reference implementation
  for (index_t i = 0; i < batch_size; ++i) {
    std::vector sub_a(A.begin() + i * stride_a, A.begin() + (i + 1) * stride_a);
    std::vector sub_b(B.begin() + i * stride_b, B.begin() + (i + 1) * stride_b);
    std::vector sub_c(C_ref.begin() + i * stride_c,
                      C_ref.begin() + (i + 1) * stride_c);
    reference_blas::omatadd(trans_a, trans_b, m, n, alpha, sub_a, ld_a_mul,
                            beta, sub_b, ld_b_mul, sub_c, ld_c_mul);
    std::copy(sub_c.begin(), sub_c.end(), C_ref.begin() + i*stride_c);
  }

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(
      A, base_size * ld_a_mul * batch_size);
  auto m_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(
      B, base_size * ld_b_mul * batch_size);
  auto m_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(
      C, base_size * ld_c_mul * batch_size);

  blas::extension::_omatadd_batch(sb_handle, trans_a, trans_b, m, n, alpha,
                                  m_a_gpu, lda, stride_a, beta, m_b_gpu, ldb,
                                  stride_b, m_c_gpu, ldc, stride_c, batch_size);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), m_c_gpu, C.data(), base_size * ld_c_mul*batch_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values<char>('n'), ::testing::Values<char>('n'),
    ::testing::Values<index_t>(6, 63, 129),
    ::testing::Values<index_t>(6, 63, 129),
    ::testing::Values<scalar_t>(0, 1, 2), // alpha
    ::testing::Values<scalar_t>(0, 1, 2), // beta
    ::testing::Values<index_t>(1, 2, 3),  // lda_mul
    ::testing::Values<index_t>(1, 2, 3),  // ldb_mul
    ::testing::Values<index_t>(1, 2, 3),  // ldc_mul
    ::testing::Values<index_t>(1, 2, 4)); // batch_size

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul, batch_size;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, trans_a, trans_b, m, n, alpha, beta, lda_mul,
                     ldb_mul, ldc_mul, batch_size);
}

BLAS_REGISTER_TEST_ALL(OmatAdd, combination_t, combi, generate_name);
