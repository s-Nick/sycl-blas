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

using index_t = int;

namespace reference_blas {

/**
 * @brief Reference omat-add implementation using reference omatcopy.
 *
 * @param trans_a (char) 'n' or 't' corresponding to non-transposed or
 * transposed matrix A respectively.
 * @param trans_b (char) 'n' or 't' corresponding to non-transposed or
 * transposed matrix B respectively.
 * @param m Number of rows in output matrix C
 * @param n Number of columns in output matrix C
 * @param alpha Scaling factor of matrix A
 * @param A (vector) Input matrix A
 * @param lda_m Matrix A leading dimension multiplier. (lda = lda_m * A_rows)
 * @param beta scaling factor of matrix B
 * @param B (vector) Input matrix B
 * @param ldb_m Matrix B leading dimension multiplier. (ldb = ldb_m * B_rows)
 * @param C (vector) Output matrix C
 * @param ldc_m Matrix C leading dimension multiplier. (ldc = ldc_m * C_rows)
 */
template <typename scalar_t>
void omatadd_batch(const char &trans_a, const char &trans_b, const index_t &m,
                   const index_t &n, const scalar_t &alpha,
                   std::vector<scalar_t> &A, const index_t &lda_m,
                   const index_t &stride_a_m, const scalar_t &beta,
                   std::vector<scalar_t> &B, const index_t &ldb_m,
                   const index_t &stride_b_m, std::vector<scalar_t> &C,
                   const index_t &ldc_m, const index_t &stride_c_m,
                   const index_t batch_size) {
  const index_t a_rows = trans_a == 't' ? n : m;
  const index_t a_cols = trans_a == 't' ? m : n;
  const index_t b_rows = trans_b == 't' ? n : m;
  const index_t b_cols = trans_b == 't' ? m : n;

  index_t lda = lda_m * a_rows;
  index_t ldb = ldb_m * b_rows;
  index_t ldc = ldc_m * m;

  // Base sizes of matrices
  index_t base_size = m * n;

  // Compute Strides using size-stride multipliers
  index_t stride_a = stride_a_m * base_size * lda_m;
  index_t stride_b = stride_b_m * base_size * ldb_m;
  index_t stride_c = stride_c_m * base_size * ldc_m;

  for (int b = 0; b < batch_size; b++) {
    // Temp Matrix 1 for computing A -> alpha * op(A)
    std::vector<scalar_t> TempMatrix1(stride_c, 0);
    omatcopy(trans_a, a_rows, a_cols, alpha, A.data() + b * stride_a, lda,
             TempMatrix1.data(), ldc);
    // Temp Matrix 2 for computing B -> beta * op(B)
    std::vector<scalar_t> TempMatrix2(stride_c, 0);
    omatcopy(trans_b, b_rows, b_cols, beta, B.data() + b * stride_b, ldb,
             TempMatrix2.data(), ldc);

    // Compute Sum of Temp matrices -> C
    for (index_t j = 0; j < n; j++) {
      for (index_t i = 0; i < m; i++) {
        C.at(i + j * ldc + b * stride_c) =
            TempMatrix1.at(i + j * ldc) + TempMatrix2.at(i + j * ldc);
      }
    }
  }
}
}  // namespace reference_blas

template <typename scalar_t>
using combination_t =
    std::tuple<char, char, index_t, index_t, scalar_t, scalar_t, index_t,
               index_t, index_t, index_t, index_t, index_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul, stride_a_m, stride_b_m,
      stride_c_m, batch_size;
  scalar_t alpha, beta;

  std::tie(trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul, ld_c_mul,
           stride_a_m, stride_b_m, stride_c_m, batch_size) = combi;

  if ((trans_a == 'n') && (trans_b == 'n')) {
    return;
  }

  // Rows & Cols per matrix
  const index_t a_rows = (trans_a == 't') ? n : m;
  const index_t a_cols = (trans_a == 't') ? m : n;
  const index_t b_rows = (trans_b == 't') ? n : m;
  const index_t b_cols = (trans_b == 't') ? m : n;

  index_t lda = ld_a_mul * a_rows;
  index_t ldb = ld_b_mul * b_rows;
  index_t ldc = ld_c_mul * m;

  // Base sizes of matrices
  index_t base_size = m * n;

  // Compute Strides using size-stride multipliers
  index_t stride_a = stride_a_m * base_size * ld_a_mul;
  index_t stride_b = stride_b_m * base_size * ld_b_mul;
  index_t stride_c = stride_c_m * base_size * ld_c_mul;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  std::vector<scalar_t> A(stride_a * batch_size);
  std::vector<scalar_t> B(stride_b * batch_size);
  std::vector<scalar_t> C(stride_c * batch_size, (scalar_t)0);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> C_ref = C;

  // Reference implementation
  reference_blas::omatadd_batch(trans_a, trans_b, m, n, alpha, A, ld_a_mul,
                                stride_a_m, beta, B, ld_b_mul, stride_b_m,
                                C_ref, ld_c_mul, stride_c_m, batch_size);

  auto m_a_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(A, stride_a * batch_size);
  auto m_b_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(B, stride_b * batch_size);
  auto m_c_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(C, stride_c * batch_size);

  blas::extension::_omatadd_batch(sb_handle, trans_a, trans_b, m, n, alpha,
                                  m_a_gpu, lda, stride_a, beta, m_b_gpu, ldb,
                                  stride_b, m_c_gpu, ldc, stride_c, batch_size);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), m_c_gpu, C.data(), stride_c * batch_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('n', 't'),         // trans_a
                       ::testing::Values<char>('n', 't'),         // trans_b
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(0, 2),         // alpha
                       ::testing::Values<scalar_t>(0, 2),         // beta
                       ::testing::Values<index_t>(1, 2),          // lda_mul
                       ::testing::Values<index_t>(1, 2),          // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3),       // ldc_mul
                       ::testing::Values<index_t>(1, 3),          // stride_a_m
                       ::testing::Values<index_t>(1, 3),          // stride_b_m
                       ::testing::Values<index_t>(1, 3),          // stride_c_m
                       ::testing::Values<index_t>(1, 5));         // batch_size

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul, stride_a_m, stride_b_m, stride_c_m,
      batch_size;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, trans_a, trans_b, m, n, alpha, beta, lda_mul,
                     ldb_mul, ldc_mul, stride_a_m, stride_b_m, stride_c_m,
                     batch_size);
}

BLAS_REGISTER_TEST_ALL(OmatAddBatch, combination_t, combi, generate_name);
