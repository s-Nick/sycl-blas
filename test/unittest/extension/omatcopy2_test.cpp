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
 *  @filename omatcopy2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "common/system_reference_blas.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<char, int64_t, int64_t, scalar_t, int64_t, int64_t>;

#include <cstdint>

template <bool col_major, typename T>
void print_matr_strided(std::vector<T>& m, const int rows, const int cols,
                        const int stride) {
  const int ldm = (col_major) ? stride * (rows - 1) + 1 : cols;

  std::cout << std::setw(4);
  for (int i = 0; i < ldm; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(4)
                << ((col_major) ? m[i + j * ldm] : m[j + i * ldm]) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

template <bool col_major, typename T>
void print_matr(std::vector<T>& m, const int rows, const int cols) {
  const int ldm = (col_major) ? rows : cols;

  std::cout << std::setw(4);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(4)
                << ((col_major) ? m[i + j * ldm] : m[j + i * ldm]) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

template <bool col_major, typename scalar_t, typename index_t>
std::enable_if_t<col_major, std::vector<scalar_t>> omatcopy2(
    const char t, const index_t rows, const index_t cols, const scalar_t alpha,
    const std::vector<scalar_t>& in_matrix, const index_t ldm,
    const index_t stride, std::vector<scalar_t>& out_matrix,
    const index_t out_ldm, const index_t out_stride) {
  for (int i = 0; i <= cols; ++i) {
    for (int j = 0, c = 0; j < rows; ++j, ++c) {
      if (((j * out_stride + i * out_ldm) < out_matrix.size()) &&
          (j * stride + i * ldm) < in_matrix.size()) {
        out_matrix[j * out_stride + i * out_ldm] =
            alpha * in_matrix[j * stride + i * ldm];
      }
    }
  }
  return out_matrix;
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans;
  int64_t m, n, stride_in, stride_out;
  scalar_t alpha;

  std::tie(trans, m, n, alpha, stride_in, stride_out) = combi;

  // Compute ld_in and ld_out following oneMKL documentation at
  // https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/omatcopy2.html#onemkl-blas-omatcopy2
  int64_t ld_in = stride_in * (m - 1) + 1;
  int64_t ld_out =
      (trans != 't') ? stride_out * (m - 1) + 1 : stride_out * (n - 1) + 1;

  // bail out early if the leading dimensions are not correct
  if (ld_in < m || ld_out < (trans == 't' ? n : m)) return;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  int64_t m_a_size =
      std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n);
  int64_t m_b_size =
      std::max(ld_in, ld_out) * (trans == 't' ? std::max(m, n) : n);
  std::vector<scalar_t> A(m_a_size);
  std::vector<scalar_t> B(m_b_size);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;

  // Reference implementation
  omatcopy2<true>(trans, m, n, alpha, A_ref, ld_in, stride_in, B_ref, ld_out,
                  stride_out);

  auto matrix_in = blas::make_sycl_iterator_buffer<scalar_t>(A, m_a_size);
  auto matrix_out = blas::make_sycl_iterator_buffer<scalar_t>(B, m_b_size);

  blas::extension::_omatcopy2(sb_handle, trans, m, n, alpha, matrix_in, ld_in,
                              stride_in, matrix_out, ld_out, stride_out);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), matrix_out, B.data(), m_b_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(B, B_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values<char>('n'),
                                      ::testing::Values<int64_t>(6, 7, 8),
                                      ::testing::Values<int64_t>(6, 7, 8),
                                      ::testing::Values<scalar_t>(0, 1, 2),
                                      ::testing::Values<int64_t>(6, 7, 8),
                                      ::testing::Values<int64_t>(6, 7, 8));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char trans;
  int64_t m, n, ld_in, ld_out, stride_in, stride_out;
  T alpha;
  BLAS_GENERATE_NAME(info.param, trans, m, n, alpha, stride_in, stride_out);
}

BLAS_REGISTER_TEST_ALL(OmatCopy, combination_t, combi, generate_name);
