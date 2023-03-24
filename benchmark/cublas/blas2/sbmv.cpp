/**************************************************************************
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
 *  @filename sbmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, int k) {
  std::ostringstream str{};
  str << "BM_Sbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << n << "/" << k;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSsbmv(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDsbmv(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         std::string uplo, index_t n, index_t k, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t xlen = n;
  index_t ylen = n;

  index_t lda = (k + 1);
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  const double n_d = static_cast<double>(n);
  const double k_d = static_cast<double>(k);

  state.counters["n"] = n_d;
  state.counters["k"] = k_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (2.0 * k_d + 1.0)) - (k_d * (k_d + 1.0));

  const double nflops_AtimesX = 2.0 * A_validVal;
  const double nflops_timesAlpha = n_d;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * n_d : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = A_validVal;
  const double mem_readX = n_d;
  const double mem_writeY = n_d;
  const double mem_readY = (beta != scalar_t{0}) ? n_d : 0;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(lda * n, m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_x_gpu(xlen, v_x.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_y_gpu(ylen, v_y.data());

  cublasFillMode_t c_uplo =
      (*uplo_str == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::sbmv(uplo_str, n, k, alpha, m_a.data(), lda, v_x.data(), incX,
                       beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp = v_y;
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> v_y_temp_gpu(
        ylen, v_y_temp.data());
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, k, &alpha, m_a_gpu, lda,
                             v_x_gpu, incX, &beta, v_y_temp_gpu, incY);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, k, &alpha, m_a_gpu, lda,
                             v_x_gpu, incX, &beta, v_y_gpu, incY);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, k, &alpha, m_a_gpu, lda,
                             v_x_gpu, incX, &beta, v_y_gpu, incY);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup);
  CUDA_CHECK(cudaStreamSynchronize(NULL));

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef_cuda(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * nflops_tot);

  blas_benchmark::utils::calc_avg_counters(state);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto sbmv_params = blas_benchmark::utils::get_sbmv_params<scalar_t>(args);

  for (auto p : sbmv_params) {
    std::string uplos;
    index_t n, k;
    scalar_t alpha, beta;
    std::tie(uplos, n, k, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         std::string uplos, index_t n, index_t k,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, uplos, n, k, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(uplos, n, k).c_str(),
                                 BM_lambda, cuda_handle_ptr, uplos, n, k, alpha,
                                 beta, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
