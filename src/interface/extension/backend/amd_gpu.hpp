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
 *  @filename amd_gpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_EXTENSION_AMD_GPU_BACKEND_HPP
#define SYCL_BLAS_EXTENSION_AMD_GPU_BACKEND_HPP
#include "interface/extension_interface.h"

namespace blas {
namespace omatcopy_batch {
namespace backend {
template <bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_omatcopy_batch(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
                in_t in_memory, index_t ld_in, index_t in_stride,
                out_t out_memory, index_t ld_out, index_t out_stride,
                index_t batch_size) {
  return blas::extension::internal::_matcopy_batch_impl<
      64, false, sb_handle_t, element_t, index_t, in_t, out_t>(
      sb_handle, m, n, alpha, in_memory, ld_in, in_stride, out_memory, ld_out,
      out_stride, batch_size);
}
}  // namespace backend
}  // namespace omatcopy_batch
}  // namespace blas
#endif


