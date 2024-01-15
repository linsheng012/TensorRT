/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_MERGE_INDICES_KERNEL_H
#define TRT_MERGE_INDICES_KERNEL_H

#include "common/checkMacrosPlugin.h"
#include "mergeIndicesPluginCommon.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>

static inline __host__ __device__ int32_t divUp(int32_t m, int32_t n)
{
    return (m + n - 1) / n;
}

class ToMeRunner {
public:
  ToMeRunner();

  ~ToMeRunner();

  void run_merge_indices_kernel(const ToMeIndicesParams& params, cudaStream_t stream);

  size_t get_gemm_workspace() const { return m_wsSizeBytes; }

private:

  void do_matmul(const ToMeIndicesParams& params,
                 cudaDataType_t type_A, cudaDataType_t type_B,
                 cudaDataType_t type_D,
                 bool A_transposed, bool B_transposed,
                 size_t m, size_t n, size_t k,
                 size_t ldA, size_t ldB, size_t ldD,
                 size_t strideA, size_t strideB, size_t strideD,
                 int batch_count,
                 cudaStream_t stream);

private:
  size_t m_wsSizeBytes = 4 * 1024 * 1024;
  cublasLtHandle_t m_ltHandle;
  cublasLtMatmulDesc_t m_matmul_desc;
};

#endif // TRT_UNMERGE_TOKENS_KERNEL_H
