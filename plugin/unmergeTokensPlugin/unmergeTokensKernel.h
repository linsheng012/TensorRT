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
#ifndef TRT_UNMERGE_TOKENS_KERNEL_H
#define TRT_UNMERGE_TOKENS_KERNEL_H

#include "common/checkMacrosPlugin.h"
#include "unmergeTokensPluginCommon.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

static inline __host__ __device__ int32_t divUp(int32_t m, int32_t n)
{
    return (m + n - 1) / n;
}

void run_unmerge_tokens_kernel(const ToMeParams& params, cudaStream_t stream);

#endif // TRT_UNMERGE_TOKENS_KERNEL_H
