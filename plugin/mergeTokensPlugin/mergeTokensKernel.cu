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

#include "mergeTokensKernel.h"

#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct DataNormalizerDummy {
    __device__ void apply(uint4& data) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Normalizer>
__global__ void create_a_b_matrices(uint16_t* a_ptr, uint16_t* b_ptr, int b_len, int b_offset, ToMeParams params) {
  constexpr int BYTES_PER_ELEMENT = 2;
  constexpr int CHANNELS_PER_LOAD = 16 / BYTES_PER_ELEMENT;

  const int ni = blockIdx.y;
  const int hwi = blockIdx.x;
  const int ci = threadIdx.x * CHANNELS_PER_LOAD;
  const int h = params.h;
  const int w = params.w;
  const int c = params.c;
  const int sy = params.sy;
  const int sx = params.sx;
  const int wi = hwi % w;
  const int hi = hwi / w;

  const int linear_idx = ni * h * w * c + hi * w * c + wi * c + ci;
  uint4 data = reinterpret_cast<const uint4*>(&params.x_ptr[linear_idx])[0];

  Normalizer normalizer;
  normalizer.apply(data);
  
  if (wi % sx == 0 && hi % sy == 0) {
    const int b_idx = ni * (b_len + b_offset) * c + (b_offset + hi / sy * w / sx + wi / sx) * c + ci;
    *reinterpret_cast<uint4*>(&b_ptr[b_idx]) = data;
  } else {
    int dst_tokens_before = ni * h * w * c / (sy * sx) + ((hi + 1) / sy) * w / sx * c + (wi / sx + 1) * static_cast<int>((hi % sy) == 0) * c;
    *reinterpret_cast<uint4*>(&a_ptr[linear_idx - dst_tokens_before]) = data;
  }
}

template __global__ void create_a_b_matrices<DataNormalizerDummy>(uint16_t* a_ptr, uint16_t* b_ptr, int b_len, int b_offset, ToMeParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gather_and_concat(ToMeParams params) {
  const int unm_idx = blockIdx.x * 4;
  const int n = blockIdx.y;
  const int c = threadIdx.x * 8;

  uint4 unm_indices = *reinterpret_cast<const uint4*>(&params.unm_idx_ptr[n*params.unm_idx_lengths+unm_idx]);
  uint32_t indices[4];
  indices[0] = unm_indices.x;
  indices[1] = unm_indices.y;
  indices[2] = unm_indices.z;
  indices[3] = unm_indices.w;

  #pragma unroll
  for ( int i = 0; i < 4; ++i ) {
    if ( c < params.c ) {
      uint4 value = *reinterpret_cast<const uint4*>(&params.a_ptr[n*params.a_len*params.c + indices[i]*params.c + c]);
      *reinterpret_cast<uint4*>(&params.merged_tokens_ptr[n*(params.unm_idx_lengths+params.b_len)*params.c + (unm_idx+i)*params.c + c]) = value;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_merge_tokens_kernel(const ToMeParams& params, cudaStream_t stream) 
{
  {
    const int threads_per_cta = 40;
    dim3 grid(params.h * params.w, params.n);
    create_a_b_matrices<DataNormalizerDummy><<<grid, threads_per_cta, 0, stream>>>(
      params.a_ptr, params.merged_tokens_ptr, params.b_len, params.unm_idx_lengths, params);
  }
  
  {
    const int threads_per_cta = 64;
    dim3 grid(params.unm_idx_lengths/4, params.n);
    gather_and_concat<<<grid, threads_per_cta, 0, stream>>>(params);
  }
}
