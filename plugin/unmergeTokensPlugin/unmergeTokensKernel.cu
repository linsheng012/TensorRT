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

#include "unmergeTokensKernel.h"

#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ uint16_t get_b_idx(int idx, int sx, int sy, int w, int wsx) {
  uint16_t wi = (idx % wsx) * sx;
  uint16_t hi = (idx / wsx) * sy;
  return hi * w + wi;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ uint16_t get_a_idx(int idx, int sx, int sy, int w, int wsx) {
  const int in_one_block = sy * w - wsx;
  const uint16_t li = idx % in_one_block;
  uint16_t local_idx = 0;
  if (li < w - wsx) {
    local_idx = li + li / (sx - 1) + 1;
  } else {
    local_idx = li + wsx;
  }
  const int h_blocks = idx / in_one_block;
  return h_blocks * sy * w + local_idx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void unmerge_tokens(ToMeParams params) {
  int s = blockIdx.x*4;
  const int n = blockIdx.y;
  const int c = threadIdx.x*8;
  const bool valid_thread = c < params.c;

  if (s < params.b_len) {
    uint32_t indices[4];
    indices[0] = get_b_idx(s+0, params.sx, params.sy, params.w, params.wsx);
    indices[1] = get_b_idx(s+1, params.sx, params.sy, params.w, params.wsx);
    indices[2] = get_b_idx(s+2, params.sx, params.sy, params.w, params.wsx);
    indices[3] = get_b_idx(s+3, params.sx, params.sy, params.w, params.wsx);
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
      if ( valid_thread ) {
        uint4 value = *reinterpret_cast<const uint4*>(&params.x_ptr[n*(params.unm_idx_lengths+params.b_len)*params.c + (params.unm_idx_lengths+s+i)*params.c + c]);
        *reinterpret_cast<uint4*>(&params.unmerged_tokens_ptr[n*params.h*params.w*params.c + indices[i]*params.c + c]) = value;
      }
    }
  } else if (s < params.b_len + params.unm_idx_lengths) {
    s = s - params.b_len;
    uint4 unm_indices = *reinterpret_cast<const uint4*>(&params.unm_idx_ptr[n*params.unm_idx_lengths+s]);
    uint32_t indices[4];
    indices[0] = unm_indices.x;
    indices[1] = unm_indices.y;
    indices[2] = unm_indices.z;
    indices[3] = unm_indices.w;

    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
      if ( valid_thread ) {
        uint32_t a_idx = get_a_idx(indices[i], params.sx, params.sy, params.w, params.wsx);
        uint4 value = *reinterpret_cast<const uint4*>(&params.x_ptr[n*(params.unm_idx_lengths+params.b_len)*params.c + (s + i)*params.c + c]);
        *reinterpret_cast<uint4*>(&params.unmerged_tokens_ptr[n*params.h*params.w*params.c + a_idx*params.c + c]) = value;
      }
    }
  } else {
    s = s - params.b_len - params.unm_idx_lengths;
    uint4 src_indices = *reinterpret_cast<const uint4*>(&params.src_idx_ptr[n*params.src_idx_lengths+s]);
    uint32_t indices[4];
    indices[0] = src_indices.x;
    indices[1] = src_indices.y;
    indices[2] = src_indices.z;
    indices[3] = src_indices.w;

    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
      if ( valid_thread ) {
        uint32_t a_idx = get_a_idx(indices[i], params.sx, params.sy, params.w, params.wsx);
        uint4 value = *reinterpret_cast<const uint4*>(&params.src_ptr[n*params.src_idx_lengths*params.c + (s + i)*params.c + c]);
        *reinterpret_cast<uint4*>(&params.unmerged_tokens_ptr[n*params.h*params.w*params.c + a_idx*params.c + c]) = value;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gather_src(ToMeParams params) {
  const int dst_idx = blockIdx.x * 4;
  const int n = blockIdx.y;
  const int c = threadIdx.x * 8;

  uint4 dst_indices = *reinterpret_cast<const uint4*>(&params.dst_idx_ptr[n*params.dst_idx_lengths+dst_idx]);
  uint32_t indices[4];
  indices[0] = dst_indices.x;
  indices[1] = dst_indices.y;
  indices[2] = dst_indices.z;
  indices[3] = dst_indices.w;

  #pragma unroll
  for ( int i = 0; i < 4; ++i ) {
    if ( c < params.c ) {
      uint4 value = *reinterpret_cast<const uint4*>(&params.x_ptr[n*(params.unm_idx_lengths+params.b_len)*params.c + (params.unm_idx_lengths + indices[i])*params.c + c]);
      *reinterpret_cast<uint4*>(&params.src_ptr[n*params.src_idx_lengths*params.c + (dst_idx+i)*params.c + c]) = value;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_unmerge_tokens_kernel(const ToMeParams& params, cudaStream_t stream) 
{
  {
    const int threads_per_cta = 64;
    gather_src<<<dim3(params.dst_idx_lengths/4, params.n), threads_per_cta, 0, stream>>>(params);
  }
  
  {
    const int threads_per_cta = 64;
    dim3 grid((params.b_len + params.unm_idx_lengths + params.src_idx_lengths)/4, params.n);
    unmerge_tokens<<<grid, threads_per_cta, 0, stream>>>(params);
  }
}
