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

#include "mergeIndicesKernel.h"

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

template <typename T>
struct mySum
{
  __host__ __device__ __forceinline__ T operator()(T const& a, T const& b) const
  {
    return a + b;
  }
};

template struct mySum<float>;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ uint16_t fp32_to_fp16_rn(float f) {
  union { __half h; uint16_t u16; } tmp;
  tmp.h = __float2half_rn(f);
  return tmp.u16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float fp16_to_fp32(uint16_t u16) {
  union { __half h; uint16_t u16; } tmp;
  tmp.u16 = u16;
  return __half2float(tmp.h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fp16_to_fp32(uint32_t h2) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(h2));
  return make_float2(fp16_to_fp32(lo), fp16_to_fp32(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fp32_to_fp16_rn(float f0, float f1) {
  uint32_t h2;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(h2) : "f"(f0), "f"(f1));
#else
  uint16_t lo = fp32_to_fp16_rn(f0);
  uint16_t hi = fp32_to_fp16_rn(f1);
  asm volatile("mov.b32 %0, {%1, %2};\n" : "=r"(h2) : "h"(lo), "h"(hi));
#endif
  return h2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t TPB>
struct DataNormalizer {

  __device__ void apply(
      uint4& data)
  {
    float threadData = 0;
    constexpr int VPT = 8;
    float localX[VPT];

    float2 f2x = fp16_to_fp32(data.x);
    float2 f2y = fp16_to_fp32(data.y);
    float2 f2z = fp16_to_fp32(data.z);
    float2 f2w = fp16_to_fp32(data.w);
    localX[0] = f2x.x;
    localX[1] = f2x.y;
    localX[2] = f2y.x;
    localX[3] = f2y.y;
    localX[4] = f2z.x;
    localX[5] = f2z.y;
    localX[6] = f2w.x;
    localX[7] = f2w.y;

  #pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
      threadData += (localX[it] * localX[it]);
    }

    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float scale;

    const float sum = BlockReduce(temp_storage).Reduce(threadData, mySum<float>());
    if (threadIdx.x == 0)
    {
      scale = rsqrt(sum);
    }
    __syncthreads();

  #pragma unroll
    for (int32_t it = 0; it < VPT; it++)
    {
      localX[it] = localX[it] * scale;
    }

    data.x = fp32_to_fp16_rn(localX[1], localX[0]);
    data.y = fp32_to_fp16_rn(localX[3], localX[2]);
    data.z = fp32_to_fp16_rn(localX[5], localX[4]);
    data.w = fp32_to_fp16_rn(localX[7], localX[6]);
  }
};

template struct DataNormalizer<40>;

template<typename Normalizer>
__global__ void create_a_b_matrices(uint16_t* a_ptr, uint16_t* b_ptr, int b_len, int b_offset, ToMeIndicesParams params) {
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
    const int dst_tokens_before = ni * h * w * c / (sy * sx) + ((hi + 1) / sy) * w / sx * c + (wi / sx + 1) * static_cast<int>((hi % sy) == 0) * c;
    *reinterpret_cast<uint4*>(&a_ptr[linear_idx - dst_tokens_before]) = data;
  }
}

template __global__ void create_a_b_matrices<DataNormalizer<40>>(uint16_t* a_ptr, uint16_t* b_ptr, int b_len, int b_offset, ToMeIndicesParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS) 
    {
        std::stringstream ss;
        ss << "CuDNNassert: (" << code << ") " << cudnnGetErrorString(code) << " " << file << " " << line;
        std::cerr << ss.str() << std::endl;
        if (abort)
        {
            throw std::runtime_error(ss.str());
        }
    }
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::stringstream ss;
        ss << "CUDAassert: (" << code << ") " << cudaGetErrorString(code) << " " << file << " " << line;
        std::cerr << ss.str() << std::endl;
        if (abort)
        {
            throw std::runtime_error(ss.str());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct kvp_t {
  __half value;
  uint16_t idx;
};

struct half8 {
  __half x1;
  __half x2;
  __half x3;
  __half x4;
  __half x5;
  __half x6;
  __half x7;
  __half x8;

  __device__ half8() {}
} __attribute__((aligned(16)));

template <typename T>
struct myMax
{
  __host__ __device__ __forceinline__ T operator()(T const& a, T const& b) const
  {
    // return __hge(a.value, __hadd(b.value, __float2half(1e-5))) ? a : b;
    return __hge(a.value, b.value) ? a : b;
  }
};

union half_uint16 {
  __half h;
  uint16_t u;
};

template <int32_t TPB, int32_t VPT>
__global__ void reduce_max(ToMeIndicesParams params) {
  using BlockReduce = cub::BlockReduce<kvp_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  uint16_t tix = threadIdx.x * VPT;
  int s = blockIdx.x;
  int n = blockIdx.y;

  // Minimum half
  half_uint16 hu;
  hu.u = 0xfbff;

  kvp_t threadData[VPT];

  #pragma unroll
  for ( uint16_t i = 0; i < VPT; i += 8 ) {
    bool valid = tix + i < params.b_len;
    half8 values = valid ? *reinterpret_cast<const half8*>(&params.scores_ptr[n*params.a_len*params.b_len + s*params.b_len + tix + i]) : half8();
    threadData[i + 0] = {valid ? values.x1 : hu.h, static_cast<uint16_t>(tix + i + 0)};
    threadData[i + 1] = {valid ? values.x2 : hu.h, static_cast<uint16_t>(tix + i + 1)};
    threadData[i + 2] = {valid ? values.x3 : hu.h, static_cast<uint16_t>(tix + i + 2)};
    threadData[i + 3] = {valid ? values.x4 : hu.h, static_cast<uint16_t>(tix + i + 3)};
    threadData[i + 4] = {valid ? values.x5 : hu.h, static_cast<uint16_t>(tix + i + 4)};
    threadData[i + 5] = {valid ? values.x6 : hu.h, static_cast<uint16_t>(tix + i + 5)};
    threadData[i + 6] = {valid ? values.x7 : hu.h, static_cast<uint16_t>(tix + i + 6)};
    threadData[i + 7] = {valid ? values.x8 : hu.h, static_cast<uint16_t>(tix + i + 7)};
  }

  auto const max = BlockReduce(temp_storage).Reduce(threadData, myMax<kvp_t>());

  __syncthreads();

  if (threadIdx.x == 0)
  {
    params.max_in_row_ptr[n*params.a_len + s] = *reinterpret_cast<const uint16_t*>(&max.value);
    params.idx_max_in_row_ptr[n*params.a_len + s] = max.idx;
  }
}

template __global__ void reduce_max<128, 8> (ToMeIndicesParams params);
template __global__ void reduce_max<256, 8> (ToMeIndicesParams params);
template __global__ void reduce_max<256, 16>(ToMeIndicesParams params);
template __global__ void reduce_max<512, 8> (ToMeIndicesParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////


template<int32_t TPB, int32_t VPT>
__global__ void sort_max(ToMeIndicesParams params) {
  // Specialize BlockRadixSort for a 1D block of TPB threads owning VPT integer keys and values each
  typedef cub::BlockRadixSort<__half, TPB, VPT, uint16_t> BlockRadixSort;
  // Allocate shared memory for BlockRadixSort
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  // Obtain a segment of consecutive items that are blocked across threads
  __half thread_keys[VPT];
  uint16_t thread_values[VPT];

  const int n = blockIdx.x;
  const int idx = threadIdx.x * VPT;
  // Minimum half
  half_uint16 hu;
  hu.u = 0xfbff;

  #pragma unroll
  for (int i = 0; i < VPT; i+=8) {
    bool valid = idx + i < params.a_len;
    half8 keys = valid ? *reinterpret_cast<const half8*>(&params.max_in_row_ptr[n*params.a_len+idx+i]) : half8();
    thread_keys[i+0] = valid ? keys.x1 : hu.h;
    thread_keys[i+1] = valid ? keys.x2 : hu.h;
    thread_keys[i+2] = valid ? keys.x3 : hu.h;
    thread_keys[i+3] = valid ? keys.x4 : hu.h;
    thread_keys[i+4] = valid ? keys.x5 : hu.h;
    thread_keys[i+5] = valid ? keys.x6 : hu.h;
    thread_keys[i+6] = valid ? keys.x7 : hu.h;
    thread_keys[i+7] = valid ? keys.x8 : hu.h;
  }
  #pragma unroll
  for (int i = 0; i < VPT; ++i) {
    thread_values[i] = idx + i;
  }
  // Collectively sort the keys and values among block threads
  BlockRadixSort(temp_storage).SortDescending(thread_keys, thread_values);

  // if (idx == 72) {
  //   printf("%f %d %f %d %f %d\n", __half2float(thread_keys[0]), thread_values[0], __half2float(thread_keys[1]), thread_values[1], __half2float(thread_keys[2]), thread_values[2]);
  // }

  #pragma unroll
  for (int i = 0; i < VPT; i+=8) {
    uint4 values1;
    uint4 values2;
    values1.x = thread_values[i+0];
    values1.y = thread_values[i+1];
    values1.z = thread_values[i+2];
    values1.w = thread_values[i+3];
    values2.x = thread_values[i+4];
    values2.y = thread_values[i+5];
    values2.z = thread_values[i+6];
    values2.w = thread_values[i+7];
    if (idx + i < params.src_idx_lengths) {
      *reinterpret_cast<uint4*>(&params.src_idx_ptr[n*params.src_idx_lengths+idx+i]) = values1;
      *reinterpret_cast<uint4*>(&params.src_idx_ptr[n*params.src_idx_lengths+idx+i+4]) = values2;
      values1.x = params.idx_max_in_row_ptr[n*params.a_len+values1.x];
      values1.y = params.idx_max_in_row_ptr[n*params.a_len+values1.y];
      values1.z = params.idx_max_in_row_ptr[n*params.a_len+values1.z];
      values1.w = params.idx_max_in_row_ptr[n*params.a_len+values1.w];
      values2.x = params.idx_max_in_row_ptr[n*params.a_len+values2.x];
      values2.y = params.idx_max_in_row_ptr[n*params.a_len+values2.y];
      values2.z = params.idx_max_in_row_ptr[n*params.a_len+values2.z];
      values2.w = params.idx_max_in_row_ptr[n*params.a_len+values2.w];
      *reinterpret_cast<uint4*>(&params.dst_idx_ptr[n*params.src_idx_lengths+idx+i]) = values1;
      *reinterpret_cast<uint4*>(&params.dst_idx_ptr[n*params.src_idx_lengths+idx+i+4]) = values2;
    } else if (idx + i < params.src_idx_lengths + params.unm_idx_lengths) {
      *reinterpret_cast<uint4*>(&params.unm_idx_ptr[n*params.unm_idx_lengths+idx+i-params.src_idx_lengths]) = values1;
      *reinterpret_cast<uint4*>(&params.unm_idx_ptr[n*params.unm_idx_lengths+idx+i-params.src_idx_lengths+4]) = values2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ToMeRunner::ToMeRunner() {
  cublasLtCreate(&m_ltHandle);
  cublasLtMatmulDescCreate(&m_matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
}

ToMeRunner::~ToMeRunner() {
  // cublasLtDestroy(m_ltHandle);
  // cublasLtMatmulDescDestroy(m_matmul_desc);
}

void ToMeRunner::do_matmul(const ToMeIndicesParams& params,
                           cudaDataType_t type_A,
                           cudaDataType_t type_B,
                           cudaDataType_t type_D,
                           bool A_transposed,
                           bool B_transposed,
                           size_t m,
                           size_t n,
                           size_t k,
                           size_t ldA = 0,
                           size_t ldB = 0,
                           size_t ldD = 0,
                           size_t strideA = 0,
                           size_t strideB = 0,
                           size_t strideD = 0,
                           int batch_count = 1,
                           cudaStream_t stream = 0)
{
  auto op_A = CUBLAS_OP_N;
  size_t rows_A = m;
  size_t cols_A = k;
  if(A_transposed) {
      std::swap(rows_A, cols_A);
      op_A = CUBLAS_OP_T;
  }
  if( ldA == 0 ) { // Default leading dim.
      ldA = cols_A;
  }
  if(strideA == 0) { // Default batch stride.
      strideA = rows_A * cols_A;
  }

  auto op_B = CUBLAS_OP_N;
  size_t rows_B = k;
  size_t cols_B = n;
  if(B_transposed) {
      std::swap(rows_B, cols_B);
      op_B = CUBLAS_OP_T;
  }
  if( ldB == 0 ) { // Default leading dim.
      ldB = cols_B;
  }
  if(strideB == 0) { // Default batch stride.
      strideB = rows_B * cols_B;
  }

  size_t rows_D = m;
  size_t cols_D = n;
  if( ldD == 0 ) { // Default leading dim.
      ldD = cols_D;
  }
  if(strideD == 0) { // Default batch stride.
      strideD = rows_D * cols_D;
  }

  cublasLtMatrixLayout_t Adesc;
  cublasLtMatrixLayout_t Bdesc;
  cublasLtMatrixLayout_t Ddesc;

  cublasLtMatmulDescSetAttribute(m_matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_B, sizeof(op_B));
  cublasLtMatmulDescSetAttribute(m_matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_A, sizeof(op_A));

  // Need to swap rows <=> cols.
  cublasLtMatrixLayoutCreate(&Adesc, type_A, cols_A, rows_A, ldA);
  cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
  cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA));

  // Need to swap rows <=> cols.
  cublasLtMatrixLayoutCreate(&Bdesc, type_B, cols_B, rows_B, ldB);
  cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
  cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));

  // Need to swap rows <=> cols.
  cublasLtMatrixLayoutCreate(&Ddesc, type_D, cols_D, rows_D, ldD);
  cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
  cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideD, sizeof(strideD));

  
  float norm = 1.f;
  uint32_t beta = 0;
  uint32_t alpha = reinterpret_cast<const uint32_t &>( norm );
  cublasStatus_t status = cublasLtMatmul(m_ltHandle,
                                         m_matmul_desc,
                                         &alpha,
                                         params.b_ptr,
                                         Bdesc,
                                         params.a_ptr,
                                         Adesc,
                                         &beta,
                                         params.scores_ptr,
                                         Ddesc,
                                         params.scores_ptr,
                                         Ddesc,
                                         nullptr,  // &algo,
                                         reinterpret_cast<void*>(params.gemm_workspace_ptr),
                                         m_wsSizeBytes,
                                         stream);
}

void ToMeRunner::run_merge_indices_kernel(const ToMeIndicesParams& params, cudaStream_t stream) {
  assert(params.c == 320);
  {
    const int threads_per_cta = 40;
    dim3 grid(params.h * params.w, params.n);
    create_a_b_matrices<DataNormalizer<40>><<<grid, threads_per_cta, 0, stream>>>(
      params.a_ptr, params.b_ptr, params.b_len, 0, params);
  }

  {
    do_matmul(params, 
              CUDA_R_16F, // a
              CUDA_R_16F, // b
              CUDA_R_16F, // d
              false, // Q
              true, // K'
              params.a_len, // m
              params.b_len, // n 
              params.c, // k
              params.c, // ld Q
              params.c, // ld K
              params.b_len, // ld P
              params.a_len * params.c, // stride Q
              params.b_len * params.c, // stride K
              params.a_len * params.b_len, // stride P
              params.n, // batch count
              stream);
  }

  {
    if (params.b_len <= 1024) {
      reduce_max<128, 8><<<dim3(params.a_len, params.n), 128, 0, stream>>>(params);
    } else if (params.b_len <= 2048) {
      reduce_max<256, 8><<<dim3(params.a_len, params.n), 256, 0, stream>>>(params);
    } else if (params.b_len <= 4096) {
      reduce_max<256, 16><<<dim3(params.a_len, params.n), 256, 0, stream>>>(params);
    } else {
      assert(false);
    }
  }

  {
    if (params.a_len <= 3072) {
      sort_max<384, 8><<<params.n, 384, 0, stream>>>(params);
    } else if (params.a_len <= 12288) {
      sort_max<384, 32><<<params.n, 384, 0, stream>>>(params);
    } else {
      assert(false);
    }
  }
}
