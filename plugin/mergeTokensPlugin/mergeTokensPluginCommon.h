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

#ifndef TRT_MERGE_TOKENS_PLUGIN_COMMON_H
#define TRT_MERGE_TOKENS_PLUGIN_COMMON_H

#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

struct ToMeParams {
  int n;
  int h;
  int w;
  int c;
  float r;
  int sx;
  int sy;
  int a_len;
  int b_len;
  int src_idx_lengths;
  int dst_idx_lengths;
  int unm_idx_lengths;
  const uint16_t* x_ptr;
  const uint32_t* unm_idx_ptr;
  uint16_t* merged_tokens_ptr;
  uint16_t* a_ptr;
};

#endif // TRT_MERGE_TOKENS_PLUGIN_COMMON_H
