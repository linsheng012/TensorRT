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

#include "mergeTokensPlugin.h"
#include "mergeTokensKernel.h"
#include "mergeTokensPluginCommon.h"
#include "common/bertCommon.h"
#include "npy.hpp"
#include <cmath>

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::MergeTokensPlugin;
using nvinfer1::plugin::MergeTokensPluginCreator;

namespace
{
static std::string const kMERGE_TOKENS_PLUGIN_NAME{"MergeTokens"};
static std::string const kMERGE_TOKENS_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float) + 5 * sizeof(int32_t)};
} // namespace

// class MergeTokensPlugin
MergeTokensPlugin::MergeTokensPlugin(std::string const& name, int32_t sx, int32_t sy, float r)
    : mName(name)
    , mSx(sx)
    , mSy(sy)
    , mR(r)
{
}

MergeTokensPlugin::MergeTokensPlugin(std::string const& name, void const* buffer, size_t length)
    : mName(name)
{
    PLUGIN_VALIDATE(buffer != nullptr);
    PLUGIN_VALIDATE(length == kSERIALIZATION_SIZE);

    auto const* d = static_cast<char const*>(buffer);
    auto const* a = d;

    mSx = read<int32_t>(d);
    mSy = read<int32_t>(d);
    mR = read<float>(d);
    mMaxN = read<int32_t>(d);
    mMaxH = read<int32_t>(d);
    mMaxW = read<int32_t>(d);

    PLUGIN_VALIDATE(d == a + length);
}

IPluginV2DynamicExt* MergeTokensPlugin::clone() const noexcept
{
    try
    {
        auto p = new MergeTokensPlugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t MergeTokensPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType MergeTokensPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    DataType ret{};
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        ret = inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

DimsExprs MergeTokensPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};

    const auto h = inputs[2].d[2];
    const auto w = inputs[2].d[3];
    const auto hsy = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *h, *exprBuilder.constant(mSy));
    const auto wsx = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *w, *exprBuilder.constant(mSx));
    const auto s = exprBuilder.operation(DimensionOperation::kPROD, *w, *h);
    const auto num_dst = exprBuilder.operation(DimensionOperation::kPROD, *hsy, *wsx);
    const auto num_src = exprBuilder.operation(DimensionOperation::kSUB, *s, *num_dst);
    const auto wanted_merged_tokens_over_10 = exprBuilder.operation(DimensionOperation::kPROD, *s, *exprBuilder.constant(static_cast<int>(mR * 10)));
    const auto wanted_merged_tokens = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *wanted_merged_tokens_over_10, *exprBuilder.constant(10));
    const auto wanted_merged_tokens_over_8 = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *wanted_merged_tokens, *exprBuilder.constant(8));
    const auto wanted_merged_tokens_rounded = exprBuilder.operation(DimensionOperation::kPROD, *wanted_merged_tokens_over_8, *exprBuilder.constant(8));
    const auto merged_tokens = exprBuilder.operation(DimensionOperation::kMIN, *num_src, *wanted_merged_tokens_rounded);
    const auto unmerged_tokens = exprBuilder.operation(DimensionOperation::kSUB, *num_src, *merged_tokens);
    const auto total_tokens = exprBuilder.operation(DimensionOperation::kSUM, *unmerged_tokens, *num_dst);

    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 3);
        ret.nbDims = 3;
        ret.d[0] = inputs[0].d[0];
        ret.d[1] = total_tokens;
        ret.d[2] = inputs[0].d[2];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

bool MergeTokensPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_VALIDATE(0 <= pos && pos < 4);
    switch (pos) {
        case 0: // input
        case 3: // output
            return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 1: // unm idx
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        case 2: // before transpose nchw
            return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kHWC8;
    }
    return false;
}

void MergeTokensPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    mMaxN = in[2].max.d[0];
    mMaxH = in[2].max.d[2];
    mMaxW = in[2].max.d[3];
}

size_t MergeTokensPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    const int n = mMaxN;
    const int h = mMaxH;
    const int w = mMaxW;
    const int c = inputs[0].dims.d[2];
    const int hsy = h / mSy;
    const int wsx = w / mSx;
    const int num_dst = hsy * wsx;
    const int num_src = h*w - num_dst;
    const int ws_size = n * num_src * c * sizeof(uint16_t); // n * a_idx_len * c
    return ws_size;
}

template<typename T>
void load_from_numpy(T* data_h, int elts, const std::string &filename) {
  std::vector<npy::ndarray_len_t> shape;
  std::vector<T> data(elts);
  npy::LoadArrayFromNumpy(filename, shape, data);

  std::cout << "Shape: [";
  for ( int i = 0; i < shape.size() - 1; ++i ) {
    std::cout << shape[i] << "; ";
  }
  std::cout << shape[shape.size() - 1] << "]" << std::endl;

  std::memcpy(data_h, data.data(), elts*sizeof(T));
}

template void load_from_numpy(uint16_t* data_h, int elts, const std::string &filename);
template void load_from_numpy(int32_t* data_h, int elts, const std::string &filename);


int32_t MergeTokensPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int n = inputDesc[2].dims.d[0];
    const int c = inputDesc[2].dims.d[1];
    const int h = inputDesc[2].dims.d[2];
    const int w = inputDesc[2].dims.d[3];

    const int s = h * w;
    const int hsy = h / mSy;
    const int wsx = w / mSx;
    const int num_dst = hsy * wsx;
    const int num_src = s - num_dst;

    const int r = static_cast<int>(mR * 10.f);
    const int sr = s * r;
    int merged_tokens = std::min(num_src, static_cast<int>(floor(sr / 10)));
    merged_tokens = floor(merged_tokens / 8) * 8;
    const int nonmerged_tokens = s - merged_tokens;

    const int unm_idx_lengths = num_src - merged_tokens;
    const int src_idx_lengths = merged_tokens;
    const int dst_idx_lengths = merged_tokens;
    const int a_idx_lengths = num_src;
    const int b_idx_lengths = num_dst;

    ToMeParams params;
    params.x_ptr = reinterpret_cast<const uint16_t *>(inputs[0]);
    params.unm_idx_ptr = reinterpret_cast<const uint32_t *>(inputs[1]);
    params.merged_tokens_ptr = reinterpret_cast<uint16_t *>(outputs[0]);
    params.a_ptr = reinterpret_cast<uint16_t*>(workspace);

    params.n = n;
    params.h = h;
    params.w = w;
    params.c = c;
    params.r = mR;
    params.sx = mSx;
    params.sy = mSy;
    params.src_idx_lengths = src_idx_lengths;
    params.dst_idx_lengths = dst_idx_lengths;
    params.unm_idx_lengths = unm_idx_lengths;
    params.a_len = a_idx_lengths;
    params.b_len = b_idx_lengths;

    run_merge_tokens_kernel(params, stream);

    return 0;
}

void MergeTokensPlugin::destroy() noexcept
{
    delete this;
}

int32_t MergeTokensPlugin::initialize() noexcept
{
    return 0;
}

void MergeTokensPlugin::terminate() noexcept {}

size_t MergeTokensPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void MergeTokensPlugin::serialize(void* buffer) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(buffer != nullptr);
        auto* d = static_cast<char*>(buffer);
        auto* a = d;
        write(d, mSx); // int32_t
        write(d, mSy); // int32_t
        write(d, mR);  // float
        write(d, mMaxN);  // int32_t
        write(d, mMaxH);  // int32_t
        write(d, mMaxW);  // int32_t
        PLUGIN_VALIDATE(d == a + getSerializationSize());
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

void MergeTokensPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* MergeTokensPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* MergeTokensPlugin::getPluginType() const noexcept
{
    return kMERGE_TOKENS_PLUGIN_NAME.c_str();
}

char const* MergeTokensPlugin::getPluginVersion() const noexcept
{
    return kMERGE_TOKENS_PLUGIN_VERSION.c_str();
}

// class MergeTokensPluginCreator
PluginFieldCollection MergeTokensPluginCreator::mFC{};
std::vector<PluginField> MergeTokensPluginCreator::mPluginAttributes;

MergeTokensPluginCreator::MergeTokensPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sx", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sy", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("r", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

MergeTokensPluginCreator::~MergeTokensPluginCreator() {}

IPluginV2* MergeTokensPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        float r = 0.5;
        int32_t sx = 2;
        int32_t sy = 2;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            if (fc->fields[i].name == std::string("r"))
            {
                r = static_cast<float>(*(static_cast<float const*>((fc->fields[i].data))));
                continue;
            }
            if (fc->fields[i].name == std::string("sx"))
            {
                sx = static_cast<int32_t>(*(static_cast<int32_t const*>((fc->fields[i].data))));
                continue;
            }
            if (fc->fields[i].name == std::string("sy"))
            {
                sy = static_cast<int32_t>(*(static_cast<int32_t const*>((fc->fields[i].data))));
                continue;
            }
        }
        return new MergeTokensPlugin(name, sx, sy, r);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* MergeTokensPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new MergeTokensPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* MergeTokensPluginCreator::getPluginName() const noexcept
{
    return kMERGE_TOKENS_PLUGIN_NAME.c_str();
}

char const* MergeTokensPluginCreator::getPluginVersion() const noexcept
{
    return kMERGE_TOKENS_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* MergeTokensPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
