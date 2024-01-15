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

#include "unmergeTokensPlugin.h"
#include "unmergeTokensKernel.h"
#include "unmergeTokensPluginCommon.h"
#include "common/bertCommon.h"
#include "npy.hpp"

#include <cmath>

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::UnmergeTokensPlugin;
using nvinfer1::plugin::UnmergeTokensPluginCreator;

namespace
{
static std::string const kUNMERGE_TOKENS_PLUGIN_NAME{"UnmergeTokens"};
static std::string const kUNMERGE_TOKENS_PLUGIN_VERSION{"1"};
size_t constexpr kSERIALIZATION_SIZE{sizeof(float) + 5 * sizeof(int32_t)};
} // namespace

// class UnmergeTokensPlugin
UnmergeTokensPlugin::UnmergeTokensPlugin(std::string const& name, int32_t sx, int32_t sy, float r)
    : mName(name)
    , mSx(sx)
    , mSy(sy)
    , mR(r)
{
}

UnmergeTokensPlugin::UnmergeTokensPlugin(std::string const& name, void const* buffer, size_t length)
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

IPluginV2DynamicExt* UnmergeTokensPlugin::clone() const noexcept
{
    try
    {
        auto p = new UnmergeTokensPlugin(*this);
        p->setPluginNamespace(mNameSpace.c_str());
        return p;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

int32_t UnmergeTokensPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DataType UnmergeTokensPlugin::getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    DataType ret{};
    try
    {
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(nbInputs == 5);
        ret = inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

DimsExprs UnmergeTokensPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    DimsExprs ret{};

    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 5);
        ret.nbDims = 3;
        ret.d[0] = inputs[4].d[0];
        ret.d[1] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[4].d[2], *inputs[4].d[3]);
        ret.d[2] = inputs[4].d[1];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return ret;
}

bool UnmergeTokensPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    PLUGIN_VALIDATE(0 <= pos && pos < 6);
    switch (pos) {
        case 0: // input
        case 5: // output
            return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 1: // src idx
        case 2: // dst idx
        case 3: // unm idx
            return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        case 4: // before transpose nchw
            return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kHWC8;
    }
    return false;
}

void UnmergeTokensPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    mMaxN = in[4].max.d[0];
    mMaxH = in[4].max.d[2];
    mMaxW = in[4].max.d[3];
}

size_t UnmergeTokensPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    const int h = mMaxH;
    const int w = mMaxW;

    const int s = h * w;
    const int hsy = h / mSy;
    const int wsx = w / mSx;
    const int num_dst = hsy * wsx;
    const int num_src = s - num_dst;

    const int r = static_cast<int>(mR * 10);
    const int sr = s * r;
    int merged_tokens = std::min(num_src, static_cast<int>(floor(sr / 10)));
    merged_tokens = floor(merged_tokens / 8) * 8;
    const int c = inputs[4].dims.d[1];
    const int ws_size = mMaxN * merged_tokens * c * sizeof(uint16_t); // n * src_idx_len * c
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


int32_t UnmergeTokensPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int n = inputDesc[4].dims.d[0];
    const int h = inputDesc[4].dims.d[2];
    const int w = inputDesc[4].dims.d[3];
    const int c = inputDesc[4].dims.d[1];

    const int s = h * w;
    const int hsy = h / mSy;
    const int wsx = w / mSx;
    const int num_dst = hsy * wsx;
    const int num_src = s - num_dst;

    const int r = static_cast<int>(mR * 10);
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
    params.src_idx_ptr = reinterpret_cast<const uint32_t *>(inputs[1]);
    params.dst_idx_ptr = reinterpret_cast<const uint32_t *>(inputs[2]);
    params.unm_idx_ptr = reinterpret_cast<const uint32_t *>(inputs[3]);

    params.unmerged_tokens_ptr = reinterpret_cast<uint16_t *>(outputs[0]);
    
    params.src_ptr = reinterpret_cast<uint16_t*>(workspace);
    
    params.n = n;
    params.h = h;
    params.w = w;
    params.c = c;
    params.r = mR;
    params.sx = mSx;
    params.sy = mSy;
    params.wsx = wsx;
    params.src_idx_lengths = src_idx_lengths;
    params.dst_idx_lengths = dst_idx_lengths;
    params.unm_idx_lengths = unm_idx_lengths;
    params.a_len = a_idx_lengths;
    params.b_len = b_idx_lengths;

    run_unmerge_tokens_kernel(params, stream);

    return 0;
}

void UnmergeTokensPlugin::destroy() noexcept
{
    delete this;
}

int32_t UnmergeTokensPlugin::initialize() noexcept
{
    return 0;
}

void UnmergeTokensPlugin::terminate() noexcept {}

size_t UnmergeTokensPlugin::getSerializationSize() const noexcept
{
    return kSERIALIZATION_SIZE;
}

void UnmergeTokensPlugin::serialize(void* buffer) const noexcept
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

void UnmergeTokensPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNameSpace = pluginNamespace;
}

char const* UnmergeTokensPlugin::getPluginNamespace() const noexcept
{
    return mNameSpace.c_str();
}

char const* UnmergeTokensPlugin::getPluginType() const noexcept
{
    return kUNMERGE_TOKENS_PLUGIN_NAME.c_str();
}

char const* UnmergeTokensPlugin::getPluginVersion() const noexcept
{
    return kUNMERGE_TOKENS_PLUGIN_VERSION.c_str();
}

// class UnmergeTokensPluginCreator
PluginFieldCollection UnmergeTokensPluginCreator::mFC{};
std::vector<PluginField> UnmergeTokensPluginCreator::mPluginAttributes;

UnmergeTokensPluginCreator::UnmergeTokensPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sx", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sy", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("r", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

UnmergeTokensPluginCreator::~UnmergeTokensPluginCreator() {}

IPluginV2* UnmergeTokensPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
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
        return new UnmergeTokensPlugin(name, sx, sy, r);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* UnmergeTokensPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new UnmergeTokensPlugin(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* UnmergeTokensPluginCreator::getPluginName() const noexcept
{
    return kUNMERGE_TOKENS_PLUGIN_NAME.c_str();
}

char const* UnmergeTokensPluginCreator::getPluginVersion() const noexcept
{
    return kUNMERGE_TOKENS_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* UnmergeTokensPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}
