#include "AddPluginV3.h"
#include <cassert>
#include <cstring>
#include <iostream>

// a trt plugin that add two tensors element-wise
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
    const char* ADD_PLUGIN_VERSION{"1"};
    const char* ADD_PLUGIN_NAME{"TestAddPlugin"};
} // namespace


PluginFieldCollection AddPluginCreator::mFC{};
std::vector<PluginField> AddPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AddPluginCreator);

template <typename scalar_t>
scalar_t readFromBuffer(const char*& buffer)
{
    scalar_t val = *reinterpret_cast<const scalar_t*>(buffer);
    buffer += sizeof(scalar_t);
    return val;
}

template <typename scalar_t>
void writeToBuffer(char*& buffer, const scalar_t& val)
{
    *reinterpret_cast<scalar_t*>(buffer) = val;
    buffer += sizeof(scalar_t);
}
AddPlugin::AddPlugin( int flag ) {
    this->flag = flag;
}

AddPlugin::~AddPlugin() {
}

AddPlugin::AddPlugin(const void* data, size_t length) {
    const char* buffer = static_cast<const char*>(data);
    flag = readFromBuffer<int>(buffer);
}

int AddPlugin::getNbOutputs() const noexcept {
    return 1;
}

nvinfer1::DimsExprs AddPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& builder) noexcept {
    
    assert(nbInputs == 2);
    assert(outputIndex == 0);
    
    return inputs[0];
}

int AddPlugin::initialize() noexcept {
    return 0;
}

void AddPlugin::terminate() noexcept {

}

size_t AddPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {   
    return 0;
}

int AddPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    const float* input0 = static_cast<const float*>(inputs[0]);
    const float* input1 = static_cast<const float*>(inputs[1]);
    float* output = static_cast<float*>(outputs[0]);

    int numElements = 1;

    cudaMemcpyAsync(output, input0, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(output + numElements, input1, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // float or half
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        std::cout << "Using float precision" << std::endl;
        AddForward<float>(input0, input1, output, numElements, stream);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        std::cout << "Using half precision" << std::endl;
        AddForward<half>( reinterpret_cast<const half*>(input0), reinterpret_cast<const half*>(input1), 
        reinterpret_cast<half*>(output), numElements, stream);
    }

    return 0;
}

size_t AddPlugin::getSerializationSize() const noexcept {
    return 0;
}

void AddPlugin::serialize(void* buffer) const noexcept {
}

bool AddPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    return true;
}

const char* AddPlugin::getPluginType() const noexcept {
    return ADD_PLUGIN_NAME;
}

const char* AddPlugin::getPluginVersion() const noexcept {
    return ADD_PLUGIN_VERSION;
}

IPluginV2DynamicExt* AddPlugin::clone() const noexcept {
    return new AddPlugin(flag);
}

void AddPlugin::destroy() noexcept {
    delete this;
}

nvinfer1::DataType AddPlugin::getOutputDataType(int outputIndex, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

void AddPlugin::detachFromContext() noexcept {

}

void AddPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* AddPlugin::getPluginNamespace() const noexcept {
    return mNamespace;
}

void AddPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {

}

AddPluginCreator::AddPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

AddPluginCreator::~AddPluginCreator() {
    
}

IPluginV2DynamicExt* AddPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
    
    int flag = 0;

    for (int i = 0; i < fc->nbFields; i++) {

        if ( strcmp(fc->fields[i].name, "flag") == 0 ){
            flag = *static_cast<const int*>(fc->fields[i].data);
        }

    }

    return new AddPlugin(flag);
}

IPluginV2DynamicExt* AddPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    auto plugin =  new AddPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

const char* AddPluginCreator::getPluginName() const noexcept {
    return ADD_PLUGIN_NAME;
}

const char* AddPluginCreator::getPluginVersion() const noexcept {
    return ADD_PLUGIN_VERSION;
}

const PluginFieldCollection* AddPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

void AddPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* AddPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}