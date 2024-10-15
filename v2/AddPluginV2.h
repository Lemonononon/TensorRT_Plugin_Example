#include <NvInfer.h>
#include <vector>
#include <string>
#include <cuda_fp16.h>

namespace nvinfer1{
namespace plugin{

template <typename T> void AddForward(const T* input0, const T* input1, T* output, int numElements, cudaStream_t stream);


class AddPlugin : public nvinfer1::IPluginV2DynamicExt{
public:
    AddPlugin( int flag);
    AddPlugin(const void* data, size_t length);
    ~AddPlugin();

    int                     getNbOutputs() const noexcept override;
    nvinfer1::DimsExprs     getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& builder) noexcept override;
    int                     initialize() noexcept override;
    void                    terminate() noexcept override;
    size_t                  getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int                     enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t                  getSerializationSize() const noexcept override;
    void                    serialize(void* buffer) const noexcept override;
    bool                    supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    const char*             getPluginType() const noexcept override;
    const char*             getPluginVersion() const noexcept override;
    IPluginV2DynamicExt*    clone() const noexcept override;
    void                    destroy() noexcept override;
    nvinfer1::DataType      getOutputDataType(int outputIndex, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
    void                    detachFromContext() noexcept override;
    void                    setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char*             getPluginNamespace() const noexcept override;
    void                    configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    
private:
    const char* mNamespace;
    int flag;
};

class AddPluginCreator : public nvinfer1::IPluginCreator{
public:
    AddPluginCreator();
    ~AddPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

}
}