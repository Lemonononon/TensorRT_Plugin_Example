#include <iostream>
#include <cassert>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <vector>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

using namespace nvinfer1;

void testAddPlugin() {
    
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    
    initLibNvInferPlugins(&gLogger, "");

    // 创建输入张量
    auto input0 = network->addInput("input0", DataType::kFLOAT, Dims{1, {1}});
    auto input1 = network->addInput("input1", DataType::kFLOAT, Dims{1, {1}});

    std::vector<ITensor*> inputs = {input0, input1};


    auto registry = getPluginRegistry();
    int32_t num;
    auto creators = registry->getPluginCreatorList(&num);

    std::cout << "TensorRT Plugin: num: " << num << std::endl;
    std::cout << "==================================================" << std::endl;

    // for (int i = 0; i < num; ++i) {
    //     auto creator = creators[i];
    //     std::cout << "插件名称: " << creator->getPluginName() << std::endl;
    //     std::cout << "插件版本: " << creator->getPluginVersion() << std::endl;
    //     std::cout << "命名空间: " << creator->getPluginNamespace() << std::endl;
    //     std::cout << "--------------------------------------------------" << std::endl;
    // }



    // 创建 AddPlugin
    auto creator = getPluginRegistry()->getPluginCreator("TestAddPlugin", "1");


    int num2 = 3;
    PluginField pluginField[1] = { {"flag", &num2, PluginFieldType::kINT32, 1} };

    PluginFieldCollection pluginData;
    pluginData.nbFields = 1;
    pluginData.fields = pluginField;
    auto plugin = creator->createPlugin("TestAddPlugin", &pluginData);

    auto addLayer = network->addPluginV2(&inputs[0], 2, *plugin);
    addLayer->getOutput(0)->setName("output");
    network->markOutput(*addLayer->getOutput(0));

    
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20); // 1MB
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    

    // network->destroy();
    // builder->destroy();
    // config->destroy();
    // plugin->destroy();

    // 创建执行上下文
    IExecutionContext* context = engine->createExecutionContext();

    // 准备输入数据
    float h_input0 = 5.0f; // 第一个标量
    float h_input1 = 3.0f; // 第二个标量
    float h_output = 0.0f; // 输出

    // 分配设备内存
    float *d_input0, *d_input1, *d_output;
    cudaMalloc(&d_input0, sizeof(h_input0));
    cudaMalloc(&d_input1, sizeof(h_input1));
    cudaMalloc(&d_output, sizeof(h_output));

    // 将输入数据复制到设备
    cudaMemcpy(d_input0, &h_input0, sizeof(h_input0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1, &h_input1, sizeof(h_input1), cudaMemcpyHostToDevice);

    // 执行推理
    void* buffers[] = {d_input0, d_input1, d_output};
    context->executeV2(buffers);

    // 将输出数据复制回主机
    cudaMemcpy(&h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost);

    // 打印输出结果
    std::cout << "Output: " << h_output << std::endl; // 应该输出 8.0

    // 清理
    cudaFree(d_input0);
    cudaFree(d_input1);
    cudaFree(d_output);
    context->destroy();
    engine->destroy();
}

int main() {
    testAddPlugin();
    return 0;
}