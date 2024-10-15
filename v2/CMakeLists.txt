

cuda_add_library(mypluginsV2 SHARED Add.cu AddPluginV2.cpp
)

# test
add_executable(testV2 main.cpp AddPluginV2.cpp)
TARGET_LINK_LIBRARIES(testV2 nvinfer)
TARGET_LINK_LIBRARIES(testV2 cudart)
TARGET_LINK_LIBRARIES(testV2 mypluginsV2)
TARGET_LINK_LIBRARIES(testV2 nvinfer_plugin)
TARGET_LINK_LIBRARIES(testV2 nvonnxparser)