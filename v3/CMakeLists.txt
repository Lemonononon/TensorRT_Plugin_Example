

cuda_add_library(mypluginsV3 SHARED Add.cu AddPluginV3.cpp
)

# test
add_executable(testV3 main.cpp AddPluginV3.cpp)
TARGET_LINK_LIBRARIES(testV3 nvinfer)
TARGET_LINK_LIBRARIES(testV3 cudart)
TARGET_LINK_LIBRARIES(testV3 mypluginsV3)
TARGET_LINK_LIBRARIES(testV3 nvinfer_plugin)
TARGET_LINK_LIBRARIES(testV3 nvonnxparser)