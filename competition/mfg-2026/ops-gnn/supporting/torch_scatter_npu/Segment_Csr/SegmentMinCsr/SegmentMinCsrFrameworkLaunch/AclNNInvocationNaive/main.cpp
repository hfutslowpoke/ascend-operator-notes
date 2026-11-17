#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_segment_min_csr.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        ERROR_LOG("Failed to open file: %s", filePath.c_str());
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    fileSize = static_cast<size_t>(size);

    if (fileSize > bufferSize) {
        ERROR_LOG("File size (%zu) is larger than buffer size (%zu).", fileSize, bufferSize);
        file.close();
        return false;
    }

    if (!file.read(static_cast<char *>(buffer), size)) {
        ERROR_LOG("Failed to read the entire file: %s", filePath.c_str());
        file.close();
        return false;
    }

    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed, buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Failed to open file: %s", filePath.c_str());
        return false;
    }

    const char *data = static_cast<const char *>(buffer);
    size_t bytesWritten = 0;
    const size_t chunkSize = 1024 * 1024 * 1024;

    while (bytesWritten < size) {
        size_t bytesToWrite = std::min(chunkSize, size - bytesWritten);
        ssize_t writeResult = write(fd, data + bytesWritten, bytesToWrite);
        if (writeResult < 0) {
            ERROR_LOG("Write file failed with error.");
            close(fd);
            return false;
        }
        bytesWritten += static_cast<size_t>(writeResult);
        if (static_cast<size_t>(writeResult) != bytesToWrite) {
            ERROR_LOG("Partial write occurred. Wrote %zu of %zu bytes.",
                      static_cast<size_t>(writeResult), bytesToWrite);
            break;
        }
    }

    close(fd);
    if (bytesWritten != size) {
        ERROR_LOG("Write file size mismatch. Expected %zu, but wrote %zu.", size, bytesWritten);
        return false;
    }
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return FAILED);

    return SUCCESS;
}

template <typename T>
int CreateInputTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
                      void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);

    return SUCCESS;
}

int CreateOutputTensor(const std::vector<int64_t> &shape, size_t type_size,
                       void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * type_size;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return FAILED);

    return SUCCESS;
}

int run_case_min_csr_fp32(int32_t deviceId, const std::string& caseName,
                          const std::string& inputDir, const std::string& outputDir)
{
    auto ret = 0;

    aclrtStream stream;
    ret = Init(deviceId, &stream);
    CHECK_RET(ret == SUCCESS, return FAILED);

    void* srcDeviceAddr = nullptr;
    aclTensor* srcTensor = nullptr;
    void* indptrDeviceAddr = nullptr;
    aclTensor* indptrTensor = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* outTensor = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;

    {
        std::string finalDataPath = inputDir + "/src_" + caseName + ".bin";
        std::vector<int64_t> srcShape = {8, 2048};
        size_t srcSize = GetShapeSize(srcShape);
        std::vector<float> srcHostData(srcSize);
        size_t srcFileSize = 0;
        ReadFile(finalDataPath, srcFileSize, srcHostData.data(), srcSize * sizeof(float));
        ret = CreateInputTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &srcTensor);
        CHECK_RET(ret == SUCCESS, return FAILED);
    }

    {
        std::string finalDataPath = inputDir + "/indptr_" + caseName + ".bin";
        std::vector<int64_t> indptrShape = {5};
        size_t indptrSize = GetShapeSize(indptrShape);
        std::vector<int32_t> indptrHostData(indptrSize);
        size_t indptrFileSize = 0;
        ReadFile(finalDataPath, indptrFileSize, indptrHostData.data(), indptrSize * sizeof(int32_t));
        ret = CreateInputTensor(indptrHostData, indptrShape, &indptrDeviceAddr, aclDataType::ACL_INT32, &indptrTensor);
        CHECK_RET(ret == SUCCESS, return FAILED);
    }

    INFO_LOG("Input preparation success.");

    {
        std::vector<int64_t> outShape = {4, 2048};
        ret = CreateOutputTensor(outShape, sizeof(float), &outDeviceAddr, aclDataType::ACL_FLOAT, &outTensor);
        CHECK_RET(ret == SUCCESS, return FAILED);
    }

    INFO_LOG("Output preparation success.");

    aclOpExecutor* executor;
    ret = aclnnSegmentMinCsrGetWorkspaceSize(srcTensor, indptrTensor, nullptr,
                                              outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("GetWorkspaceSize failed.\n"); return FAILED);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc workspace failed.\n"); return FAILED);
    }

    ret = aclnnSegmentMinCsr(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Execution failed.\n"); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("Synchronize stream failed.\n"); return FAILED);

    {
        std::string finalOutputPath = outputDir + "/output_y_" + caseName + ".bin";
        std::vector<int64_t> outShape = {4, 2048};
        size_t outSize = GetShapeSize(outShape);
        std::vector<float> outResultData(outSize);
        ret = aclrtMemcpy(outResultData.data(), outSize * sizeof(float), outDeviceAddr,
                          outSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == SUCCESS, LOG_PRINT("Copy result failed.\n"); return FAILED);
        WriteFile(finalOutputPath, outResultData.data(), outSize * sizeof(float));
    }

    INFO_LOG("Write output success.");

    if (srcTensor) aclDestroyTensor(srcTensor);
    if (srcDeviceAddr) aclrtFree(srcDeviceAddr);
    if (indptrTensor) aclDestroyTensor(indptrTensor);
    if (indptrDeviceAddr) aclrtFree(indptrDeviceAddr);
    if (outTensor) aclDestroyTensor(outTensor);
    if (outDeviceAddr) aclrtFree(outDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    int32_t deviceId = 0;
    std::string caseName = "single_case";
    std::string inputDir = "./input";
    std::string outputDir = "./output";

    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }
    if (argc > 2) {
        caseName = argv[2];
    }
    if (argc > 3) {
        inputDir = argv[3];
    }
    if (argc > 4) {
        outputDir = argv[4];
    }

    INFO_LOG("Running SegmentMinCsr test case: %s", caseName.c_str());
    auto ret = run_case_min_csr_fp32(deviceId, caseName, inputDir, outputDir);

    if (ret == SUCCESS) {
        INFO_LOG("Successfully completed SegmentMinCsr test case!");
    } else {
        ERROR_LOG("Failed to complete SegmentMinCsr test case!");
    }

    return ret;
}
