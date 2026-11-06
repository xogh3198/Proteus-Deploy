#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

namespace {
thread_local std::string last_error;

template <typename F, typename T>
T checked(F operation, T failure) noexcept {
  try {
    return operation();
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "Unknown native inference error";
  }
  return failure;
}

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

Ort::Session open_model(const char* path, int intra_threads, int inter_threads) {
  static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "proteus-ort181");
  check_cuda(cudaSetDevice(0));
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(intra_threads);
  options.SetInterOpNumThreads(inter_threads);
  options.SetExecutionMode(ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  OrtCUDAProviderOptions cuda{};
  cuda.cudnn_conv_algo_search = EXHAUSTIVE;
  cuda.gpu_mem_limit = std::numeric_limits<size_t>::max();
  cuda.do_copy_in_default_stream = 1;
  options.AppendExecutionProvider_CUDA(cuda);
  return Ort::Session(environment, path, options);
}

struct Session {
  Session(const char* path, int intra, int inter) : ort(open_model(path, intra, inter)) {
    if (ort.GetInputCount() != 1 || ort.GetOutputCount() != 1) {
      throw std::runtime_error("Expected a single ResNet input and output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    const auto free_name = [&allocator](char* name) { allocator.Free(name); };
    std::unique_ptr<char, decltype(free_name)> input(ort.GetInputName(0, allocator), free_name);
    std::unique_ptr<char, decltype(free_name)> output(ort.GetOutputName(0, allocator), free_name);
    input_name = input.get();
    output_name = output.get();
  }

  Ort::Session ort;
  std::string input_name;
  std::string output_name;
};
}  // namespace

extern "C" {
const char* proteus_ort181_error() { return last_error.c_str(); }
const char* proteus_ort181_version() { return OrtGetApiBase()->GetVersionString(); }

int proteus_ort181_cuda_version(int* version) {
  return checked([&] { check_cuda(cudaRuntimeGetVersion(version)); return 0; }, -1);
}

int proteus_ort181_device(char* name, size_t length) {
  return checked([&] {
    check_cuda(cudaSetDevice(0));
    cudaDeviceProp props;
    check_cuda(cudaGetDeviceProperties(&props, 0));
    if (std::strlen(props.name) >= length) {
      throw std::runtime_error("GPU name buffer is too small");
    }
    std::strcpy(name, props.name);
    return 0;
  }, -1);
}

int proteus_ort181_synchronize() {
  return checked([] { check_cuda(cudaDeviceSynchronize()); return 0; }, -1);
}

int proteus_ort181_memory(size_t* free_bytes, size_t* total_bytes) {
  return checked([&] { check_cuda(cudaMemGetInfo(free_bytes, total_bytes)); return 0; }, -1);
}

void* proteus_ort181_open(const char* path, int intra, int inter) {
  return checked([&] { return static_cast<void*>(new Session(path, intra, inter)); },
                 static_cast<void*>(nullptr));
}

void proteus_ort181_close(void* handle) { delete static_cast<Session*>(handle); }

int proteus_ort181_run(void* handle, float* input, int64_t batch, float* output) {
  return checked([&] {
    auto& session = *static_cast<Session*>(handle);
    const std::array<int64_t, 4> dims{batch, 3, 224, 224};
    const auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const Ort::MemoryInfo gpu("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        cpu, input, static_cast<size_t>(batch) * 3 * 224 * 224, dims.data(), dims.size());
    Ort::IoBinding binding(session.ort);
    binding.BindInput(session.input_name.c_str(), tensor);
    binding.BindOutput(session.output_name.c_str(), gpu);
    session.ort.Run(Ort::RunOptions{nullptr}, binding);
    auto results = binding.GetOutputValues();
    if (results.size() != 1) {
      throw std::runtime_error("Expected one bound ResNet output");
    }
    const auto info = results.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetShape() != std::vector<int64_t>{batch, 1000}) {
      throw std::runtime_error("Expected FP32 ImageNet logits with shape [batch, 1000]");
    }
    check_cuda(cudaMemcpy(output, results.front().GetTensorMutableData<float>(),
                          static_cast<size_t>(batch) * 1000 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
  }, -1);
}
}  // extern "C"
