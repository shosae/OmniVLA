#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
class Logger final : public nvinfer1::ILogger { public: void log(Severity s, char const* m) noexcept override { if (s <= Severity::kWARNING) std::cerr << "[TensorRT] " << m << '\n'; } };
void cuda_ok(cudaError_t s, char const* w) { if (s != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(s)); }
void trt_ok(bool s, char const* w) { if (!s) throw std::runtime_error(std::string(w) + " failed"); }
size_t size_of(nvinfer1::DataType t) { switch (t) { case nvinfer1::DataType::kFLOAT: return 4; case nvinfer1::DataType::kHALF: return 2; case nvinfer1::DataType::kINT32: return 4; case nvinfer1::DataType::kINT64: return 8; default: throw std::runtime_error("unsupported TensorRT data type"); } }
size_t volume(nvinfer1::Dims const& d) { size_t n = 1; for (int i = 0; i < d.nbDims; ++i) { if (d.d[i] < 0) throw std::runtime_error("unresolved shape"); n *= d.d[i]; } return n; }
float gpu_used_mib() { size_t free_bytes{}, total_bytes{}; cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo"); return float(total_bytes - free_bytes) / float(1 << 20); }
nvinfer1::Dims shape(std::initializer_list<int64_t> values) { nvinfer1::Dims d{}; d.nbDims = values.size(); int i = 0; for (auto v : values) d.d[i++] = v; return d; }
std::vector<char> read_file(fs::path const& p) { std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) throw std::runtime_error("cannot read " + p.string()); auto n = size_t(f.tellg()); std::vector<char> x(n); f.seekg(0); f.read(x.data(), n); if (!f) throw std::runtime_error("short read " + p.string()); return x; }
struct Device { void* p{}; size_t bytes; explicit Device(size_t n) : bytes(n) { cuda_ok(cudaMalloc(&p, n ? n : 1), "cudaMalloc"); } ~Device() { if (p) cudaFree(p); } Device(Device const&) = delete; };

int main(int argc, char** argv) {
    if (argc != 7) { std::cerr << "usage: " << argv[0] << " <engine> <plugin> <input_dir> <output_dir> <sequence_length> <kv_cache_capacity>\n"; return 1; }
    try {
        fs::path engine_path(argv[1]), plugin_path(argv[2]), input(argv[3]), output(argv[4]); fs::create_directories(output);
        int seq = std::stoi(argv[5]), capacity = std::stoi(argv[6]); if (seq <= 0 || capacity < seq) throw std::runtime_error("invalid sequence length or KV cache capacity");
        Logger logger; void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL); if (!handle) throw std::runtime_error(dlerror());
        auto init = reinterpret_cast<bool (*)(void*, char const*)>(dlsym(handle, "initEdgellmPlugins")); if (!init || !init(&logger, "")) throw std::runtime_error("initEdgellmPlugins failed");
        float const gpu_used_mib_start = gpu_used_mib();
        auto bytes = read_file(engine_path); std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger)); std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size())); if (!engine) throw std::runtime_error("engine deserialization failed"); std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext()); if (!ctx) throw std::runtime_error("context creation failed");
        cudaStream_t stream{}; cuda_ok(cudaStreamCreate(&stream), "stream"); trt_ok(ctx->setOptimizationProfileAsync(0, stream), "set profile");
        constexpr int batch=1, hidden=4096, rope_dim=128, layers=32, kv_heads=32, head_dim=128;
        trt_ok(ctx->setInputShape("inputs_embeds", shape({batch,seq,hidden})), "inputs shape"); trt_ok(ctx->setInputShape("rope_rotary_cos_sin", shape({batch,capacity,rope_dim})), "rope shape"); trt_ok(ctx->setInputShape("context_lengths", shape({batch})), "context shape"); trt_ok(ctx->setInputShape("last_token_ids", shape({batch,1})), "last token shape"); trt_ok(ctx->setInputShape("kvcache_start_index", shape({0})), "KV start shape");
        for (int i=0;i<layers;++i) trt_ok(ctx->setInputShape(("past_key_values_"+std::to_string(i)).c_str(), shape({batch,2,kv_heads,capacity,head_dim})), "KV shape");
        Device embeds(size_t(batch)*seq*hidden*2), rope(size_t(batch)*capacity*rope_dim*4), context(4), last(8), dummy(1); auto e=read_file(input/"inputs_embeds_fp16.bin"), r=read_file(input/"rope_rotary_cos_sin_fp32.bin"); if(e.size()!=embeds.bytes || r.size()!=rope.bytes) throw std::runtime_error("unexpected input size"); cuda_ok(cudaMemcpyAsync(embeds.p,e.data(),e.size(),cudaMemcpyHostToDevice,stream),"copy embeds"); cuda_ok(cudaMemcpyAsync(rope.p,r.data(),r.size(),cudaMemcpyHostToDevice,stream),"copy rope"); int32_t context_length=seq; int64_t last_token=seq-1; cuda_ok(cudaMemcpyAsync(context.p,&context_length,4,cudaMemcpyHostToDevice,stream),"copy context"); cuda_ok(cudaMemcpyAsync(last.p,&last_token,8,cudaMemcpyHostToDevice,stream),"copy last"); cuda_ok(cudaMemsetAsync(dummy.p,0,1,stream),"zero dummy");
        trt_ok(ctx->setTensorAddress("inputs_embeds",embeds.p),"bind embeds"); trt_ok(ctx->setTensorAddress("rope_rotary_cos_sin",rope.p),"bind rope"); trt_ok(ctx->setTensorAddress("context_lengths",context.p),"bind context"); trt_ok(ctx->setTensorAddress("last_token_ids",last.p),"bind last"); trt_ok(ctx->setTensorAddress("kvcache_start_index",dummy.p),"bind KV start");
        std::unordered_map<std::string,std::unique_ptr<Device>> out; for(int i=0;i<layers;++i) { auto past="past_key_values_"+std::to_string(i), present="present_key_values_"+std::to_string(i); auto d=ctx->getTensorShape(present.c_str()); auto x=std::make_unique<Device>(volume(d)*size_of(engine->getTensorDataType(present.c_str()))); cuda_ok(cudaMemsetAsync(x->p,0,x->bytes,stream),"zero KV"); trt_ok(ctx->setTensorAddress(past.c_str(),x->p),"bind past"); trt_ok(ctx->setTensorAddress(present.c_str(),x->p),"bind present"); out.emplace(present,std::move(x)); }
        for (auto const& name : {std::string("logits"),std::string("hidden_states")}) { auto d=ctx->getTensorShape(name.c_str()); auto x=std::make_unique<Device>(volume(d)*size_of(engine->getTensorDataType(name.c_str()))); trt_ok(ctx->setTensorAddress(name.c_str(),x->p),"bind output"); out.emplace(name,std::move(x)); }
        cuda_ok(cudaStreamSynchronize(stream),"sync inputs"); float const gpu_used_mib_before_enqueue = gpu_used_mib(); cudaEvent_t start{}, end{}; cuda_ok(cudaEventCreate(&start),"event"); cuda_ok(cudaEventCreate(&end),"event"); cuda_ok(cudaEventRecord(start,stream),"start event"); trt_ok(ctx->enqueueV3(stream),"enqueue"); cuda_ok(cudaEventRecord(end,stream),"end event"); std::vector<char> hidden_state(out.at("hidden_states")->bytes); cuda_ok(cudaMemcpyAsync(hidden_state.data(),out.at("hidden_states")->p,hidden_state.size(),cudaMemcpyDeviceToHost,stream),"copy hidden"); cuda_ok(cudaStreamSynchronize(stream),"sync output"); std::ofstream f(output/"hidden_states_fp16.bin",std::ios::binary); f.write(hidden_state.data(),hidden_state.size()); float ms{}; cuda_ok(cudaEventElapsedTime(&ms,start,end),"elapsed"); std::cout << "engine_enqueue_ms=" << ms << '\n'; std::cout << "engine_cuda_used_mib_start=" << gpu_used_mib_start << '\n'; std::cout << "engine_cuda_used_mib_before_enqueue=" << gpu_used_mib_before_enqueue << '\n'; std::cout << "engine_cuda_used_mib_after_enqueue=" << gpu_used_mib() << '\n'; cudaEventDestroy(start); cudaEventDestroy(end); cudaStreamDestroy(stream); dlclose(handle); return 0;
    } catch (std::exception const& e) { std::cerr << "engine inference failed: " << e.what() << '\n'; return 1; }
}
