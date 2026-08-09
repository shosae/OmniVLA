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
size_t volume(nvinfer1::Dims const& d) { size_t n=1; for (int i=0;i<d.nbDims;++i) { if (d.d[i]<0) throw std::runtime_error("unresolved shape"); n*=d.d[i]; } return n; }
float gpu_used_mib() { size_t free_bytes{}, total_bytes{}; cuda_ok(cudaMemGetInfo(&free_bytes,&total_bytes),"cudaMemGetInfo"); return float(total_bytes-free_bytes)/float(1<<20); }
nvinfer1::Dims shape(std::initializer_list<int64_t> x) { nvinfer1::Dims d{}; d.nbDims=x.size(); int i=0; for(auto v:x)d.d[i++]=v; return d; }
std::vector<char> read_file(fs::path const& p) { std::ifstream f(p,std::ios::binary|std::ios::ate); if(!f)throw std::runtime_error("cannot read "+p.string()); auto n=size_t(f.tellg()); std::vector<char>x(n); f.seekg(0); f.read(x.data(),n); if(!f)throw std::runtime_error("short read "+p.string()); return x; }
struct Device { void* p{}; size_t bytes; explicit Device(size_t n):bytes(n){cuda_ok(cudaMalloc(&p,n?n:1),"cudaMalloc");} ~Device(){if(p)cudaFree(p);} Device(Device const&)=delete; };

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <engine> <plugin>\n"; return 1; }
    try {
        Logger logger; void* plugin=dlopen(argv[2],RTLD_NOW|RTLD_GLOBAL); if(!plugin)throw std::runtime_error(dlerror());
        auto init=reinterpret_cast<bool(*)(void*,char const*)>(dlsym(plugin,"initEdgellmPlugins")); if(!init||!init(&logger,""))throw std::runtime_error("plugin init failed");
        auto bytes=read_file(argv[1]); std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger)); std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(),bytes.size())); if(!engine)throw std::runtime_error("engine deserialization failed"); std::unique_ptr<nvinfer1::IExecutionContext> ctx(engine->createExecutionContext()); if(!ctx)throw std::runtime_error("context creation failed");
        cudaStream_t stream{}; cuda_ok(cudaStreamCreate(&stream),"stream"); trt_ok(ctx->setOptimizationProfileAsync(0,stream),"set profile"); std::cout << "READY\n" << std::flush;
        std::string input_dir, output_file; int seq=0, capacity=0;
        while (std::cin >> input_dir >> output_file >> seq >> capacity) {
            try {
                if(seq<=0||capacity<seq)throw std::runtime_error("invalid sequence length or KV capacity"); constexpr int B=1,H=4096,R=128,L=32,KH=32,HD=128;
                trt_ok(ctx->setInputShape("inputs_embeds",shape({B,seq,H})),"inputs shape"); trt_ok(ctx->setInputShape("rope_rotary_cos_sin",shape({B,capacity,R})),"rope shape"); trt_ok(ctx->setInputShape("context_lengths",shape({B})),"context shape"); trt_ok(ctx->setInputShape("last_token_ids",shape({B,1})),"last token shape"); trt_ok(ctx->setInputShape("kvcache_start_index",shape({0})),"KV start shape");
                for(int i=0;i<L;++i)trt_ok(ctx->setInputShape(("past_key_values_"+std::to_string(i)).c_str(),shape({B,2,KH,capacity,HD})),"KV shape");
                Device embeds(size_t(B)*seq*H*2), rope(size_t(B)*capacity*R*4), context(4), last(8), dummy(1); auto e=read_file(fs::path(input_dir)/"inputs_embeds_fp16.bin"), r=read_file(fs::path(input_dir)/"rope_rotary_cos_sin_fp32.bin"); if(e.size()!=embeds.bytes||r.size()!=rope.bytes)throw std::runtime_error("unexpected input size"); cuda_ok(cudaMemcpyAsync(embeds.p,e.data(),e.size(),cudaMemcpyHostToDevice,stream),"copy embeds"); cuda_ok(cudaMemcpyAsync(rope.p,r.data(),r.size(),cudaMemcpyHostToDevice,stream),"copy rope"); int32_t length=seq; int64_t last_token=seq-1; cuda_ok(cudaMemcpyAsync(context.p,&length,4,cudaMemcpyHostToDevice,stream),"copy context"); cuda_ok(cudaMemcpyAsync(last.p,&last_token,8,cudaMemcpyHostToDevice,stream),"copy last"); cuda_ok(cudaMemsetAsync(dummy.p,0,1,stream),"zero dummy");
                trt_ok(ctx->setTensorAddress("inputs_embeds",embeds.p),"bind embeds"); trt_ok(ctx->setTensorAddress("rope_rotary_cos_sin",rope.p),"bind rope"); trt_ok(ctx->setTensorAddress("context_lengths",context.p),"bind context"); trt_ok(ctx->setTensorAddress("last_token_ids",last.p),"bind last"); trt_ok(ctx->setTensorAddress("kvcache_start_index",dummy.p),"bind KV start"); std::unordered_map<std::string,std::unique_ptr<Device>> out;
                for(int i=0;i<L;++i){auto past="past_key_values_"+std::to_string(i), present="present_key_values_"+std::to_string(i); auto b=std::make_unique<Device>(size_t(B)*2*KH*capacity*HD*2); cuda_ok(cudaMemsetAsync(b->p,0,b->bytes,stream),"zero KV"); trt_ok(ctx->setTensorAddress(past.c_str(),b->p),"bind past"); trt_ok(ctx->setTensorAddress(present.c_str(),b->p),"bind present"); out.emplace(present,std::move(b));}
                for(auto const& name:{std::string("logits"),std::string("hidden_states")}){auto d=ctx->getTensorShape(name.c_str()); auto b=std::make_unique<Device>(volume(d)*size_of(engine->getTensorDataType(name.c_str()))); trt_ok(ctx->setTensorAddress(name.c_str(),b->p),"bind output"); out.emplace(name,std::move(b));}
                cuda_ok(cudaStreamSynchronize(stream),"sync inputs"); float before=gpu_used_mib(), ms{}; cudaEvent_t start{},end{}; cuda_ok(cudaEventCreate(&start),"start event"); cuda_ok(cudaEventCreate(&end),"end event"); cuda_ok(cudaEventRecord(start,stream),"start"); trt_ok(ctx->enqueueV3(stream),"enqueue"); cuda_ok(cudaEventRecord(end,stream),"end"); std::vector<char> hidden(out.at("hidden_states")->bytes); cuda_ok(cudaMemcpyAsync(hidden.data(),out.at("hidden_states")->p,hidden.size(),cudaMemcpyDeviceToHost,stream),"copy hidden"); cuda_ok(cudaStreamSynchronize(stream),"sync output"); std::ofstream f(output_file,std::ios::binary); f.write(hidden.data(),hidden.size()); if(!f)throw std::runtime_error("cannot write output"); cuda_ok(cudaEventElapsedTime(&ms,start,end),"elapsed"); cudaEventDestroy(start); cudaEventDestroy(end); std::cout << "OK " << ms << " " << before << " " << gpu_used_mib() << "\n" << std::flush;
            } catch(std::exception const& e) { std::cout << "ERROR " << e.what() << "\n" << std::flush; }
        }
        cudaStreamDestroy(stream); dlclose(plugin); return 0;
    } catch(std::exception const& e) { std::cerr << "persistent engine failed: " << e.what() << '\n'; return 1; }
}
