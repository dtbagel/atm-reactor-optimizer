#include "gpu.hpp"
#include "embedded_kernel.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace er2 {
namespace {
#ifdef _WIN32
using Library=HMODULE;
Library open_library(const std::filesystem::path& path) { return LoadLibraryW(path.c_str()); }
void* symbol(Library library,const char* name) { return reinterpret_cast<void*>(GetProcAddress(library,name)); }
void close_library(Library library) { if (library) FreeLibrary(library); }
#else
using Library=void*;
Library open_library(const std::filesystem::path& path) { return dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL); }
void* symbol(Library library,const char* name) { return dlsym(library,name); }
void close_library(Library library) { if (library) dlclose(library); }
#endif
template<class Function> Function load(Library library,const char* name) {
    auto pointer=symbol(library,name);
    if (!pointer) throw std::runtime_error(std::string("Missing GPU API: ")+name);
    return reinterpret_cast<Function>(pointer);
}
void check(int status,const char* operation) {
    if (status) throw std::runtime_error(std::string(operation)+" failed (code "+std::to_string(status)+")");
}
int environment_integer(const char* name,int fallback,int minimum,int maximum) {
    const char* value=std::getenv(name);
    if (!value) return fallback;
    char* end=nullptr;
    long parsed=std::strtol(value,&end,10);
    if (!*value || *end || parsed<minimum || parsed>maximum)
        throw std::runtime_error(std::string(name)+" must be "+std::to_string(minimum)+".."+std::to_string(maximum));
    return static_cast<int>(parsed);
}
bool same_settings(const Settings& a,const Settings& b) {
    return a.width==b.width && a.depth==b.depth && a.height==b.height &&
           a.insertion==b.insertion && a.fill==b.fill && a.variant==b.variant &&
           a.moderator.absorption==b.moderator.absorption &&
           a.moderator.heat_efficiency==b.moderator.heat_efficiency &&
           a.moderator.moderation==b.moderator.moderation &&
           a.moderator.conductivity==b.moderator.conductivity;
}
std::string numeric_definition(const char* name,double value) {
    std::ostringstream text;
    text<<"-D"<<name<<'='<<std::setprecision(std::numeric_limits<double>::max_digits10)<<value;
    return text.str();
}
template<class T> struct CompactResult {
    T power, fuel, efficiency, fuel_heat, reactor_heat, fertility;
    int rods, ticks;
};
static_assert(sizeof(CompactResult<float>)==32);
static_assert(sizeof(CompactResult<double>)==56);
template<class T> void unpack_results(const void* packed,std::span<const Layout> masks,std::span<Result> output,std::span<const int> order) {
    const auto* values=static_cast<const CompactResult<T>*>(packed);
    for (size_t i=0;i<output.size();++i) {
        const size_t destination=order.empty() ? i : size_t(order[i]);
        auto& result=output[destination];
        const auto& value=values[i];
        result.mask=value.rods ? masks[destination] : Layout{};
        result.power=value.power; result.fuel=value.fuel; result.efficiency=value.efficiency;
        result.fuel_heat=value.fuel_heat; result.reactor_heat=value.reactor_heat; result.fertility=value.fertility;
        result.rods=value.rods; result.ticks=value.ticks;
    }
}
Library find_nvrtc() {
    if (const char* override_path=std::getenv("ER2_NVRTC_PATH")) {
        auto library=open_library(override_path);
        if (!library) throw std::runtime_error("Cannot load ER2_NVRTC_PATH");
        return library;
    }
#ifdef _WIN32
    std::vector<std::filesystem::path> dirs;
    wchar_t executable[32768]{};
    GetModuleFileNameW(nullptr,executable,32768);
    dirs.emplace_back(std::filesystem::path(executable).parent_path());
    dirs.emplace_back("tools/local/nvidia/cuda_nvrtc/bin");
    if (const char* cuda=std::getenv("CUDA_PATH")) dirs.emplace_back(std::filesystem::path(cuda)/"bin");
    for (auto& directory:dirs) {
        std::error_code error;
        if (!std::filesystem::is_directory(directory,error)) continue;
        for (auto& entry:std::filesystem::directory_iterator(directory,error)) {
            auto name=entry.path().filename().string();
            if (name.starts_with("nvrtc64_") && name.find(".alt.")==std::string::npos && entry.path().extension()==".dll")
                if (auto library=open_library(std::filesystem::absolute(entry.path()))) return library;
        }
    }
    for (const char* name:{"nvrtc64_130_0.dll","nvrtc64_120_0.dll","nvrtc64_112_0.dll"})
        if (auto library=open_library(name)) return library;
#else
    for (const char* name:{"libnvrtc.so","libnvrtc.so.13","libnvrtc.so.12"})
        if (auto library=open_library(name)) return library;
#endif
    throw std::runtime_error("NVIDIA NVRTC compiler was not found. Run tools/bootstrap.py or set ER2_NVRTC_PATH to its DLL/library.");
}
}
struct Gpu::Impl {
    Library driver=nullptr,nvrtc=nullptr,builtins=nullptr;
    void* context=nullptr; void* module=nullptr; void* kernel=nullptr;
    unsigned long long inputs=0,outputs=0;
    void* host_inputs=nullptr; void* host_outputs=nullptr;
    void* stream=nullptr; void* events[4]{};
    int capacity=0;
    int cells=49;
    bool profiling=false;
    Settings specialized_settings;
    GpuDiagnostics diagnostics;
    GpuTimings timings;
    std::vector<unsigned short> rod_counts;
    std::vector<int> rod_histogram, rod_offsets, input_order;
    unsigned long long tail_mask=~0ULL;
    std::string device_name;
    int (*ctx_destroy)(void*)=nullptr;
    int (*module_unload)(void*)=nullptr;
    int (*mem_free)(unsigned long long)=nullptr;
    int (*host_free)(void*)=nullptr;
    int (*stream_destroy)(void*)=nullptr;
    int (*event_destroy)(void*)=nullptr;
    int (*event_record)(void*,void*)=nullptr;
    int (*event_elapsed)(float*,void*,void*)=nullptr;
    int (*copy_to_gpu)(unsigned long long,const void*,size_t,void*)=nullptr;
    int (*copy_from_gpu)(void*,unsigned long long,size_t,void*)=nullptr;
    int (*launch)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)=nullptr;
    int (*synchronize)(void*)=nullptr;
    ~Impl() {
        if (stream && synchronize) synchronize(stream);
        for (auto event:events) if(event && event_destroy) event_destroy(event);
        if (stream && stream_destroy) stream_destroy(stream);
        if (host_inputs && host_free) host_free(host_inputs);
        if (host_outputs && host_free) host_free(host_outputs);
        if (inputs && mem_free) mem_free(inputs);
        if (outputs && mem_free) mem_free(outputs);
        if (module && module_unload) module_unload(module);
        if (context && ctx_destroy) ctx_destroy(context);
        close_library(nvrtc); close_library(builtins); close_library(driver);
    }
};
Gpu::Gpu(bool exact,int device,int capacity,int cells,const Settings* specialized_settings):impl(std::make_unique<Impl>()) {
    if(cells<1||cells>1024)throw std::runtime_error("GPU footprint must be 1..1024 cells");
    if(capacity<1)throw std::runtime_error("GPU capacity must be positive");
    impl->cells=cells;
    auto& diagnostics=impl->diagnostics;
    diagnostics.layout_words=(cells+63)/64;
    diagnostics.input_bytes_per_layout=diagnostics.layout_words*int(sizeof(unsigned long long));
    diagnostics.output_bytes_per_layout=exact ? sizeof(CompactResult<double>) : sizeof(CompactResult<float>);
    diagnostics.double_precision=exact;
    diagnostics.block_threads=environment_integer("ER2_GPU_BLOCK_THREADS",128,32,1024);
    if(diagnostics.block_threads%32)throw std::runtime_error("ER2_GPU_BLOCK_THREADS must be a multiple of 32");
    diagnostics.fma=environment_integer("ER2_GPU_FMAD",exact ? 0 : 1,0,1)!=0;
    diagnostics.packed_rays=environment_integer("ER2_GPU_PACKED_RAYS",0,0,1)!=0;
    diagnostics.unroll_rays=environment_integer("ER2_GPU_UNROLL_RAYS",0,0,1)!=0;
    diagnostics.reciprocal_divisors=environment_integer("ER2_GPU_RECIPROCAL_DIVISORS",exact ? 0 : 1,0,1)!=0;
    diagnostics.group_rods=environment_integer("ER2_GPU_GROUP_RODS",1,0,1)!=0;
    diagnostics.branchless_rays=environment_integer("ER2_GPU_BRANCHLESS_RAYS",1,0,1)!=0;
    if(diagnostics.group_rods) {
        impl->rod_counts.resize(capacity);
        impl->input_order.resize(capacity);
        impl->rod_histogram.resize(cells+1);
        impl->rod_offsets.resize(cells+1);
        if(cells%64)impl->tail_mask=(1ULL<<(cells%64))-1;
    }
    diagnostics.settings_specialized=specialized_settings && environment_integer("ER2_GPU_SPECIALIZE_SETTINGS",1,0,1)!=0;
    if(diagnostics.settings_specialized) {
        if(specialized_settings->cells()!=cells)throw std::runtime_error("GPU specialization does not match the footprint");
        impl->specialized_settings=*specialized_settings;
    }
    impl->profiling=environment_integer("ER2_GPU_PROFILE",0,0,1)!=0;
#ifdef _WIN32
    impl->driver=open_library("nvcuda.dll");
#else
    impl->driver=open_library("libcuda.so.1");
#endif
    if (!impl->driver) throw std::runtime_error("NVIDIA CUDA driver was not found");
    auto driver=impl->driver;
    check(load<int(*)(unsigned)>(driver,"cuInit")(0),"cuInit");
    int count=0;
    check(load<int(*)(int*)>(driver,"cuDeviceGetCount")(&count),"cuDeviceGetCount");
    if (device<0 || device>=count) throw std::runtime_error("CUDA device index is unavailable");
    int gpu_device;
    check(load<int(*)(int*,int)>(driver,"cuDeviceGet")(&gpu_device,device),"cuDeviceGet");
    char name[256]{};
    check(load<int(*)(char*,int,int)>(driver,"cuDeviceGetName")(name,256,gpu_device),"cuDeviceGetName");
    impl->device_name=name;
    check(load<int(*)(int*,int,int)>(driver,"cuDeviceGetAttribute")(&diagnostics.multiprocessors,16,gpu_device),"cuDeviceGetAttribute multiprocessors");
    int major,minor;
    check(load<int(*)(int*,int*,int)>(driver,"cuDeviceComputeCapability")(&major,&minor,gpu_device),"cuDeviceComputeCapability");
    impl->ctx_destroy=load<int(*)(void*)>(driver,"cuCtxDestroy_v2");
    impl->module_unload=load<int(*)(void*)>(driver,"cuModuleUnload");
    impl->mem_free=load<int(*)(unsigned long long)>(driver,"cuMemFree_v2");
    check(load<int(*)(void**,unsigned,int)>(driver,"cuCtxCreate_v2")(&impl->context,0,gpu_device),"cuCtxCreate");
    impl->nvrtc=find_nvrtc();
    auto rtc=impl->nvrtc;
#ifdef _WIN32
    // NVRTC opens its builtins by basename. Preload the matching companion from
    // its own directory so a project-local or ER2_NVRTC_PATH runtime works too.
    wchar_t rtc_path[32768]{};
    if(GetModuleFileNameW(rtc,rtc_path,32768)) {
        int rtc_major=0,rtc_minor=0;
        if(load<int(*)(int*,int*)>(rtc,"nvrtcVersion")(&rtc_major,&rtc_minor)==0) {
            auto builtin_name="nvrtc-builtins64_"+std::to_string(rtc_major)+std::to_string(rtc_minor)+".dll";
            impl->builtins=open_library(std::filesystem::path(rtc_path).parent_path()/builtin_name);
        }
    }
#endif
    auto create=load<int(*)(void**,const char*,const char*,int,const char* const*,const char* const*)>(rtc,"nvrtcCreateProgram");
    auto compile=load<int(*)(void*,int,const char* const*)>(rtc,"nvrtcCompileProgram");
    auto destroy=load<int(*)(void**)>(rtc,"nvrtcDestroyProgram");
    void* program=nullptr;
    check(create(&program,kernel_source,"reactor.cu",0,nullptr,nullptr),"nvrtcCreateProgram");
    std::string arch="--gpu-architecture=compute_"+std::to_string(major)+std::to_string(minor);
    std::string topology="-DER2_TOPOLOGY_CELLS="+std::to_string(cells);
    std::string layout="-DER2_LAYOUT_WORDS="+std::to_string(diagnostics.layout_words);
    std::vector<const char*> options={"--std=c++17",arch.c_str(),diagnostics.fma ? "--fmad=true" : "--fmad=false",topology.c_str(),layout.c_str()};
    std::vector<std::string> specialization;
    if(diagnostics.settings_specialized) {
        const auto& settings=impl->specialized_settings;
        specialization={"-DER2_SPECIALIZE_SETTINGS",
            numeric_definition("ER2_GPU_WIDTH",settings.width),numeric_definition("ER2_GPU_DEPTH",settings.depth),numeric_definition("ER2_GPU_HEIGHT",settings.height),
            numeric_definition("ER2_GPU_INSERTION",settings.insertion),numeric_definition("ER2_GPU_FILL",settings.fill),numeric_definition("ER2_GPU_VARIANT",settings.variant),
            numeric_definition("ER2_GPU_ABSORPTION",settings.moderator.absorption),numeric_definition("ER2_GPU_HEAT_EFFICIENCY",settings.moderator.heat_efficiency),
            numeric_definition("ER2_GPU_MODERATION",settings.moderator.moderation),numeric_definition("ER2_GPU_CONDUCTIVITY",settings.moderator.conductivity)};
        for(const auto& definition:specialization)options.push_back(definition.c_str());
    }
    if (exact) options.push_back("-DER2_GPU_DOUBLE");
    if (diagnostics.packed_rays) options.push_back("-DER2_PACKED_RAYS");
    if (diagnostics.unroll_rays) options.push_back("-DER2_UNROLL_RAYS");
    if (diagnostics.reciprocal_divisors) options.push_back("-DER2_RECIPROCAL_DIVISORS");
    if (diagnostics.branchless_rays) options.push_back("-DER2_BRANCHLESS_RAYS");
    int status=compile(program,static_cast<int>(options.size()),options.data());
    if (status) {
        size_t size=0;
        load<int(*)(void*,size_t*)>(rtc,"nvrtcGetProgramLogSize")(program,&size);
        std::string log(size,'\0');
        load<int(*)(void*,char*)>(rtc,"nvrtcGetProgramLog")(program,log.data());
        destroy(&program);
        throw std::runtime_error("CUDA compilation failed:\n"+log);
    }
    size_t size=0;
    status=load<int(*)(void*,size_t*)>(rtc,"nvrtcGetPTXSize")(program,&size);
    if (status) { destroy(&program); check(status,"nvrtcGetPTXSize"); }
    std::vector<char> ptx(size);
    status=load<int(*)(void*,char*)>(rtc,"nvrtcGetPTX")(program,ptx.data());
    destroy(&program); check(status,"nvrtcGetPTX");
    if(const char* path=std::getenv("ER2_GPU_DUMP_PTX")) {
        std::ofstream file(path,std::ios::binary);
        if(!file.write(ptx.data(),static_cast<std::streamsize>(ptx.size()-1)))throw std::runtime_error("Cannot write ER2_GPU_DUMP_PTX");
    }
    check(load<int(*)(void**,const void*)>(driver,"cuModuleLoadData")(&impl->module,ptx.data()),"cuModuleLoadData");
    check(load<int(*)(void**,void*,const char*)>(driver,"cuModuleGetFunction")(&impl->kernel,impl->module,"evaluate_layouts"),"cuModuleGetFunction");
    auto attribute=load<int(*)(int*,int,void*)>(driver,"cuFuncGetAttribute");
    int max_threads=0;
    check(attribute(&max_threads,0,impl->kernel),"cuFuncGetAttribute max threads");
    if(diagnostics.block_threads>max_threads)throw std::runtime_error("ER2_GPU_BLOCK_THREADS exceeds this kernel's limit of "+std::to_string(max_threads));
    check(attribute(&diagnostics.shared_bytes_per_block,1,impl->kernel),"cuFuncGetAttribute shared bytes");
    check(attribute(&diagnostics.local_bytes_per_thread,3,impl->kernel),"cuFuncGetAttribute local bytes");
    check(attribute(&diagnostics.registers_per_thread,4,impl->kernel),"cuFuncGetAttribute registers");
    auto allocate=load<int(*)(unsigned long long*,size_t)>(driver,"cuMemAlloc_v2");
    const size_t input_size=size_t(capacity)*diagnostics.input_bytes_per_layout;
    const size_t output_size=size_t(capacity)*diagnostics.output_bytes_per_layout;
    check(allocate(&impl->inputs,input_size),"cuMemAlloc inputs");
    check(allocate(&impl->outputs,output_size),"cuMemAlloc outputs");
    impl->host_free=load<int(*)(void*)>(driver,"cuMemFreeHost");
    auto allocate_host=load<int(*)(void**,size_t)>(driver,"cuMemAllocHost_v2");
    check(allocate_host(&impl->host_inputs,input_size),"cuMemAllocHost inputs");
    check(allocate_host(&impl->host_outputs,output_size),"cuMemAllocHost outputs");
    impl->stream_destroy=load<int(*)(void*)>(driver,"cuStreamDestroy_v2");
    check(load<int(*)(void**,unsigned)>(driver,"cuStreamCreate")(&impl->stream,1),"cuStreamCreate");
    impl->event_destroy=load<int(*)(void*)>(driver,"cuEventDestroy_v2");
    auto create_event=load<int(*)(void**,unsigned)>(driver,"cuEventCreate");
    for (auto& event:impl->events)check(create_event(&event,0),"cuEventCreate");
    impl->event_record=load<int(*)(void*,void*)>(driver,"cuEventRecord");
    impl->event_elapsed=load<int(*)(float*,void*,void*)>(driver,"cuEventElapsedTime");
    impl->capacity=capacity;
    impl->copy_to_gpu=load<decltype(impl->copy_to_gpu)>(driver,"cuMemcpyHtoDAsync_v2");
    impl->copy_from_gpu=load<decltype(impl->copy_from_gpu)>(driver,"cuMemcpyDtoHAsync_v2");
    impl->launch=load<decltype(impl->launch)>(driver,"cuLaunchKernel");
    impl->synchronize=load<int(*)(void*)>(driver,"cuStreamSynchronize");
}
Gpu::~Gpu()=default;
const std::string& Gpu::name() const { return impl->device_name; }
const GpuDiagnostics& Gpu::diagnostics() const { return impl->diagnostics; }
const GpuTimings& Gpu::timings() const { return impl->timings; }
void Gpu::set_profiling(bool enabled) { impl->profiling=enabled; }
void Gpu::evaluate(std::span<const Layout> masks,std::span<Result> output,Settings settings,SimConfig config) {
    if(settings.cells()!=impl->cells)throw std::runtime_error("GPU was compiled for a different footprint");
    if(impl->diagnostics.settings_specialized && !same_settings(settings,impl->specialized_settings))throw std::runtime_error("GPU was compiled for different reactor settings");
    if (masks.size()!=output.size() || masks.size()>size_t(impl->capacity)) throw std::runtime_error("GPU batch exceeds capacity");
    impl->timings={};
    if (masks.empty()) return;
    using Clock=std::chrono::steady_clock;
    const auto started=impl->profiling ? Clock::now() : Clock::time_point{};
    const int words=impl->diagnostics.layout_words;
    std::span<const int> order;
    impl->timings.layouts=static_cast<int>(masks.size());
    if(impl->diagnostics.group_rods) {
        int common=-1, distinct=1;
        bool heterogeneous=false;
        for(size_t i=0;i<masks.size();++i) {
            int rods=0;
            for(int word=0;word<words-1;++word)rods+=std::popcount(masks[i].words[word]);
            rods+=std::popcount(masks[i].words[words-1]&impl->tail_mask);
            if(common<0)common=rods;
            if(!heterogeneous && rods!=common) {
                // A homogeneous batch pays only for counting; defer histogram
                // and index storage until another rod count actually appears.
                std::fill(impl->rod_histogram.begin(),impl->rod_histogram.end(),0);
                std::fill_n(impl->rod_counts.begin(),i,static_cast<unsigned short>(common));
                impl->rod_histogram[common]=static_cast<int>(i);
                heterogeneous=true;
            }
            if(heterogeneous) {
                impl->rod_counts[i]=static_cast<unsigned short>(rods);
                if(impl->rod_histogram[rods]++==0)++distinct;
            }
        }
        impl->timings.distinct_rod_counts=distinct;
        // Keep a homogeneous batch in place; extra scatter work cannot improve
        // its source-period coherence.
        if(distinct>1) {
            int next=0;
            for(int rods=0;rods<=impl->cells;++rods) {
                impl->rod_offsets[rods]=next;
                next+=impl->rod_histogram[rods];
            }
            for(size_t i=0;i<masks.size();++i)impl->input_order[impl->rod_offsets[impl->rod_counts[i]]++]=static_cast<int>(i);
            order=std::span(impl->input_order).first(masks.size());
            impl->timings.grouping_applied=true;
        }
    }
    auto* packed=static_cast<unsigned long long*>(impl->host_inputs);
    if(words==1 && order.empty()) {
        for(size_t i=0;i<masks.size();++i)packed[i]=masks[i].words[0];
    } else if(words==1) {
        for(size_t i=0;i<masks.size();++i)packed[i]=masks[order[i]].words[0];
    } else {
        for(size_t i=0;i<masks.size();++i) {
            const size_t source=order.empty() ? i : size_t(order[i]);
            std::memcpy(packed+i*words,masks[source].words,size_t(words)*sizeof(*packed));
        }
    }
    const auto packed_at=impl->profiling ? Clock::now() : Clock::time_point{};
    int count=static_cast<int>(masks.size());
    if(impl->profiling)check(impl->event_record(impl->events[0],impl->stream),"cuEventRecord upload start");
    check(impl->copy_to_gpu(impl->inputs,impl->host_inputs,masks.size()*impl->diagnostics.input_bytes_per_layout,impl->stream),"cuMemcpyHtoDAsync");
    if(impl->profiling)check(impl->event_record(impl->events[1],impl->stream),"cuEventRecord upload end");
    void* arguments[]={&impl->inputs,&impl->outputs,&count,&settings,&config};
    const unsigned threads=impl->diagnostics.block_threads;
    check(impl->launch(impl->kernel,(count+threads-1)/threads,1,1,threads,1,1,0,impl->stream,arguments,nullptr),"cuLaunchKernel");
    if(impl->profiling)check(impl->event_record(impl->events[2],impl->stream),"cuEventRecord kernel end");
    check(impl->copy_from_gpu(impl->host_outputs,impl->outputs,output.size()*impl->diagnostics.output_bytes_per_layout,impl->stream),"cuMemcpyDtoHAsync");
    if(impl->profiling)check(impl->event_record(impl->events[3],impl->stream),"cuEventRecord download end");
    check(impl->synchronize(impl->stream),"cuStreamSynchronize");
    const auto downloaded_at=impl->profiling ? Clock::now() : Clock::time_point{};
    if(impl->diagnostics.double_precision)unpack_results<double>(impl->host_outputs,masks,output,order);
    else unpack_results<float>(impl->host_outputs,masks,output,order);
    if(impl->profiling) {
        const auto finished=Clock::now();
        auto& timings=impl->timings;
        timings.layouts=count;
        timings.packing_ms=std::chrono::duration<double,std::milli>(packed_at-started).count();
        timings.unpacking_ms=std::chrono::duration<double,std::milli>(finished-downloaded_at).count();
        timings.total_ms=std::chrono::duration<double,std::milli>(finished-started).count();
        float duration=0;
        check(impl->event_elapsed(&duration,impl->events[0],impl->events[1]),"cuEventElapsedTime upload");timings.upload_ms=duration;
        check(impl->event_elapsed(&duration,impl->events[1],impl->events[2]),"cuEventElapsedTime kernel");timings.kernel_ms=duration;
        check(impl->event_elapsed(&duration,impl->events[2],impl->events[3]),"cuEventElapsedTime download");timings.download_ms=duration;
    }
}
}
