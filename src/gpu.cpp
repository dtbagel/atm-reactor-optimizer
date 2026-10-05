#include "gpu.hpp"
#include "embedded_kernel.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
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
    int capacity=0;
    int cells=49;
    std::string device_name;
    int (*ctx_destroy)(void*)=nullptr;
    int (*module_unload)(void*)=nullptr;
    int (*mem_free)(unsigned long long)=nullptr;
    int (*copy_to_gpu)(unsigned long long,const void*,size_t)=nullptr;
    int (*copy_from_gpu)(void*,unsigned long long,size_t)=nullptr;
    int (*launch)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**)=nullptr;
    int (*synchronize)()=nullptr;
    ~Impl() {
        if (inputs && mem_free) mem_free(inputs);
        if (outputs && mem_free) mem_free(outputs);
        if (module && module_unload) module_unload(module);
        if (context && ctx_destroy) ctx_destroy(context);
        close_library(nvrtc); close_library(builtins); close_library(driver);
    }
};
Gpu::Gpu(bool exact,int device,int capacity,int cells):impl(std::make_unique<Impl>()) {
    if(cells<1||cells>1024)throw std::runtime_error("GPU footprint must be 1..1024 cells");
    impl->cells=cells;
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
    std::vector<const char*> options={"--std=c++17",arch.c_str(),"--fmad=false",topology.c_str()};
    if (exact) options.push_back("-DER2_GPU_DOUBLE");
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
    check(load<int(*)(void**,const void*)>(driver,"cuModuleLoadData")(&impl->module,ptx.data()),"cuModuleLoadData");
    check(load<int(*)(void**,void*,const char*)>(driver,"cuModuleGetFunction")(&impl->kernel,impl->module,"evaluate_layouts"),"cuModuleGetFunction");
    auto allocate=load<int(*)(unsigned long long*,size_t)>(driver,"cuMemAlloc_v2");
    check(allocate(&impl->inputs,size_t(capacity)*sizeof(Layout)),"cuMemAlloc inputs");
    check(allocate(&impl->outputs,size_t(capacity)*sizeof(Result)),"cuMemAlloc outputs");
    impl->capacity=capacity;
    impl->copy_to_gpu=load<int(*)(unsigned long long,const void*,size_t)>(driver,"cuMemcpyHtoD_v2");
    impl->copy_from_gpu=load<int(*)(void*,unsigned long long,size_t)>(driver,"cuMemcpyDtoH_v2");
    impl->launch=load<decltype(impl->launch)>(driver,"cuLaunchKernel");
    impl->synchronize=load<int(*)()>(driver,"cuCtxSynchronize");
}
Gpu::~Gpu()=default;
const std::string& Gpu::name() const { return impl->device_name; }
void Gpu::evaluate(std::span<const Layout> masks,std::span<Result> output,Settings settings,SimConfig config) {
    if(settings.cells()!=impl->cells)throw std::runtime_error("GPU was compiled for a different footprint");
    if (masks.size()!=output.size() || masks.size()>size_t(impl->capacity)) throw std::runtime_error("GPU batch exceeds capacity");
    if (masks.empty()) return;
    int count=static_cast<int>(masks.size());
    check(impl->copy_to_gpu(impl->inputs,masks.data(),masks.size_bytes()),"cuMemcpyHtoD");
    void* arguments[]={&impl->inputs,&impl->outputs,&count,&settings,&config};
    check(impl->launch(impl->kernel,(count+127)/128,1,1,128,1,1,0,nullptr,arguments,nullptr),"cuLaunchKernel");
    check(impl->synchronize(),"cuCtxSynchronize");
    check(impl->copy_from_gpu(output.data(),impl->outputs,output.size_bytes()),"cuMemcpyDtoH");
}
}
