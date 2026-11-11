#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <sstream>
inline std::string number(double v){
    std::ostringstream s;
    s<<std::setprecision(17)<<v;
    return s.str();
}
inline void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
#define CU(x) check(x)
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point t){
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
struct Stream {
    cudaStream_t s;
    Stream(){
        CU(cudaStreamCreate(&s));
    }
    ~Stream(){
        cudaStreamDestroy(s);
    }
    operator cudaStream_t()const{
        return s;
    }
};
struct Events {
    cudaEvent_t a,b;
    Events(){
        CU(cudaEventCreate(&a));
        CU(cudaEventCreate(&b));
    }
    ~Events(){
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    double time(cudaStream_t s,const std::function<void()>& f){
        CU(cudaEventRecord(a,s));
        f();
        CU(cudaEventRecord(b,s));
        CU(cudaEventSynchronize(b));
        float ms;
        CU(cudaEventElapsedTime(&ms,a,b));
        return ms;
    }
};
struct Env {
    cudaDeviceProp p{
    };
    int runtime,driver,managed,concurrent,mapped,pageable,ats;
    std::string cpu;
    Env(){
        CU(cudaSetDeviceFlags(cudaDeviceMapHost));
        CU(cudaSetDevice(0));
        CU(cudaGetDeviceProperties(&p,0));
        CU(cudaRuntimeGetVersion(&runtime));
        CU(cudaDriverGetVersion(&driver));
        auto attr=[](cudaDeviceAttr a){
            int v;
            CU(cudaDeviceGetAttribute(&v,a,0));
            return v;
        };
        managed=attr(cudaDevAttrManagedMemory);
        concurrent=attr(cudaDevAttrConcurrentManagedAccess);
        mapped=attr(cudaDevAttrCanMapHostMemory);
        pageable=attr(cudaDevAttrPageableMemoryAccess);
        ats=attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
        std::ifstream f("/proc/cpuinfo");
        std::string l;
        while(std::getline(f,l))if(l.find("model name")!=std::string::npos||l.find("CPU part")!=std::string::npos){
            cpu=l;
            break;
        }
        std::cout<<p.name<<" CUDA runtime "<<runtime<<" driver "<<driver<<" mapped="<<mapped<<" managed="<<managed<<" concurrent="<<concurrent<<" pageable="<<pageable<<" ATS="<<ats<<"\n";
    }
};
inline std::string quote(std::string s){
    size_t p=0;
    while((p=s.find('"',p))!=std::string::npos){
        s.insert(p,"\"");
        p+=2;
    }
    return "\""+s+"\"";
}
struct CSV {
    std::ofstream f;
    CSV(const std::string& path,const Env&e,const std::string& header):f(path){
        if(!f)throw std::runtime_error("Cannot open "+path);
        f<<"gpu,cpu,cuda_runtime,cuda_driver,compute_capability,mapped_host,pageable_access,host_page_tables,"<<header<<"\n";
        prefix=quote(e.p.name)+","+quote(e.cpu)+","+std::to_string(e.runtime)+","+std::to_string(e.driver)+","+std::to_string(e.p.major)+"."+std::to_string(e.p.minor)+","+std::to_string(e.mapped)+","+std::to_string(e.pageable)+","+std::to_string(e.ats)+",";
    }
    std::string prefix;
    void row(const std::string&s){
        f<<prefix<<s<<"\n";
        f.flush();
        if(!f)throw std::runtime_error("CSV write failed");
    }
};
struct Buffer {
    void *host=nullptr,*device=nullptr;
    std::string mode;
    size_t bytes;
    Buffer(size_t n,std::string m):mode(m),bytes(n){
        n=std::max(size_t(1),n);
        if(m=="hbm")CU(cudaMalloc(&device,n));
        else if(m=="managed"){
            CU(cudaMallocManaged(&device,n));
            host=device;
        }
        else{
            CU(cudaHostAlloc(&host,n,cudaHostAllocMapped));
            cudaError_t status = cudaHostGetDevicePointer(&device,host,0);
            if (status != cudaSuccess) {
                cudaFreeHost(host);
                host = nullptr;
                CU(status);
            }
        }
    }
    ~Buffer(){
        if(mode=="host")cudaFreeHost(host);
        else cudaFree(device);
    }
    Buffer(const Buffer&)=delete;
    void init(const void* src){
        if(mode=="hbm")CU(cudaMemcpy(device,src,bytes,cudaMemcpyHostToDevice));
        else std::memcpy(host,src,bytes);
    }
    void prefetch(cudaStream_t s){
        if(mode=="managed"){
#if CUDART_VERSION >= 13000
            cudaMemLocation loc{
            };
            loc.type=cudaMemLocationTypeDevice;
            loc.id=0;
            CU(cudaMemPrefetchAsync(device,bytes,loc,0,s));
#else
            CU(cudaMemPrefetchAsync(device,bytes,0,s));
#endif
        }
    }
};
inline int positive(const char*s){
    int n=std::stoi(s);
    if(n<1)throw std::runtime_error("repetitions must be positive");
    return n;
}
