// gpu_info.cu -- print the GPU properties that matter for SpMV: memory size,
// memory bus width and L2 cache size (a vector x smaller than L2 can stay
// on chip, so reordering x accesses matters less).
//
//   nvcc -O2 tools/gpu_info.cu -o gpu_info && ./gpu_info
//
// Generated with assistance of claude code

#include <cstdio>
#include <cuda_runtime.h>

int main()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    {
        std::printf("no CUDA device\n");
        return 1;
    }
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    std::printf("name=%s\n", p.name);
    std::printf("compute_capability=%d.%d\n", p.major, p.minor);
    std::printf("sm_count=%d\n", p.multiProcessorCount);
    std::printf("global_mem_bytes=%zu\n", p.totalGlobalMem);
    std::printf("mem_bus_width_bits=%d\n", p.memoryBusWidth);
    std::printf("l2_cache_bytes=%d\n", p.l2CacheSize);
    std::printf("persisting_l2_max_bytes=%d\n", p.persistingL2CacheMaxSize);
    return 0;
}
