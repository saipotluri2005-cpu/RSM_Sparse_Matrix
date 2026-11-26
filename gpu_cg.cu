// gpu_cg.cu -- Conjugate Gradient on the Vista GH200 (Grace CPU + H200 GPU).
//
// How the work is split (see docs/GPU_PLAN.md):
//   CPU (72 Grace cores, OpenMP): read .mtx, build COO -> RCM -> CSR.
//                                 This is done for matrix k+1 while the GPU
//                                 solves matrix k (std::async pipeline).
//   NVLink-C2C (~400 GB/s):       the CSR is copied into HBM ONCE per matrix.
//   GPU (H200, HBM3 ~4 TB/s):     the whole CG loop. alpha/beta stay in GPU
//                                 memory, so the CPU never waits on a dot
//                                 product. The CPU only reads a 48-byte status
//                                 block every CHECK_EVERY iterations.
//   NVLink-C2C back:              x (8N bytes) is copied once at the end into
//                                 pinned memory, and the CPU checks the residual.
//
// Same input/output conventions as cpu_sparse_v2, so benchmark.sh parses it:
//   ./gpu_cg [folder-or-file]           (default folder: Matrices)
//
// Environment options:
//   GPU_MATRIX=hbm      (default) cudaMalloc + one copy over C2C
//   GPU_MATRIX=managed  cudaMallocManaged + prefetch to the GPU
//   GPU_MATRIX=host     no copy: GPU reads the CSR from CPU memory over C2C
//   STAGE=pageable      (default) copy straight from the malloc'd CSR
//   STAGE=pinned        cudaHostRegister the CSR first (registration timed)
//   SPMV=cusparse       (default) cuSPARSE SpMV + separate p.Ap kernel
//   SPMV=fused          our CSR kernel: Ap = A p and p.Ap in one pass
//   CUDA_GRAPH=1        (default) one CUDA graph per CHECK_EVERY iterations
//   CHECK_EVERY=16      iterations between convergence checks by the CPU
//   PRECOND=jacobi, REORDER=rcm|auto|none, MAX_ITERS, TOL  as on the CPU
//   VERIFY=1            (default) recompute ||b - Ax|| on the CPU from the copied x
//   SAVE_SOLUTION=0     (default here) 1 writes <name>_gpu_solution.txt
//   PIPELINE=1          (default) prepare the next matrix while this one solves
//   RHS=ones            (default) b = 1, as in the CPU programs
//   RHS=file            b from <name>_b.mtx next to the matrix (gen_bridge.py, SuiteSparse)
//   SOLUTION_DIR=.      where SAVE_SOLUTION=1 writes <name>_gpu_solution.txt
//
// Generated with assistance of claude code
#include "sparse_prep.hpp"
#include "gpu_cg_logic.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <climits>
#include <future>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CK(call)                                                                     \
    do {                                                                             \
        cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), \
                         __FILE__, __LINE__, cudaGetErrorString(e_));                \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)
#define CKS(call)                                                                    \
    do {                                                                             \
        cusparseStatus_t s_ = (call);                                                \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                         \
            std::fprintf(stderr, "cuSPARSE error %d at %s:%d: %s\n", (int)s_,        \
                         __FILE__, __LINE__, cusparseGetErrorString(s_));            \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
constexpr int BLOCK = 256;

// Sum v over the block and add it to *out with one atomic per block.
__device__ inline void block_add(double v, double *out)
{
    __shared__ double warp_sum[BLOCK / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) warp_sum[w] = v;
    __syncthreads();
    if (w == 0)
    {
        v = lane < BLOCK / 32 ? warp_sum[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) atomicAdd(out, v);
    }
}
// Two sums at once (rz and rr share one pass).
__device__ inline void block_add2(double a, double b, double *oa, double *ob)
{
    __shared__ double wa[BLOCK / 32], wb[BLOCK / 32];
    for (int o = 16; o > 0; o >>= 1)
    {
        a += __shfl_down_sync(0xffffffffu, a, o);
        b += __shfl_down_sync(0xffffffffu, b, o);
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { wa[w] = a; wb[w] = b; }
    __syncthreads();
    if (w == 0)
    {
        a = lane < BLOCK / 32 ? wa[lane] : 0.0;
        b = lane < BLOCK / 32 ? wb[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1)
        {
            a += __shfl_down_sync(0xffffffffu, a, o);
            b += __shfl_down_sync(0xffffffffu, b, o);
        }
        if (lane == 0) { atomicAdd(oa, a); atomicAdd(ob, b); }
    }
}

// x = 0, r = b, z = M^-1 r, p = z; rz = r.z, bb = b.b
// (r = b - A*0 = b, exactly what the CPU computes for x = 0). b == nullptr means b = 1.
template <bool PRE>
__global__ void k_init(int n, const double *__restrict b, const double *__restrict dinv, double *x,
                       double *r, double *z, double *p, CGScalars *s)
{
    double rz = 0.0, bb = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const double bi = b ? b[i] : 1.0, ri = bi;
        const double zi = PRE ? dinv[i] * ri : ri;
        x[i] = 0.0; r[i] = ri; p[i] = zi;
        if (PRE) z[i] = zi;
        rz += ri * zi;
        bb += bi * bi;
    }
    block_add2(rz, bb, &s->rz, &s->bb);
}

// Jacobi: dinv[i] = 1 / A(i,i)  (1 when the diagonal is missing or <= 0, as on the CPU)
__global__ void k_dinv(int n, const int *__restrict rp, const int *__restrict ci,
                       const double *__restrict val, double *dinv)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        double d = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k)
            if (ci[k] == i) { d = val[k]; break; }
        dinv[i] = d > 0.0 ? 1.0 / d : 1.0;
    }
}

// pap += p . Ap   (used after cuSPARSE SpMV)
__global__ void k_dot_pap(int n, const double *__restrict p, const double *__restrict ap,
                          CGScalars *s)
{
    if (s->done) return;
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += p[i] * ap[i];
    block_add(acc, &s->pap);
}

// Fused SpMV: Ap = A p and pap += p . Ap, one group of G lanes per row.
// G is picked from the average nonzeros per row (FEM rows: ~7-80).
// The row loop is uniform across each warp (every lane runs the same number
// of trips) so the full-mask shuffles below are always legal.
template <int G>
__global__ void k_spmv_pap(int n, const int *__restrict rp, const int *__restrict ci,
                           const double *__restrict val, const double *__restrict p,
                           double *__restrict ap, CGScalars *s)
{
    if (s->done) return;
    constexpr int GPW = 32 / G;                         // row groups per warp
    const int lane = threadIdx.x & (G - 1);             // lane inside the group
    const int sub = (threadIdx.x & 31) / G;             // group inside the warp
    const long long warp = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / 32;
    const long long warps = gridDim.x * (long long)blockDim.x / 32;
    double acc = 0.0;
    for (long long base = warp * GPW; base < n; base += warps * GPW)
    {
        const long long row = base + sub;
        double sum = 0.0;
        if (row < n)
        {
            const int end = rp[row + 1];
            for (int k = rp[row] + lane; k < end; k += G) sum += val[k] * p[ci[k]];
        }
        for (int o = G / 2; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o, G);
        if (lane == 0 && row < n)
        {
            ap[row] = sum;
            acc += p[row] * sum;
        }
    }
    block_add(acc, &s->pap);
}

// alpha = rz / pap; x += alpha p; r -= alpha Ap; z = M^-1 r; rz_new += r.z; rr += r.r
template <bool PRE>
__global__ void k_update1(int n, const double *__restrict dinv, const double *__restrict p,
                          const double *__restrict ap, double *x, double *r, double *z,
                          CGScalars *s)
{
    if (s->done) return;
    const double alpha = cg_alpha(*s);
    double rz = 0.0, rr = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        x[i] += alpha * p[i];
        const double ri = r[i] - alpha * ap[i];
        r[i] = ri;
        rr += ri * ri;
        if (PRE)
        {
            const double zi = dinv[i] * ri;
            z[i] = zi;
            rz += ri * zi;
        }
    }
    if (PRE) block_add2(rz, rr, &s->rz_new, &s->rr);
    else block_add(rr, &s->rr); // rz_new = rr, copied in k_scalar
}

// One thread: iteration count, convergence test, beta. See gpu_cg_logic.hpp.
template <bool PRE>
__global__ void k_scalar(CGScalars *s, double tol, int max_iters)
{
    if (!PRE && !s->done) s->rz_new = s->rr; // without a preconditioner z = r
    cg_after_update(*s, tol, max_iters);
}

// p = z + beta p   (z = r without a preconditioner)
__global__ void k_update2(int n, const double *__restrict z, double *p, const CGScalars *s)
{
    if (s->done) return;
    const double beta = s->beta;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        p[i] = z[i] + beta * p[i];
}

// res2 += (b - A x)^2  (true residual, after the solve; b == nullptr means b = 1)
__global__ void k_true_residual(int n, const int *__restrict rp, const int *__restrict ci,
                                const double *__restrict val, const double *__restrict x,
                                const double *__restrict b, double *res2)
{
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        double ax = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) ax += val[k] * x[ci[k]];
        const double d = (b ? b[i] : 1.0) - ax;
        acc += d * d;
    }
    block_add(acc, res2);
}

// ---------------------------------------------------------------------------
// Device-side matrix: where the CSR lives depends on GPU_MATRIX
// ---------------------------------------------------------------------------
struct DeviceMatrix
{
    std::string mode;
    int n = 0;
    long long nnz = 0;
    int *rp = nullptr, *ci = nullptr; // device-usable pointers
    double *val = nullptr;
    double h2d_ms = 0.0, register_ms = 0.0;
    double bytes = 0.0;
    // ownership for cleanup
    bool owns_device = false, owns_managed = false;
    std::vector<void *> registered;
};

static double ms_between(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// rp32 is the int32 copy of the row pointers (cuSPARSE wants one index type).
static DeviceMatrix upload_matrix(const CSR &A, std::vector<int> &rp32, cudaStream_t st, int dev)
{
    DeviceMatrix D;
    D.mode = env_str("GPU_MATRIX", "hbm");
    D.n = A.n;
    D.nnz = A.nnz;
    const size_t b_rp = sizeof(int) * (size_t)(A.n + 1), b_ci = sizeof(int) * (size_t)A.nnz,
                 b_val = sizeof(double) * (size_t)A.nnz;
    D.bytes = (double)(b_rp + b_ci + b_val);

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    const bool pin = env_str("STAGE", "pageable") == "pinned" || D.mode == "host";
    if (pin)
    {
        auto t0 = std::chrono::steady_clock::now();
        void *ptrs[3] = {rp32.data(), A.ci.get(), A.val.get()};
        size_t sizes[3] = {b_rp, b_ci, b_val};
        for (int k = 0; k < 3; ++k)
        {
            unsigned flags = D.mode == "host" ? cudaHostRegisterMapped : cudaHostRegisterDefault;
            CK(cudaHostRegister(ptrs[k], sizes[k], flags));
            D.registered.push_back(ptrs[k]);
        }
        D.register_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
    }

    if (D.mode == "host")
    {
        // No copy at all: every SpMV streams the matrix over NVLink-C2C.
        void *p;
        CK(cudaHostGetDevicePointer(&p, rp32.data(), 0)); D.rp = (int *)p;
        CK(cudaHostGetDevicePointer(&p, A.ci.get(), 0));  D.ci = (int *)p;
        CK(cudaHostGetDevicePointer(&p, A.val.get(), 0)); D.val = (double *)p;
    }
    else if (D.mode == "managed")
    {
        CK(cudaMallocManaged(&D.rp, b_rp));
        CK(cudaMallocManaged(&D.ci, b_ci));
        CK(cudaMallocManaged(&D.val, b_val));
        D.owns_managed = true;
        // first written by the CPU (pages land in LPDDR5X) ...
        std::memcpy(D.rp, rp32.data(), b_rp);
#pragma omp parallel for schedule(static)
        for (long long k = 0; k < A.nnz; ++k) { D.ci[k] = A.ci[k]; D.val[k] = A.val[k]; }
        // ... then migrated to HBM before the solve, and that migration is timed
        CK(cudaEventRecord(e0, st));
#if CUDART_VERSION >= 13000
        cudaMemLocation loc{};
        loc.type = cudaMemLocationTypeDevice;
        loc.id = dev;
        CK(cudaMemPrefetchAsync(D.rp, b_rp, loc, 0, st));
        CK(cudaMemPrefetchAsync(D.ci, b_ci, loc, 0, st));
        CK(cudaMemPrefetchAsync(D.val, b_val, loc, 0, st));
#else
        CK(cudaMemPrefetchAsync(D.rp, b_rp, dev, st));
        CK(cudaMemPrefetchAsync(D.ci, b_ci, dev, st));
        CK(cudaMemPrefetchAsync(D.val, b_val, dev, st));
#endif
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        D.h2d_ms = ms_between(e0, e1);
    }
    else
    {
        if (D.mode != "hbm")
            std::fprintf(stderr, "Unknown GPU_MATRIX=%s, using hbm\n", D.mode.c_str());
        D.mode = "hbm";
        CK(cudaMalloc(&D.rp, b_rp));
        CK(cudaMalloc(&D.ci, b_ci));
        CK(cudaMalloc(&D.val, b_val));
        D.owns_device = true;
        CK(cudaEventRecord(e0, st));
        CK(cudaMemcpyAsync(D.rp, rp32.data(), b_rp, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(D.ci, A.ci.get(), b_ci, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(D.val, A.val.get(), b_val, cudaMemcpyHostToDevice, st));
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        D.h2d_ms = ms_between(e0, e1);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    (void)dev;
    return D;
}

static void free_matrix(DeviceMatrix &D)
{
    if (D.owns_device || D.owns_managed)
    {
        CK(cudaFree(D.rp));
        CK(cudaFree(D.ci));
        CK(cudaFree(D.val));
    }
    for (void *p : D.registered) CK(cudaHostUnregister(p));
    D.registered.clear();
}

// ---------------------------------------------------------------------------
// The solve
// ---------------------------------------------------------------------------
struct GpuContext
{
    int dev = 0, sms = 0;
    cudaStream_t st{};
    cusparseHandle_t sp{};
};

struct SolveResult
{
    int iterations = 0;
    double solve_ms = 0.0, setup_ms = 0.0, d2h_ms = 0.0, gpu_rel_res = 0.0;
    int chunks = 0;
    bool graph_used = false;
    std::string spmv;
};

template <bool PRE>
static SolveResult solve_on_gpu(GpuContext &g, const DeviceMatrix &D, const double *d_b,
                                double *h_x, int max_iters, double tol)
{
    SolveResult R;
    const int n = D.n;
    const size_t vb = sizeof(double) * (size_t)n;
    cudaStream_t st = g.st;

    // vectors always in HBM
    double *x, *r, *p, *ap, *z = nullptr, *dinv = nullptr;
    CK(cudaMalloc(&x, vb));
    CK(cudaMalloc(&r, vb));
    CK(cudaMalloc(&p, vb));
    CK(cudaMalloc(&ap, vb));
    if (PRE) { CK(cudaMalloc(&z, vb)); CK(cudaMalloc(&dinv, vb)); }
    CGScalars *s;
    CK(cudaMalloc(&s, sizeof(CGScalars)));
    CGScalars *h_s; // pinned, written by the GPU every check
    CK(cudaMallocHost(&h_s, 2 * sizeof(CGScalars)));

    // enough blocks to fill the GPU, few enough that atomics stay cheap
    const int vec_blocks = std::max(1, std::min((n + BLOCK - 1) / BLOCK, g.sms * 8));

    // SpMV choice
    R.spmv = env_str("SPMV", "cusparse");
    const double per_row = (double)D.nnz / std::max(1, n);
    int G = 2;
    while (G < 32 && G < per_row) G <<= 1;
    const int spmv_blocks =
        std::max(1, std::min((int)(((long long)n * G + BLOCK - 1) / BLOCK), g.sms * 16));

    cusparseSpMatDescr_t mA{};
    cusparseDnVecDescr_t vP{}, vAP{};
    void *sp_buf = nullptr;
    const double one = 1.0, zero = 0.0;
    if (R.spmv == "cusparse")
    {
        CKS(cusparseCreateCsr(&mA, n, n, D.nnz, D.rp, D.ci, D.val, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CKS(cusparseCreateDnVec(&vP, n, p, CUDA_R_64F));
        CKS(cusparseCreateDnVec(&vAP, n, ap, CUDA_R_64F));
        size_t sz = 0;
        CKS(cusparseSpMV_bufferSize(g.sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mA, vP, &zero,
                                    vAP, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &sz));
        CK(cudaMalloc(&sp_buf, std::max<size_t>(sz, 16)));
    }
    else
        R.spmv = "fused(G=" + std::to_string(G) + ")";

    auto launch_spmv = [&]() {
        if (mA)
        {
            CKS(cusparseSpMV(g.sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mA, vP, &zero, vAP,
                             CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, sp_buf));
            k_dot_pap<<<vec_blocks, BLOCK, 0, st>>>(n, p, ap, s);
            return;
        }
        switch (G)
        {
        case 2:  k_spmv_pap<2><<<spmv_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, p, ap, s); break;
        case 4:  k_spmv_pap<4><<<spmv_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, p, ap, s); break;
        case 8:  k_spmv_pap<8><<<spmv_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, p, ap, s); break;
        case 16: k_spmv_pap<16><<<spmv_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, p, ap, s); break;
        default: k_spmv_pap<32><<<spmv_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, p, ap, s); break;
        }
    };
    auto launch_iteration = [&]() {
        launch_spmv();
        k_update1<PRE><<<vec_blocks, BLOCK, 0, st>>>(n, dinv, p, ap, x, r, z, s);
        k_scalar<PRE><<<1, 1, 0, st>>>(s, tol, max_iters);
        k_update2<<<vec_blocks, BLOCK, 0, st>>>(n, PRE ? z : r, p, s);
    };

    const int check_every = (int)std::max(1LL, env_ll("CHECK_EVERY", 16));
    auto launch_chunk_plain = [&]() {
        for (int k = 0; k < check_every; ++k) launch_iteration();
    };

    // ---- setup: init vectors, then (optionally) capture one chunk as a graph ----
    cudaEvent_t e_t0, e_t1, e_chk[2];
    CK(cudaEventCreate(&e_t0));
    CK(cudaEventCreate(&e_t1));
    CK(cudaEventCreateWithFlags(&e_chk[0], cudaEventDisableTiming | cudaEventBlockingSync));
    CK(cudaEventCreateWithFlags(&e_chk[1], cudaEventDisableTiming | cudaEventBlockingSync));

    auto t_setup = std::chrono::steady_clock::now();
    CK(cudaMemsetAsync(s, 0, sizeof(CGScalars), st));
    if (PRE) k_dinv<<<vec_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, dinv);
    k_init<PRE><<<vec_blocks, BLOCK, 0, st>>>(n, d_b, dinv, x, r, z, p, s);
    CK(cudaGetLastError());

    cudaGraphExec_t exec = nullptr;
    if (env_str("CUDA_GRAPH", "1") != "0")
    {
        cudaGraph_t graph = nullptr;
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess)
        {
            launch_chunk_plain();
            cudaError_t ce = cudaStreamEndCapture(st, &graph);
            if (ce == cudaSuccess && graph &&
                cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess)
                R.graph_used = true;
            else
                exec = nullptr;
            if (graph) cudaGraphDestroy(graph);
        }
        if (!R.graph_used)
        {
            (void)cudaGetLastError(); // clear the capture error, fall back to launches
            std::fprintf(stderr, "Note: CUDA graph capture failed, using plain launches\n");
        }
    }
    CK(cudaStreamSynchronize(st));
    R.setup_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_setup).count();

    // ---- solve: GPU runs ahead one chunk while the CPU checks the previous one ----
    // Chunk k+1 is launched before the CPU looks at chunk k's status, so the
    // GPU never idles waiting for the CPU. Iterations past convergence are
    // no-ops: every kernel returns at once when s->done is set (cuSPARSE SpMV
    // cannot see the flag, so with SPMV=cusparse those SpMVs still run).
    auto launch_chunk = [&](int slot) {
        if (exec) CK(cudaGraphLaunch(exec, st));
        else launch_chunk_plain();
        CK(cudaMemcpyAsync(&h_s[slot], s, sizeof(CGScalars), cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(e_chk[slot], st));
    };
    CK(cudaEventRecord(e_t0, st));
    launch_chunk(0);
    int chunk = 0;
    while (true)
    {
        const int cur = chunk & 1;
        launch_chunk(cur ^ 1);            // keep the GPU busy ...
        CK(cudaEventSynchronize(e_chk[cur])); // ... while we read the older status
        ++chunk;
        if (h_s[cur].done) break;
    }
    CK(cudaEventRecord(e_t1, st));
    CK(cudaEventSynchronize(e_t1));
    R.solve_ms = ms_between(e_t0, e_t1);
    R.chunks = chunk + 1;
    CK(cudaMemcpy(h_s, s, sizeof(CGScalars), cudaMemcpyDeviceToHost));
    R.iterations = h_s[0].it;

    // ---- true residual on the GPU ----
    double *res2;
    CK(cudaMalloc(&res2, sizeof(double)));
    CK(cudaMemsetAsync(res2, 0, sizeof(double), st));
    k_true_residual<<<vec_blocks, BLOCK, 0, st>>>(n, D.rp, D.ci, D.val, x, d_b, res2);
    double h_res2 = 0.0;
    CK(cudaMemcpyAsync(&h_res2, res2, sizeof(double), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    const double bb = h_s[0].bb > 0.0 ? h_s[0].bb : 1.0;
    R.gpu_rel_res = std::sqrt(h_res2) / std::sqrt(bb);

    // ---- relay x back to the CPU over C2C (pinned destination) ----
    CK(cudaEventRecord(e_t0, st));
    CK(cudaMemcpyAsync(h_x, x, vb, cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecord(e_t1, st));
    CK(cudaEventSynchronize(e_t1));
    R.d2h_ms = ms_between(e_t0, e_t1);

    // ---- cleanup ----
    if (exec) cudaGraphExecDestroy(exec);
    if (mA)
    {
        CKS(cusparseDestroySpMat(mA));
        CKS(cusparseDestroyDnVec(vP));
        CKS(cusparseDestroyDnVec(vAP));
        CK(cudaFree(sp_buf));
    }
    for (cudaEvent_t e : {e_t0, e_t1, e_chk[0], e_chk[1]})
        CK(cudaEventDestroy(e));
    CK(cudaFree(res2));
    CK(cudaFree(x)); CK(cudaFree(r)); CK(cudaFree(p)); CK(cudaFree(ap)); CK(cudaFree(s));
    if (PRE) { CK(cudaFree(z)); CK(cudaFree(dinv)); }
    CK(cudaFreeHost(h_s));
    return R;
}

// ---------------------------------------------------------------------------
// One matrix: upload, solve, copy back, verify, report
// ---------------------------------------------------------------------------
static void run_matrix(GpuContext &g, Prepared &P)
{
    std::cout << "\n==================================================\n";
    std::cout << "Processing File: " << P.file.string() << "\n";
    std::cout << "==================================================\n";
    if (!P.skip_reason.empty())
    {
        std::cout << P.skip_reason << std::endl;
        return;
    }
    CSR &A = P.A;
    const int N = A.n;
    if (N <= 0) { std::cout << "Skipped: empty matrix\n"; return; }
    if (A.nnz > INT_MAX)
    {
        std::cout << "Skipped: " << A.nnz << " nonzeros needs 64-bit indices (not built)\n";
        return;
    }
    std::cout << "Matrix Size:        " << N << " x " << N << std::endl;
    std::cout << "Nonzeros:           " << A.nnz << "  (" << (double)A.nnz / N << " per row)"
              << std::endl;
    std::cout << "Bandwidth:          " << P.bw_before << " -> " << P.bw_after << P.bw_note
              << std::endl;

    size_t free_b = 0, total_b = 0;
    CK(cudaMemGetInfo(&free_b, &total_b));
    const double need = 12.0 * A.nnz + 4.0 * (N + 1) + 7.0 * 8.0 * N;
    if (need > 0.9 * (double)free_b)
    {
        std::cout << "Skipped: needs " << need / 1e9 << " GB of HBM, " << free_b / 1e9
                  << " GB free\n";
        return;
    }

    std::vector<int> rp32(N + 1);
    for (int i = 0; i <= N; ++i) rp32[i] = (int)A.rp[i];

    DeviceMatrix D = upload_matrix(A, rp32, g.st, g.dev);

    double *h_x, *d_b = nullptr;
    CK(cudaMallocHost(&h_x, sizeof(double) * (size_t)N));
    if (!P.b.empty())
    {
        CK(cudaMalloc(&d_b, sizeof(double) * (size_t)N));
        CK(cudaMemcpy(d_b, P.b.data(), sizeof(double) * (size_t)N, cudaMemcpyHostToDevice));
    }
    const bool jacobi = env_str("PRECOND", "none") == "jacobi";
    const int max_iters = std::atoi(env_str("MAX_ITERS", "5000").c_str());
    const double tol = std::atof(env_str("TOL", "1e-5").c_str());
    SolveResult R = jacobi ? solve_on_gpu<true>(g, D, d_b, h_x, max_iters, tol)
                           : solve_on_gpu<false>(g, D, d_b, h_x, max_iters, tol);

    // CPU check of the copied-back x against the CPU's own copy of A
    double rel_res = R.gpu_rel_res;
    std::string res_src = "GPU";
    if (env_str("VERIFY", "1") != "0")
    {
        double res2 = 0.0, b2 = 0.0;
#pragma omp parallel for reduction(+ : res2, b2) schedule(static)
        for (int i = 0; i < N; ++i)
        {
            const double bi = P.b.empty() ? 1.0 : P.b[i];
            double d = spmv_row(A.rp.get(), A.ci.get(), A.val.get(), h_x, i) - bi;
            res2 += d * d;
            b2 += bi * bi;
        }
        rel_res = std::sqrt(res2) / std::sqrt(b2 > 0.0 ? b2 : 1.0);
        res_src = "CPU check of copied x";
    }

    const double bytes_it = 12.0 * A.nnz + 4.0 * N + 104.0 * N; // same model as the CPU code
    const double ms_it = R.solve_ms / std::max(1, R.iterations);

    std::cout << "Threads:            GPU (" << g.sms << " SMs)" << std::endl;
    std::cout << "Solver:             CG" << (jacobi ? " + Jacobi" : "") << ", CSR"
              << (P.perm.empty() ? "" : " + RCM") << ", GPU [matrix=" << D.mode
              << ", spmv=" << R.spmv << ", graph=" << (R.graph_used ? "on" : "off")
              << ", check_every=" << env_ll("CHECK_EVERY", 16) << "]" << std::endl;
    std::cout << "Right-hand Side:    " << P.rhs_note << std::endl;
    std::cout << "Iterations Taken:   " << R.iterations << std::endl;
    std::cout << "Read Time:          " << P.read_s * 1000.0 << " ms" << std::endl;
    std::cout << "Build Time:         " << P.build_s * 1000.0 << " ms" << std::endl;
    if (D.register_ms > 0.0)
        std::cout << "Pin Time:           " << D.register_ms << " ms" << std::endl;
    if (D.mode == "host")
        std::cout << "Transfer Time:      0 ms  (GPU reads the CSR over C2C every iteration)"
                  << std::endl;
    else
        std::printf("Transfer Time:      %g ms  (%.3f GB at %.1f GB/s, CPU -> GPU)\n", D.h2d_ms,
                    D.bytes / 1e9, D.bytes / (D.h2d_ms * 1e-3) / 1e9);
    std::cout << "Setup Time:         " << R.setup_ms << " ms" << std::endl;
    std::cout << "Solve Time:         " << R.solve_ms << " ms" << std::endl;
    std::cout << "Time per Iteration: " << ms_it << " ms" << std::endl;
    std::printf("Bandwidth Achieved: %.1f GB/s (estimated)\n", bytes_it / (ms_it * 1e-3) / 1e9);
    std::printf("Result Copy:        %g ms  (%.3f GB at %.1f GB/s, GPU -> CPU)\n", R.d2h_ms,
                8.0 * N / 1e9, 8.0 * N / (R.d2h_ms * 1e-3) / 1e9);
    std::cout << "Relative Residual:  " << rel_res << "  (" << res_src << "; GPU says "
              << R.gpu_rel_res << ")" << std::endl;

    if (env_str("SAVE_SOLUTION", "0") != "0")
    {
        fs::path out_dir = env_str("SOLUTION_DIR", ".");
        fs::create_directories(out_dir);
        std::string out_name = (out_dir / (matrix_stem(P.file) + "_gpu_solution.txt")).string();
        std::ofstream out(out_name);
        out << std::setprecision(17);
        for (int i = 0; i < N; ++i) out << i << " " << h_x[P.perm.empty() ? i : P.perm[i]] << "\n";
        std::cout << "Solution saved to:  " << out_name << std::endl;
    }

    CK(cudaFreeHost(h_x));
    if (d_b) CK(cudaFree(d_b));
    free_matrix(D);
}

int main(int argc, char *argv[])
{
    fs::path target = (argc > 1) ? argv[1] : "Matrices";
    std::cout << "Reading matrices from: " << fs::absolute(target) << std::endl;
    std::vector<fs::path> files;
    if (fs::is_regular_file(target)) files.push_back(target);
    else files = find_all_mtx_files(target);
    if (files.empty())
    {
        std::cout << "No .mtx or .mtx.gz files found in: " << target << std::endl;
        return 0;
    }
    std::cout << "Found " << files.size() << " matrix file(s) to process." << std::endl;

    // The host thread sleeps while waiting on the GPU instead of spinning,
    // so its core is free for the next matrix's preparation.
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    GpuContext g;
    CK(cudaGetDevice(&g.dev));
    CK(cudaDeviceGetAttribute(&g.sms, cudaDevAttrMultiProcessorCount, g.dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, g.dev));
    std::cout << "GPU: " << prop.name << ", " << prop.totalGlobalMem / (1 << 30) << " GiB, "
              << g.sms << " SMs" << std::endl;
    CK(cudaStreamCreateWithFlags(&g.st, cudaStreamNonBlocking));
    CKS(cusparseCreate(&g.sp));
    CKS(cusparseSetStream(g.sp, g.st));

    // Pipeline: the CPU prepares file k+1 while the GPU works on file k.
    const bool pipeline = env_str("PIPELINE", "1") != "0";
    std::future<Prepared> next;
    if (pipeline) next = std::async(std::launch::async, prepare_matrix, files[0]);
    for (size_t k = 0; k < files.size(); ++k)
    {
        Prepared cur = pipeline ? next.get() : prepare_matrix(files[k]);
        if (pipeline && k + 1 < files.size())
            next = std::async(std::launch::async, prepare_matrix, files[k + 1]);
        run_matrix(g, cur);
    }

    CKS(cusparseDestroy(g.sp));
    CK(cudaStreamDestroy(g.st));
    return 0;
}
