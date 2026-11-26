// gpu_cg_logic.hpp -- the scalar part of one CG iteration, shared by the GPU
// kernels (gpu_cg.cu) and the host-side test (test_gpu_logic.cpp).
//
// All CG scalars live in GPU memory, so the CPU never has to read a dot
// product to compute alpha or beta. Reduction kernels add their partial sums
// into the three accumulators; cg_after_update() then runs on one GPU thread.
//
// Generated with assistance of claude code
#pragma once
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDA__)
#define CG_HD __host__ __device__
#else
#define CG_HD
#endif

struct CGScalars
{
    double pap;    // accumulator: p . Ap           (summed by k_dot_pap)
    double rz_new; // accumulator: r . z            (summed by k_update1)
    double rr;     // accumulator: r . r            (summed by k_update1)
    double rz;     // r . z of the current iterate
    double bb;     // b . b
    double beta;   // rz_new / rz, read by k_update2
    int it;        // iterations completed
    int done;      // 0 = running, 1 = converged (or non-finite), 2 = hit max_iters
};

// alpha is computed by every thread of k_update1 from these two values.
CG_HD inline double cg_alpha(const CGScalars &s) { return s.rz / s.pap; }

// Runs after k_update1's reductions. Exactly the stopping test of cpu_sparse_v2:
//   sqrt(rr) / ||b|| < tol, or rr not finite   (||b|| treated as 1 when b = 0)
CG_HD inline void cg_after_update(CGScalars &s, double tol, int max_iters)
{
    if (s.done) return;
    s.it += 1;
    double b_norm = sqrt(s.bb);
    if (b_norm == 0.0) b_norm = 1.0;
    const bool finite = s.rr <= 1.7976931348623157e308; // false for NaN and +inf
    if (!finite || sqrt(s.rr) / b_norm < tol)
        s.done = 1;
    else if (s.it >= max_iters)
        s.done = 2;
    else
    {
        s.beta = s.rz_new / s.rz;
        s.rz = s.rz_new;
    }
    // reset the accumulators for the next iteration
    s.pap = 0.0;
    s.rz_new = 0.0;
    s.rr = 0.0;
}
