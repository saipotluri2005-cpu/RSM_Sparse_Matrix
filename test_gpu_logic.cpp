// test_gpu_logic.cpp -- checks the GPU solver's logic on the CPU (no GPU needed).
//
//   g++ -std=c++17 -O2 -fopenmp test_gpu_logic.cpp -o test_gpu_logic && ./test_gpu_logic
//
// 1. prepare_matrix() (shared with gpu_cg) reads a generated FEM-style matrix.
// 2. The GPU's kernel sequence is replayed serially, including the parts that
//    are easy to get wrong: the device-side done flag, chunked launches, the
//    speculative extra chunk, and MAX_ITERS. Its iteration count and x must
//    match a direct port of cpu_sparse_v2's CG.
// 3. The fused SpMV kernel's thread -> row mapping is replayed for every G and
//    several grid sizes: each row must be computed exactly once.
//
// Generated with assistance of claude code
#include "sparse_prep.hpp"
#include "gpu_cg_logic.hpp"

static int failures = 0;
#define EXPECT(cond, msg)                                                   \
    do {                                                                    \
        if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; }        \
        else std::printf("ok:   %s\n", msg);                                \
    } while (0)

// 2D 5-point Laplacian (+ optional varying diagonal), symmetric lower triangle
static void write_poisson(const std::string &path, int m, bool vary_diag)
{
    std::ofstream f(path);
    const int n = m * m;
    std::vector<std::string> lines;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
        {
            const int r = i * m + j;
            const double d = 4.0 + (vary_diag ? 3.0 * ((r * 7919) % 11) / 10.0 : 0.0);
            lines.push_back(std::to_string(r + 1) + " " + std::to_string(r + 1) + " " +
                            std::to_string(d));
            if (j > 0) lines.push_back(std::to_string(r + 1) + " " + std::to_string(r) + " -1");
            if (i > 0) lines.push_back(std::to_string(r + 1) + " " + std::to_string(r - m + 1) + " -1");
        }
    f << "%%MatrixMarket matrix coordinate real symmetric\n";
    f << n << " " << n << " " << lines.size() << "\n";
    for (auto &l : lines) f << l << "\n";
}

static void spmv(const CSR &A, const double *x, double *y)
{
    for (int i = 0; i < A.n; ++i) y[i] = spmv_row(A.rp.get(), A.ci.get(), A.val.get(), x, i);
}

static std::vector<double> make_dinv(const CSR &A)
{
    std::vector<double> d(A.n, 1.0);
    for (int i = 0; i < A.n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            if (A.ci[k] == i) { d[i] = A.val[k] > 0.0 ? 1.0 / A.val[k] : 1.0; break; }
    return d;
}

// Direct serial port of solve_cg_v2 (one thread).
static int cpu_cg(const CSR &A, const double *dinv, std::vector<double> &x, int max_iters, double tol)
{
    const int n = A.n;
    std::vector<double> r(n), z(n), p(n), ap(n);
    x.assign(n, 0.0);
    double bb = 0.0, rz_old = 0.0;
    for (int i = 0; i < n; ++i)
    {
        r[i] = 1.0; z[i] = dinv ? dinv[i] : 1.0; p[i] = z[i];
        bb += 1.0; rz_old += r[i] * z[i];
    }
    double b_norm = std::sqrt(bb);
    for (int it = 0; it < max_iters; ++it)
    {
        spmv(A, p.data(), ap.data());
        double pap = 0.0;
        for (int i = 0; i < n; ++i) pap += p[i] * ap[i];
        const double alpha = rz_old / pap;
        double rz = 0.0, rr = 0.0;
        for (int i = 0; i < n; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = dinv ? dinv[i] * r[i] : r[i];
            rz += r[i] * z[i];
            rr += r[i] * r[i];
        }
        if (std::sqrt(rr) / b_norm < tol || !std::isfinite(rr)) return it + 1;
        const double beta = rz / rz_old;
        rz_old = rz;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    return max_iters;
}

// Serial replay of gpu_cg.cu's kernels and host loop.
static int gpu_replay(const CSR &A, const double *dinv, std::vector<double> &x, int max_iters,
                      double tol, int check_every, int *iterations_launched, int *chunks)
{
    const bool PRE = dinv != nullptr;
    const int n = A.n;
    std::vector<double> r(n), z(n), p(n), ap(n);
    x.assign(n, 0.0);
    CGScalars s{};
    // k_init
    for (int i = 0; i < n; ++i)
    {
        r[i] = 1.0;
        const double zi = PRE ? dinv[i] : 1.0;
        p[i] = zi;
        if (PRE) z[i] = zi;
        s.rz += 1.0 * zi;
        s.bb += 1.0;
    }
    int launched = 0;
    auto iteration = [&]() {
        ++launched;
        // SpMV runs regardless (cuSPARSE does not see the flag)
        spmv(A, p.data(), ap.data());
        if (!s.done) // k_dot_pap / fused
            for (int i = 0; i < n; ++i) s.pap += p[i] * ap[i];
        if (!s.done) // k_update1
        {
            const double alpha = cg_alpha(s);
            for (int i = 0; i < n; ++i)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
                s.rr += r[i] * r[i];
                if (PRE) { z[i] = dinv[i] * r[i]; s.rz_new += r[i] * z[i]; }
            }
        }
        if (!PRE && !s.done) s.rz_new = s.rr; // k_scalar
        cg_after_update(s, tol, max_iters);
        if (!s.done) // k_update2
            for (int i = 0; i < n; ++i) p[i] = (PRE ? z[i] : r[i]) + s.beta * p[i];
    };
    // host loop: chunk k+1 is launched before chunk k's status is read
    CGScalars h_s[2];
    auto launch_chunk = [&](int slot) {
        for (int k = 0; k < check_every; ++k) iteration();
        h_s[slot] = s;
    };
    launch_chunk(0);
    int chunk = 0;
    while (true)
    {
        const int cur = chunk & 1;
        launch_chunk(cur ^ 1);
        ++chunk;
        if (h_s[cur].done) break;
        if (chunk > 100000) { std::printf("replay did not stop\n"); return -1; }
    }
    *iterations_launched = launched;
    *chunks = chunk + 1;
    return s.it;
}

static double rel_residual(const CSR &A, const std::vector<double> &x)
{
    std::vector<double> ax(A.n);
    spmv(A, x.data(), ax.data());
    double r2 = 0.0;
    for (int i = 0; i < A.n; ++i) r2 += (1.0 - ax[i]) * (1.0 - ax[i]);
    return std::sqrt(r2 / A.n);
}

// Replays k_spmv_pap's index math for one grid; counts how often each row is hit.
static bool spmv_mapping_ok(int n, int G, int blocks, int block)
{
    std::vector<int> hits(n, 0);
    const int GPW = 32 / G;
    const long long warps = (long long)blocks * block / 32;
    for (long long tid = 0; tid < (long long)blocks * block; ++tid)
    {
        const int tx = (int)(tid % block);
        const int lane = tx & (G - 1), sub = (tx & 31) / G;
        const long long warp = tid / 32;
        for (long long base = warp * GPW; base < n; base += warps * GPW)
        {
            const long long row = base + sub;
            if (lane == 0 && row < n) hits[row]++;
        }
    }
    for (int h : hits) if (h != 1) return false;
    return true;
}

int main()
{
    std::printf("== matrix preparation (shared with gpu_cg)\n");
    fs::create_directories("test_tmp");
    write_poisson("test_tmp/poisson_40.mtx", 40, false);
    write_poisson("test_tmp/poisson_var_30.mtx", 30, true);

    for (const char *reorder : {"rcm", "none"})
    {
        setenv("REORDER", reorder, 1);
        Prepared P = prepare_matrix("test_tmp/poisson_40.mtx");
        EXPECT(P.skip_reason.empty(), (std::string("prepare_matrix ok, REORDER=") + reorder).c_str());
        EXPECT(P.A.n == 1600 && P.A.nnz == 1600 + 2 * 2 * 40 * 39,
               "size and nonzeros (symmetric expanded)");
        EXPECT(P.bw_after <= P.bw_before, "bandwidth not increased");
    }
    setenv("REORDER", "rcm", 1);

    std::printf("== GPU kernel sequence vs CPU CG\n");
    struct Case { const char *file; bool jacobi; int max_iters; double tol; int check_every; };
    Case cases[] = {
        {"test_tmp/poisson_40.mtx", false, 5000, 1e-5, 16},
        {"test_tmp/poisson_40.mtx", false, 5000, 1e-5, 1},
        {"test_tmp/poisson_40.mtx", false, 5000, 1e-8, 7},
        {"test_tmp/poisson_40.mtx", false, 10, 1e-12, 16},  // stops on MAX_ITERS
        {"test_tmp/poisson_40.mtx", false, 33, 1e-12, 16},  // MAX_ITERS mid-chunk
        {"test_tmp/poisson_var_30.mtx", true, 5000, 1e-5, 16},
        {"test_tmp/poisson_var_30.mtx", false, 5000, 1e-5, 16},
        {"test_tmp/poisson_var_30.mtx", true, 5000, 1e-10, 5},
    };
    for (const Case &c : cases)
    {
        Prepared P = prepare_matrix(c.file);
        std::vector<double> dinv = make_dinv(P.A);
        const double *d = c.jacobi ? dinv.data() : nullptr;
        std::vector<double> xc, xg;
        const int ic = cpu_cg(P.A, d, xc, c.max_iters, c.tol);
        int launched = 0, chunks = 0;
        const int ig = gpu_replay(P.A, d, xg, c.max_iters, c.tol, c.check_every, &launched, &chunks);
        double maxdiff = 0.0;
        for (int i = 0; i < P.A.n; ++i) maxdiff = std::max(maxdiff, std::fabs(xc[i] - xg[i]));
        char msg[256];
        std::snprintf(msg, sizeof msg,
                      "%s jacobi=%d max_iters=%d tol=%g check=%d: CPU %d it, GPU %d it "
                      "(%d launched, %d chunks), max|dx|=%.1e, res=%.2e",
                      fs::path(c.file).filename().c_str(), c.jacobi, c.max_iters, c.tol,
                      c.check_every, ic, ig, launched, chunks, maxdiff, rel_residual(P.A, xg));
        EXPECT(ic == ig && maxdiff == 0.0, msg);
        EXPECT(launched - ig < 2 * c.check_every + 1, "  wasted launches < 2 chunks");
    }

    std::printf("== fused SpMV thread->row mapping\n");
    bool all = true;
    for (int G : {2, 4, 8, 16, 32})
        for (int n : {1, 7, 31, 32, 33, 1000, 4097})
            for (int blocks : {1, 3, 64})
                all = all && spmv_mapping_ok(n, G, blocks, 256);
    EXPECT(all, "every row computed exactly once (G=2..32, 7 sizes, 3 grids)");

    fs::remove_all("test_tmp");
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
