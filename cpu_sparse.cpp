// cpu_sparse.cpp -- OpenMP Conjugate Gradient on sparse (CSR) matrices.
//
// Same inputs, outputs and command line as cpu_only.cpp (the dense version),
// so benchmark.sh / cg_perf.slurm parse both the same way.
//
// Pipeline (cpu_high_level.txt):
//   1. read .mtx / .mtx.gz into a COO list (row, col, value)
//   2. report the sparsity ratio  nnz / (N*N)
//   3. RCM reordering (Reverse Cuthill-McKee) to pull nonzeros toward the diagonal
//   4. COO -> CSR (only nonzeros are stored)
//   5. CG with a CSR sparse matrix-vector product (SpMV), fused passes,
//      whole iteration loop inside one OpenMP parallel region
//
// Environment variables:
//   REORDER=none        skip RCM                (default: rcm)
//   REORDER=auto        apply RCM only if it shrinks the bandwidth
//   PRECOND=jacobi      Jacobi-preconditioned CG (default: none, plain CG like cpu_only)
//   MAX_ITERS=5000  TOL=1e-5
//   SAVE_SOLUTION=0     do not write <name>_csr_solution.txt
//
// Build:  make cpu_sparse
//
// qa_rcm branch edits (REORDER=auto, actual thread count, full-precision
// solution file): generated with assistance of claude code

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// 64-byte aligned buffers
// ---------------------------------------------------------------------------
struct FreeDeleter { void operator()(void *p) const { std::free(p); } };
template <class T> using Buf = std::unique_ptr<T[], FreeDeleter>;

template <class T> static Buf<T> alloc_buf(size_t n)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, std::max<size_t>(n, 1) * sizeof(T)) != 0 || !p)
        throw std::bad_alloc();
    return Buf<T>(static_cast<T *>(p));
}

static std::string env_str(const char *name, const char *def)
{
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string(def);
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------
struct COO
{
    int n = 0;
    std::vector<int> r, c;
    std::vector<double> v;
    bool symmetric = false;
};

struct CSR
{
    int n = 0;
    long long nnz = 0;
    Buf<long long> rp; // row pointers, n+1
    Buf<int> ci;       // column index per nonzero
    Buf<double> val;   // value per nonzero
};

// (A v)[i] for one row
static inline double spmv_row(const long long *__restrict rp, const int *__restrict ci,
                              const double *__restrict val, const double *__restrict v, int i)
{
    double s = 0.0;
    const long long b = rp[i], e = rp[i + 1];
#pragma omp simd reduction(+ : s)
    for (long long k = b; k < e; ++k)
        s += val[k] * v[ci[k]];
    return s;
}

// ---------------------------------------------------------------------------
// Step 4: COO -> CSR. Expands symmetric storage, applies the permutation
// perm (new index = perm[old], or identity if empty), sorts each row by
// column and sums duplicates. The big arrays are first-touched in parallel
// with the same static row split the solver uses (NUMA).
// ---------------------------------------------------------------------------
static CSR coo_to_csr(const COO &m, const std::vector<int> &perm)
{
    const int n = m.n;
    const size_t E = m.r.size();
    auto P = [&](int i) { return perm.empty() ? i : perm[i]; };

    CSR A;
    A.n = n;
    A.rp = alloc_buf<long long>(n + 1);
    long long *rp = A.rp.get();
    std::fill(rp, rp + n + 1, 0LL);
    for (size_t k = 0; k < E; ++k)
    {
        rp[P(m.r[k]) + 1]++;
        if (m.symmetric && m.r[k] != m.c[k]) rp[P(m.c[k]) + 1]++;
    }
    for (int i = 0; i < n; ++i) rp[i + 1] += rp[i];
    const long long total = rp[n];

    auto ci = alloc_buf<int>(total);
    auto val = alloc_buf<double>(total);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (long long k = rp[i]; k < rp[i + 1]; ++k) { ci[k] = 0; val[k] = 0.0; }

    std::vector<long long> fill(rp, rp + n);
    for (size_t k = 0; k < E; ++k)
    {
        int r = P(m.r[k]), c = P(m.c[k]);
        long long p = fill[r]++;
        ci[p] = c; val[p] = m.v[k];
        if (m.symmetric && m.r[k] != m.c[k]) { p = fill[c]++; ci[p] = r; val[p] = m.v[k]; }
    }

    // sort each row by column, merge duplicate entries
    std::vector<long long> len(n);
#pragma omp parallel
    {
        std::vector<std::pair<int, double>> tmp;
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            tmp.clear();
            for (long long k = rp[i]; k < rp[i + 1]; ++k) tmp.emplace_back(ci[k], val[k]);
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            long long w = rp[i];
            for (const auto &t : tmp)
            {
                if (w > rp[i] && ci[w - 1] == t.first) val[w - 1] += t.second;
                else { ci[w] = t.first; val[w] = t.second; ++w; }
            }
            len[i] = w - rp[i];
        }
    }
    long long nnz = std::accumulate(len.begin(), len.end(), 0LL);
    if (nnz != total) // compact after merging duplicates
    {
        long long w = 0;
        for (int i = 0; i < n; ++i)
        {
            std::memmove(&ci[w], &ci[rp[i]], len[i] * sizeof(int));
            std::memmove(&val[w], &val[rp[i]], len[i] * sizeof(double));
            rp[i] = w;
            w += len[i];
        }
        rp[n] = w;
    }
    A.nnz = nnz;
    A.ci = std::move(ci);
    A.val = std::move(val);
    return A;
}

// Bandwidth = max |i - j| over nonzeros (what RCM tries to shrink)
static long long bandwidth(const CSR &A)
{
    long long bw = 0;
#pragma omp parallel for reduction(max : bw) schedule(static)
    for (int i = 0; i < A.n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            bw = std::max<long long>(bw, std::llabs((long long)A.ci[k] - i));
    return bw;
}

// ---------------------------------------------------------------------------
// Step 3: Reverse Cuthill-McKee on the symmetric pattern of A.
// Returns perm with perm[old] = new. Each connected component is started
// from a pseudo-peripheral node (repeated BFS to the farthest low-degree node).
// ---------------------------------------------------------------------------
static std::vector<int> rcm_order(const CSR &A)
{
    const int n = A.n;
    // symmetric adjacency (pattern of A + A^T, no diagonal)
    std::vector<int> deg(n, 0);
    for (int i = 0; i < n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            if (A.ci[k] != i) { deg[i]++; deg[A.ci[k]]++; }
    std::vector<long long> ap(n + 1, 0);
    for (int i = 0; i < n; ++i) ap[i + 1] = ap[i] + deg[i];
    std::vector<int> adj(ap[n]);
    std::vector<long long> pos(ap.begin(), ap.end() - 1);
    for (int i = 0; i < n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            if (int j = A.ci[k]; j != i) { adj[pos[i]++] = j; adj[pos[j]++] = i; }
    for (int i = 0; i < n; ++i) // dedupe (A + A^T doubles symmetric entries)
    {
        auto b = adj.begin() + ap[i], e = adj.begin() + pos[i];
        std::sort(b, e);
        deg[i] = (int)(std::unique(b, e) - b);
    }

    std::vector<int> order, level(n, -1);
    order.reserve(n);
    std::vector<char> done(n, 0);

    auto bfs_last_level = [&](int s, std::vector<int> &q) {
        // BFS from s over unvisited nodes; returns eccentricity, q holds BFS order
        q.clear(); q.push_back(s); level[s] = 0;
        int ecc = 0;
        for (size_t h = 0; h < q.size(); ++h)
        {
            int u = q[h];
            for (long long k = ap[u]; k < ap[u] + deg[u]; ++k)
            {
                int w = adj[k];
                if (!done[w] && level[w] < 0) { level[w] = level[u] + 1; ecc = level[w]; q.push_back(w); }
            }
        }
        return ecc;
    };

    std::vector<int> q;
    for (int start = 0; start < n; ++start)
    {
        if (done[start]) continue;
        // pseudo-peripheral node: hop to a min-degree node on the last BFS level
        int s = start, ecc = bfs_last_level(s, q);
        for (int tries = 0; tries < 8; ++tries)
        {
            int best = -1;
            for (int u : q)
                if (level[u] == ecc && (best < 0 || deg[u] < deg[best])) best = u;
            for (int u : q) level[u] = -1;
            int e2 = bfs_last_level(best, q);
            if (e2 <= ecc) { for (int u : q) level[u] = -1; break; }
            s = best; ecc = e2;
        }
        for (int u : q) level[u] = -1;

        // Cuthill-McKee BFS, neighbours visited in increasing degree
        size_t head = order.size();
        order.push_back(s); done[s] = 1;
        std::vector<int> nb;
        while (head < order.size())
        {
            int u = order[head++];
            nb.clear();
            for (long long k = ap[u]; k < ap[u] + deg[u]; ++k)
                if (!done[adj[k]]) { nb.push_back(adj[k]); done[adj[k]] = 1; }
            std::sort(nb.begin(), nb.end(), [&](int a, int b) { return deg[a] < deg[b]; });
            order.insert(order.end(), nb.begin(), nb.end());
        }
    }
    std::vector<int> perm(n);
    for (int k = 0; k < n; ++k) perm[order[n - 1 - k]] = k; // reverse
    return perm;
}

// ---------------------------------------------------------------------------
// Step 5: (preconditioned) CG, entire loop inside ONE parallel region.
//   Pass 1: Ap = A p, pAp = p.Ap
//   Pass 2: x += a p, r -= a Ap, z = Minv r, rz = r.z, rr = r.r
//   Pass 3: p = z + beta p
// ---------------------------------------------------------------------------
static int solve_csr_cg(const CSR &A, const double *dinv, const double *b, double *x,
                        int max_iters, double tol)
{
    const int N = A.n;
    const long long *rp = A.rp.get();
    const int *ci = A.ci.get();
    const double *val = A.val.get();
    auto r_ = alloc_buf<double>(N), p_ = alloc_buf<double>(N),
         Ap_ = alloc_buf<double>(N), z_ = alloc_buf<double>(N);
    double *R = r_.get(), *P = p_.get(), *AP = Ap_.get(), *Z = z_.get();

    double b_norm2 = 0.0, rz0 = 0.0, pAp = 0.0, rz_new = 0.0, rr = 0.0;
    int iters = max_iters;

#pragma omp parallel
    {
#pragma omp for reduction(+ : b_norm2) schedule(static)
        for (int i = 0; i < N; ++i)
        {
            b_norm2 += b[i] * b[i];
            AP[i] = 0.0; // first touch
        }
        double b_norm = std::sqrt(b_norm2);
        if (b_norm == 0.0) b_norm = 1.0;

#pragma omp for reduction(+ : rz0) schedule(static)
        for (int i = 0; i < N; ++i)
        {
            double ri = b[i] - spmv_row(rp, ci, val, x, i);
            double zi = dinv ? dinv[i] * ri : ri;
            R[i] = ri; Z[i] = zi; P[i] = zi;
            rz0 += ri * zi;
        }
        double rz_old = rz0;

        for (int it = 0; it < max_iters; ++it)
        {
#pragma omp single
            { pAp = 0.0; rz_new = 0.0; rr = 0.0; }

#pragma omp for reduction(+ : pAp) schedule(static)
            for (int i = 0; i < N; ++i)
            {
                double s = spmv_row(rp, ci, val, P, i);
                AP[i] = s;
                pAp += P[i] * s;
            }
            double alpha = rz_old / pAp;

#pragma omp for reduction(+ : rz_new, rr) schedule(static)
            for (int i = 0; i < N; ++i)
            {
                x[i] += alpha * P[i];
                double ri = R[i] - alpha * AP[i];
                double zi = dinv ? dinv[i] * ri : ri;
                R[i] = ri; Z[i] = zi;
                rz_new += ri * zi;
                rr += ri * ri;
            }

            if (std::sqrt(rr) / b_norm < tol || !std::isfinite(rr))
            {
#pragma omp single
                { iters = it + 1; }
                break; // every thread sees the same rr
            }
            double beta = rz_new / rz_old;
            rz_old = rz_new; // read before pass 3's barrier

#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i)
                P[i] = Z[i] + beta * P[i];
        }
    }
    return iters;
}

// ---------------------------------------------------------------------------
// File discovery and reading (same behaviour as cpu_only.cpp)
// ---------------------------------------------------------------------------
static bool ends_with(const std::string &s, const std::string &suf)
{
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}
static bool is_gz(const fs::path &p) { return ends_with(p.filename().string(), ".gz"); }
static bool is_matrix_file(const fs::path &p)
{
    std::string n = p.filename().string();
    return ends_with(n, ".mtx") || ends_with(n, ".mtx.gz");
}
static std::string matrix_stem(const fs::path &p)
{
    std::string n = p.filename().string();
    if (ends_with(n, ".gz")) n.resize(n.size() - 3);
    if (ends_with(n, ".mtx")) n.resize(n.size() - 4);
    return n;
}

std::vector<fs::path> find_all_mtx_files(const fs::path &folder)
{
    std::vector<fs::path> files;
    if (!fs::exists(folder) || !fs::is_directory(folder))
    {
        std::cerr << "Error: Folder " << folder << " does not exist or is not a directory.\n";
        return files;
    }
    for (const auto &e : fs::recursive_directory_iterator(folder))
        if (e.is_regular_file() && is_matrix_file(e.path())) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    return files;
}

static std::string slurp(const fs::path &path)
{
    std::string s;
    if (is_gz(path))
    {
        std::string cmd = "gzip -dc '" + path.string() + "'";
        FILE *f = popen(cmd.c_str(), "r");
        if (!f) return {};
        char chunk[1 << 16];
        size_t n;
        while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) s.append(chunk, n);
        pclose(f);
        return s;
    }
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::rewind(f);
    s.resize(sz);
    s.resize(std::fread(&s[0], 1, sz, f));
    std::fclose(f);
    return s;
}

// ---------------------------------------------------------------------------
// One matrix
// ---------------------------------------------------------------------------
void process_matrix_file(const fs::path &file_path)
{
    std::cout << "\n==================================================\n";
    std::cout << "Processing File: " << file_path.string() << "\n";
    std::cout << "==================================================\n";

    // ---- Step 1: read into COO ----
    double t_read0 = omp_get_wtime();
    std::string buf = slurp(file_path);
    if (buf.empty()) { std::cerr << "Error opening/reading file " << file_path << std::endl; return; }
    size_t eol = buf.find('\n');
    if (eol == std::string::npos) { std::cerr << "SKIPPED: no header line.\n"; return; }
    std::string header = buf.substr(0, eol);
    if (header.find("coordinate") == std::string::npos || header.find("complex") != std::string::npos)
    {
        std::cerr << "SKIPPED: not a real coordinate-format matrix. Header: " << header << "\n";
        return;
    }
    COO m;
    m.symmetric = header.find("symmetric") != std::string::npos;
    if (header.find("skew-symmetric") != std::string::npos)
    {
        std::cerr << "SKIPPED: skew-symmetric matrix (never SPD).\n";
        return;
    }
    bool is_pattern = header.find("pattern") != std::string::npos;

    const char *c = buf.c_str() + eol + 1;
    while (*c == '%') { while (*c && *c != '\n') ++c; if (*c) ++c; }
    char *end;
    long long num_rows = std::strtoll(c, &end, 10); c = end;
    long long num_cols = std::strtoll(c, &end, 10); c = end;
    long long num_entries = std::strtoll(c, &end, 10); c = end;
    if (num_rows != num_cols || num_rows <= 0)
    {
        std::cerr << "SKIPPED: matrix is not square (" << num_rows << " x " << num_cols << ").\n";
        return;
    }
    m.n = (int)num_rows;
    m.r.resize(num_entries); m.c.resize(num_entries); m.v.resize(num_entries);
    for (long long k = 0; k < num_entries; ++k)
    {
        long long r = std::strtoll(c, &end, 10); c = end;
        long long cc = std::strtoll(c, &end, 10); c = end;
        double v = 1.0;
        if (!is_pattern) { v = std::strtod(c, &end); c = end; }
        if (r < 1 || r > num_rows || cc < 1 || cc > num_cols)
        {
            std::cerr << "SKIPPED: bad entry index at entry " << k << ".\n";
            return;
        }
        m.r[k] = (int)r - 1; m.c[k] = (int)cc - 1; m.v[k] = v;
    }
    buf.clear(); buf.shrink_to_fit();
    double t_read = omp_get_wtime() - t_read0;
    const int N = m.n;
    if (!m.symmetric)
        std::cerr << "WARNING: matrix is not marked symmetric; CG needs SPD and may not converge.\n";

    // ---- Steps 3 + 4: RCM, then CSR ----
    double t_build0 = omp_get_wtime();
    CSR A = coo_to_csr(m, {});
    long long bw_before = bandwidth(A), bw_after = bw_before;
    std::vector<int> perm;
    const std::string reorder = env_str("REORDER", "rcm");
    std::string bw_note = "  (no reordering)";
    if (reorder == "rcm" || reorder == "auto")
    {
        perm = rcm_order(A);
        CSR B = coo_to_csr(m, perm);
        const long long bw_rcm = bandwidth(B);
        if (reorder == "auto" && bw_rcm >= bw_before)
        {
            perm.clear(); // RCM would not shrink the bandwidth: keep the input order
            bw_note = "  (RCM skipped: it would give " + std::to_string(bw_rcm) + ")";
        }
        else
        {
            A = std::move(B);
            bw_after = bw_rcm;
            bw_note = "  (after RCM)";
        }
    }
    { COO tmp; std::swap(tmp, m); } // free the COO list
    double t_build = omp_get_wtime() - t_build0;

    // ---- Step 2: sparsity ----
    double density = (double)A.nnz / ((double)N * N);
    std::cout << "Matrix Size:        " << N << " x " << N << std::endl;
    std::cout << "Nonzeros:           " << A.nnz << "  (" << (double)A.nnz / N << " per row)" << std::endl;
    std::printf("Sparsity:           %.6f%% nonzero  (dense would need %.3f GB, CSR uses %.4f GB)\n",
                density * 100.0, (double)N * N * 8 / 1e9, (12.0 * A.nnz + 8.0 * (N + 1)) / 1e9);
    std::cout << "Bandwidth:          " << bw_before << " -> " << bw_after << bw_note << std::endl;

    // ---- Jacobi preconditioner (optional) ----
    Buf<double> dinv;
    if (env_str("PRECOND", "none") == "jacobi")
    {
        dinv = alloc_buf<double>(N);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            double d = 0.0;
            for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
                if (A.ci[k] == i) { d = A.val[k]; break; }
            dinv[i] = d > 0.0 ? 1.0 / d : 1.0;
        }
    }

    // ---- Step 5: solve ----
    const int max_iters = std::atoi(env_str("MAX_ITERS", "5000").c_str());
    const double tol = std::atof(env_str("TOL", "1e-5").c_str());
    std::vector<double> b(N, 1.0), x(N, 0.0); // b = ones is unchanged by any permutation

    double t0 = omp_get_wtime();
    int iterations = solve_csr_cg(A, dinv.get(), b.data(), x.data(), max_iters, tol);
    double duration_ms = (omp_get_wtime() - t0) * 1000.0;

    double res2 = 0.0, b2 = 0.0;
#pragma omp parallel for reduction(+ : res2, b2) schedule(static)
    for (int i = 0; i < N; ++i)
    {
        double d = spmv_row(A.rp.get(), A.ci.get(), A.val.get(), x.data(), i) - b[i];
        res2 += d * d;
        b2 += b[i] * b[i];
    }
    double rel_res = std::sqrt(res2) / std::sqrt(b2);

    int team = 1; // threads actually created (OMP_THREAD_LIMIT / OMP_DYNAMIC can lower it)
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    std::cout << "Threads:            " << team << std::endl;
    std::cout << "Solver:             CG" << (dinv ? " + Jacobi" : "") << ", CSR"
              << (perm.empty() ? "" : " + RCM") << std::endl;
    std::cout << "Iterations Taken:   " << iterations << std::endl;
    std::cout << "Read Time:          " << t_read * 1000.0 << " ms" << std::endl;
    std::cout << "Build Time:         " << t_build * 1000.0 << " ms" << std::endl;
    std::cout << "Solve Time:         " << duration_ms << " ms" << std::endl;
    std::cout << "Time per Iteration: " << duration_ms / std::max(1, iterations) << " ms" << std::endl;
    std::cout << "Relative Residual:  " << rel_res << std::endl;

    if (env_str("SAVE_SOLUTION", "1") != "0")
    {
        // undo the permutation so the file matches cpu_only's row numbering
        std::string out_name = matrix_stem(file_path) + "_csr_solution.txt";
        std::ofstream out(out_name);
        out << std::setprecision(17); // round-trip precision, so the file can be re-checked
        for (int i = 0; i < N; ++i)
            out << i << " " << x[perm.empty() ? i : perm[i]] << "\n";
        std::cout << "Solution saved to:  " << out_name << std::endl;
    }
}

int main(int argc, char *argv[])
{
    fs::path target = (argc > 1) ? argv[1] : "Matrices";
    std::cout << "Reading matrices from: " << fs::absolute(target) << std::endl;

    std::vector<fs::path> mtx_files;
    if (fs::is_regular_file(target)) mtx_files.push_back(target);
    else mtx_files = find_all_mtx_files(target);
    if (mtx_files.empty())
    {
        std::cout << "No .mtx or .mtx.gz files found in: " << target << std::endl;
        return 0;
    }
    std::cout << "Found " << mtx_files.size() << " matrix file(s) to process." << std::endl;
    for (const auto &f : mtx_files) process_matrix_file(f);
    return 0;
}
