// cpu_sparse_v2.cpp -- CSR Conjugate Gradient with the scaling fixes from
// docs/RESULTS.md section 3. Same input, output and command line as
// cpu_sparse.cpp (v1), so benchmark.sh and plot_compare.py work unchanged.
//
// Fixes over v1, each switchable so its effect can be measured alone:
//
//   SYNC        (always on) v1 synchronizes 5 times per iteration: three
//               OpenMP reductions, a `single` and a barrier, and every
//               reduction ends with all threads combining into one shared
//               variable. Those variables (pAp, rz_new, rr, b_norm2) sit on
//               one 64-byte cache line, which moves between cores each time
//               (true sharing). v2 gives each thread its own 64-byte slot per
//               sum, uses 3 plain barriers per iteration (the minimum for CG),
//               and every thread adds up the slots itself in the same order.
//
//   FIRST_TOUCH=1  (default 1) v1 initialises x, b and the row pointers on
//               one thread, so their memory lives on socket 0. v2 copies the
//               whole matrix and all vectors with the solver's own row split,
//               so each page sits on the socket of the thread that uses it.
//
//   BALANCE=nnz    (default nnz) split rows so every thread gets the same
//               number of nonzeros; BALANCE=rows splits by row count like v1.
//
//   THREAD_GRAIN=N (default 0 = off) use at most one thread per N nonzeros,
//               so small matrices do not pay 144-thread synchronization for a
//               few microseconds of work. Example: THREAD_GRAIN=16384.
//
// Thread placement is chosen at launch, not in the code:
//   OMP_PLACES=cores OMP_PROC_BIND=spread   uses both sockets from the start.
//
// Other options as in v1: REORDER=none, PRECOND=jacobi, MAX_ITERS, TOL,
// SAVE_SOLUTION=0.
//
// Build:  make cpu_sparse_v2

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Buffers and options
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
static long long env_ll(const char *name, long long def)
{
    const char *v = std::getenv(name);
    return v ? std::atoll(v) : def;
}

// One partial sum per thread, alone on its 64-byte cache line.
struct alignas(64) Slot { double v; char pad[64 - sizeof(double)]; };

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
    Buf<long long> rp;
    Buf<int> ci;
    Buf<double> val;
};

// Row split: thread t owns rows [start[t], start[t+1]).
struct Partition
{
    int T = 1;
    std::vector<int> start;
};

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

// COO -> CSR (expand symmetric storage, apply perm, sort rows, merge duplicates).
// Same as v1; the result is relocated afterwards for first-touch placement.
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
    // same placement as v1 (static row split), so FIRST_TOUCH=0 matches v1
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
    if (nnz != total)
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

static long long bandwidth(const CSR &A)
{
    long long bw = 0;
#pragma omp parallel for reduction(max : bw) schedule(static)
    for (int i = 0; i < A.n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            bw = std::max<long long>(bw, std::llabs((long long)A.ci[k] - i));
    return bw;
}

// Reverse Cuthill-McKee (same as v1). perm[old] = new.
static std::vector<int> rcm_order(const CSR &A)
{
    const int n = A.n;
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
    for (int i = 0; i < n; ++i)
    {
        auto b = adj.begin() + ap[i], e = adj.begin() + pos[i];
        std::sort(b, e);
        deg[i] = (int)(std::unique(b, e) - b);
    }

    std::vector<int> order, level(n, -1), q, nb;
    order.reserve(n);
    std::vector<char> done(n, 0);
    auto bfs = [&](int s) {
        q.clear(); q.push_back(s); level[s] = 0;
        int ecc = 0;
        for (size_t h = 0; h < q.size(); ++h)
        {
            int u = q[h];
            for (long long k = ap[u]; k < ap[u] + deg[u]; ++k)
                if (int w = adj[k]; !done[w] && level[w] < 0)
                { level[w] = level[u] + 1; ecc = level[w]; q.push_back(w); }
        }
        return ecc;
    };
    for (int start = 0; start < n; ++start)
    {
        if (done[start]) continue;
        int s = start, ecc = bfs(s);
        for (int tries = 0; tries < 8; ++tries)
        {
            int best = -1;
            for (int u : q)
                if (level[u] == ecc && (best < 0 || deg[u] < deg[best])) best = u;
            for (int u : q) level[u] = -1;
            int e2 = bfs(best);
            if (e2 <= ecc) break;
            s = best; ecc = e2;
        }
        for (int u : q) level[u] = -1;

        size_t head = order.size();
        order.push_back(s); done[s] = 1;
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
    for (int k = 0; k < n; ++k) perm[order[n - 1 - k]] = k;
    return perm;
}

// ---------------------------------------------------------------------------
// Fix: row partition (by nonzeros or by rows) and thread count
// ---------------------------------------------------------------------------
static Partition make_partition(const CSR &A, int T, bool by_nnz)
{
    Partition p;
    p.T = T;
    p.start.assign(T + 1, 0);
    p.start[T] = A.n;
    for (int t = 1; t < T; ++t)
    {
        if (by_nnz)
        {
            long long target = (long long)((double)A.nnz * t / T);
            p.start[t] = (int)(std::lower_bound(A.rp.get(), A.rp.get() + A.n + 1, target) - A.rp.get());
        }
        else
            p.start[t] = (int)((long long)A.n * t / T);
        p.start[t] = std::clamp(p.start[t], p.start[t - 1], A.n);
    }
    return p;
}

// Fix: first touch. Copy the matrix so each thread's rows are written first
// by that thread (same team size and binding as the solver, so same cores).
static CSR relocate(const CSR &A, const Partition &p)
{
    CSR B;
    B.n = A.n;
    B.nnz = A.nnz;
    B.rp = alloc_buf<long long>(A.n + 1);
    B.ci = alloc_buf<int>(A.nnz);
    B.val = alloc_buf<double>(A.nnz);
#pragma omp parallel num_threads(p.T)
    {
        const int t = omp_get_thread_num();
        const int lo = p.start[t], hi = p.start[t + 1];
        for (int i = lo; i < hi; ++i) B.rp[i] = A.rp[i];
        const long long klo = A.rp[lo], khi = A.rp[hi];
        std::memcpy(&B.ci[klo], &A.ci[klo], (khi - klo) * sizeof(int));
        std::memcpy(&B.val[klo], &A.val[klo], (khi - klo) * sizeof(double));
    }
    B.rp[A.n] = A.rp[A.n];
    return B;
}

// ---------------------------------------------------------------------------
// Fix: CG with per-thread padded partial sums and 3 barriers per iteration.
//   Pass 1: Ap = A p, partial p.Ap                        -> barrier B1
//   every thread sums the p.Ap slots -> alpha
//   Pass 2: x += a p, r -= a Ap, z = Minv r, partial r.z, r.r -> barrier B2
//   every thread sums the r.z and r.r slots -> beta / convergence test
//   Pass 3: p = z + beta p                                 -> barrier B3
// Slots are written only by their owner, and each kind of slot is next
// written one phase after every thread has finished reading it, so plain
// barriers are enough. Every thread adds the slots in the same order, so all
// threads get the same value and take the same branch.
// ---------------------------------------------------------------------------
static int solve_cg_v2(const CSR &A, const Partition &part, const double *dinv,
                       const double *b, double *x, double *R, double *P, double *AP,
                       double *Z, int max_iters, double tol)
{
    const int T = part.T;
    const long long *rp = A.rp.get();
    const int *ci = A.ci.get();
    const double *val = A.val.get();

    // separate arrays per quantity, so writing one sum never touches a line
    // another thread is reading for a different sum
    std::vector<Slot> s_pap(T), s_rz(T), s_rr(T), s_bb(T);
    int iters = max_iters;

    auto total = [T](const std::vector<Slot> &s) {
        double sum = 0.0;
        for (int t = 0; t < T; ++t) sum += s[t].v;
        return sum;
    };

#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int lo = part.start[t], hi = part.start[t + 1];

        double bb = 0.0, rz0 = 0.0;
        for (int i = lo; i < hi; ++i)
        {
            double ri = b[i] - spmv_row(rp, ci, val, x, i);
            double zi = dinv ? dinv[i] * ri : ri;
            R[i] = ri; Z[i] = zi; P[i] = zi; AP[i] = 0.0;
            bb += b[i] * b[i];
            rz0 += ri * zi;
        }
        s_bb[t].v = bb;
        s_rz[t].v = rz0;
#pragma omp barrier
        double b_norm = std::sqrt(total(s_bb));
        if (b_norm == 0.0) b_norm = 1.0;
        double rz_old = total(s_rz);

        for (int it = 0; it < max_iters; ++it)
        {
            // Pass 1
            double pap = 0.0;
            for (int i = lo; i < hi; ++i)
            {
                double s = spmv_row(rp, ci, val, P, i);
                AP[i] = s;
                pap += P[i] * s;
            }
            s_pap[t].v = pap;
#pragma omp barrier
            const double alpha = rz_old / total(s_pap);

            // Pass 2
            double rz = 0.0, rr = 0.0;
            for (int i = lo; i < hi; ++i)
            {
                x[i] += alpha * P[i];
                double ri = R[i] - alpha * AP[i];
                double zi = dinv ? dinv[i] * ri : ri;
                R[i] = ri; Z[i] = zi;
                rz += ri * zi;
                rr += ri * ri;
            }
            s_rz[t].v = rz;
            s_rr[t].v = rr;
#pragma omp barrier
            const double rz_new = total(s_rz);
            const double rr_all = total(s_rr);

            if (std::sqrt(rr_all) / b_norm < tol || !std::isfinite(rr_all))
            {
                if (t == 0) iters = it + 1;
                break; // every thread computed the same rr_all
            }
            const double beta = rz_new / rz_old;
            rz_old = rz_new;

            // Pass 3
            for (int i = lo; i < hi; ++i)
                P[i] = Z[i] + beta * P[i];
#pragma omp barrier
        }
    }
    return iters;
}

// ---------------------------------------------------------------------------
// File reading (same as v1)
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
    if (header.find("skew-symmetric") != std::string::npos)
    {
        std::cerr << "SKIPPED: skew-symmetric matrix (never SPD).\n";
        return;
    }
    COO m;
    m.symmetric = header.find("symmetric") != std::string::npos;
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

    // ---- build CSR, RCM ----
    double t_build0 = omp_get_wtime();
    CSR A = coo_to_csr(m, {});
    long long bw_before = bandwidth(A), bw_after = bw_before;
    std::vector<int> perm;
    if (env_str("REORDER", "rcm") == "rcm")
    {
        perm = rcm_order(A);
        A = coo_to_csr(m, perm);
        bw_after = bandwidth(A);
    }
    { COO tmp; std::swap(tmp, m); }

    // ---- threads and partition ----
    const int requested = omp_get_max_threads();
    const long long grain = env_ll("THREAD_GRAIN", 0);
    int T = requested;
    if (grain > 0) T = (int)std::clamp<long long>(A.nnz / grain, 1, requested);
    T = std::min(T, std::max(1, N));
    const bool by_nnz = env_str("BALANCE", "nnz") != "rows";
    const bool first_touch = env_str("FIRST_TOUCH", "1") != "0";
    Partition part = make_partition(A, T, by_nnz);
    if (first_touch) A = relocate(A, part);
    double t_build = omp_get_wtime() - t_build0;

    double density = (double)A.nnz / ((double)N * N);
    std::cout << "Matrix Size:        " << N << " x " << N << std::endl;
    std::cout << "Nonzeros:           " << A.nnz << "  (" << (double)A.nnz / N << " per row)" << std::endl;
    std::printf("Sparsity:           %.6f%% nonzero  (dense would need %.3f GB, CSR uses %.4f GB)\n",
                density * 100.0, (double)N * N * 8 / 1e9, (12.0 * A.nnz + 8.0 * (N + 1)) / 1e9);
    std::cout << "Bandwidth:          " << bw_before << " -> " << bw_after
              << (perm.empty() ? "  (no reordering)" : "  (after RCM)") << std::endl;

    // ---- vectors: placed by the owning thread when FIRST_TOUCH=1 ----
    auto b_ = alloc_buf<double>(N), x_ = alloc_buf<double>(N), r_ = alloc_buf<double>(N),
         p_ = alloc_buf<double>(N), ap_ = alloc_buf<double>(N), z_ = alloc_buf<double>(N);
    Buf<double> dinv;
    const bool jacobi = env_str("PRECOND", "none") == "jacobi";
    if (jacobi) dinv = alloc_buf<double>(N);
    auto init_rows = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i)
        {
            b_[i] = 1.0; x_[i] = 0.0; r_[i] = 0.0; p_[i] = 0.0; ap_[i] = 0.0; z_[i] = 0.0;
            if (jacobi)
            {
                double d = 0.0;
                for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
                    if (A.ci[k] == i) { d = A.val[k]; break; }
                dinv[i] = d > 0.0 ? 1.0 / d : 1.0;
            }
        }
    };
    if (first_touch)
    {
#pragma omp parallel num_threads(T)
        init_rows(part.start[omp_get_thread_num()], part.start[omp_get_thread_num() + 1]);
    }
    else
        init_rows(0, N);

    // ---- solve ----
    const int max_iters = std::atoi(env_str("MAX_ITERS", "5000").c_str());
    const double tol = std::atof(env_str("TOL", "1e-5").c_str());
    double t0 = omp_get_wtime();
    int iterations = solve_cg_v2(A, part, dinv.get(), b_.get(), x_.get(), r_.get(), p_.get(),
                                 ap_.get(), z_.get(), max_iters, tol);
    double duration_ms = (omp_get_wtime() - t0) * 1000.0;

    double res2 = 0.0, b2 = 0.0;
#pragma omp parallel for reduction(+ : res2, b2) schedule(static) num_threads(T)
    for (int i = 0; i < N; ++i)
    {
        double d = spmv_row(A.rp.get(), A.ci.get(), A.val.get(), x_.get(), i) - b_[i];
        res2 += d * d;
        b2 += b_[i] * b_[i];
    }
    double rel_res = std::sqrt(res2) / std::sqrt(b2);

    // per-iteration memory traffic: matrix (12 B/nnz) + row pointers + ~13 vector streams
    const double bytes_it = 12.0 * A.nnz + 8.0 * N + 104.0 * N;
    const double ms_it = duration_ms / std::max(1, iterations);

    std::cout << "Threads:            " << T;
    if (T != requested) std::cout << "  (requested " << requested << ", THREAD_GRAIN=" << grain << ")";
    std::cout << std::endl;
    std::cout << "Solver:             CG" << (jacobi ? " + Jacobi" : "") << ", CSR"
              << (perm.empty() ? "" : " + RCM") << ", v2 [sync=padded"
              << ", first_touch=" << (first_touch ? "on" : "off")
              << ", balance=" << (by_nnz ? "nnz" : "rows") << "]" << std::endl;
    std::cout << "Iterations Taken:   " << iterations << std::endl;
    std::cout << "Read Time:          " << t_read * 1000.0 << " ms" << std::endl;
    std::cout << "Build Time:         " << t_build * 1000.0 << " ms" << std::endl;
    std::cout << "Solve Time:         " << duration_ms << " ms" << std::endl;
    std::cout << "Time per Iteration: " << ms_it << " ms" << std::endl;
    std::printf("Bandwidth Achieved: %.1f GB/s (estimated)\n", bytes_it / (ms_it * 1e-3) / 1e9);
    std::cout << "Relative Residual:  " << rel_res << std::endl;

    if (env_str("SAVE_SOLUTION", "1") != "0")
    {
        std::string out_name = matrix_stem(file_path) + "_csr_v2_solution.txt";
        std::ofstream out(out_name);
        for (int i = 0; i < N; ++i)
            out << i << " " << x_[perm.empty() ? i : perm[i]] << "\n";
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
