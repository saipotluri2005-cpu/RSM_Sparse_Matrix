// sparse_prep.hpp -- matrix preparation shared by the GPU solver (gpu_cg.cu).
//
// Reading Matrix Market files, COO -> CSR, RCM reordering and bandwidth,
// copied unchanged from cpu_sparse_v2.cpp (qa_rcm branch) so the CPU and GPU
// programs prepare every matrix the same way. prepare_matrix() at the end
// wraps the steps cpu_sparse_v2 runs before its solve.
//
// Generated with assistance of claude code
#pragma once

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
    if (ends_with(n, ".gz")) n.resize(n.size() - 3);
    if (!ends_with(n, ".mtx")) return false;
    // companion files are not matrices: right-hand sides and node coordinates
    return !ends_with(n, "_b.mtx") && !ends_with(n, "_coord.mtx");
}
static std::string matrix_stem(const fs::path &p)
{
    std::string n = p.filename().string();
    if (ends_with(n, ".gz")) n.resize(n.size() - 3);
    if (ends_with(n, ".mtx")) n.resize(n.size() - 4);
    return n;
}
static std::vector<fs::path> find_all_mtx_files(const fs::path &folder)
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
// Right-hand side: <stem>_b.mtx (or .mtx.gz) next to the matrix, the SuiteSparse
// convention. Matrix Market "array" (n x 1) or "coordinate" format.
// Returns an empty vector and sets err when it is missing or malformed.
// ---------------------------------------------------------------------------
static std::vector<double> read_rhs(const fs::path &matrix_path, int n, std::string &err, fs::path &used)
{
    const fs::path dir = matrix_path.parent_path();
    const std::string stem = matrix_stem(matrix_path);
    used = dir / (stem + "_b.mtx");
    if (!fs::exists(used)) used = dir / (stem + "_b.mtx.gz");
    if (!fs::exists(used)) { err = "RHS=file but " + stem + "_b.mtx not found"; return {}; }
    std::string buf = slurp(used);
    size_t eol = buf.find('\n');
    if (buf.empty() || eol == std::string::npos) { err = "cannot read " + used.string(); return {}; }
    const bool array = buf.substr(0, eol).find("array") != std::string::npos;
    const char *c = buf.c_str() + eol + 1;
    while (*c == '%') { while (*c && *c != '\n') ++c; if (*c) ++c; }
    char *end;
    long long rows = std::strtoll(c, &end, 10); c = end;
    long long cols = std::strtoll(c, &end, 10); c = end;
    if (rows != n || cols != 1) { err = used.filename().string() + " is not " + std::to_string(n) + " x 1"; return {}; }
    std::vector<double> b(n, 0.0);
    if (array)
        for (int i = 0; i < n; ++i) { b[i] = std::strtod(c, &end); if (end == c) { err = "short " + used.string(); return {}; } c = end; }
    else
    {
        long long cnt = std::strtoll(c, &end, 10); c = end;
        for (long long k = 0; k < cnt; ++k)
        {
            long long r = std::strtoll(c, &end, 10); c = end;
            std::strtoll(c, &end, 10); c = end;
            double v = std::strtod(c, &end); c = end;
            if (r < 1 || r > n) { err = "bad row in " + used.string(); return {}; }
            b[r - 1] = v;
        }
    }
    return b;
}

// ---------------------------------------------------------------------------
// prepare_matrix: everything cpu_sparse_v2 does before its solve.
// Returns a ready CSR (RCM-reordered unless REORDER=none) or a skip reason.
// ---------------------------------------------------------------------------
struct Prepared
{
    fs::path file;
    std::string skip_reason;   // empty when the matrix is usable
    bool symmetric = false;
    int n = 0;
    CSR A;
    std::vector<int> perm;     // perm[old] = new; empty when not reordered
    std::vector<double> b;     // right-hand side in the reordered numbering; empty = all ones
    std::string rhs_note = "b = 1";
    long long bw_before = 0, bw_after = 0;
    std::string bw_note;
    double read_s = 0.0, build_s = 0.0;
};

static Prepared prepare_matrix(const fs::path &file_path)
{
    Prepared P;
    P.file = file_path;
    double t_read0 = omp_get_wtime();
    std::string buf = slurp(file_path);
    if (buf.empty()) { P.skip_reason = "Error opening/reading file " + file_path.string(); return P; }
    size_t eol = buf.find('\n');
    if (eol == std::string::npos) { P.skip_reason = "SKIPPED: no header line."; return P; }
    std::string header = buf.substr(0, eol);
    if (header.find("coordinate") == std::string::npos || header.find("complex") != std::string::npos)
    {
        P.skip_reason = "SKIPPED: not a real coordinate-format matrix. Header: " + header;
        return P;
    }
    if (header.find("skew-symmetric") != std::string::npos)
    {
        P.skip_reason = "SKIPPED: skew-symmetric matrix (never SPD).";
        return P;
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
        P.skip_reason = "SKIPPED: matrix is not square (" + std::to_string(num_rows) + " x " +
                        std::to_string(num_cols) + ").";
        return P;
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
            P.skip_reason = "SKIPPED: bad entry index at entry " + std::to_string(k) + ".";
            return P;
        }
        m.r[k] = (int)r - 1; m.c[k] = (int)cc - 1; m.v[k] = v;
    }
    buf.clear(); buf.shrink_to_fit();
    P.read_s = omp_get_wtime() - t_read0;
    P.n = m.n;
    P.symmetric = m.symmetric;

    double t_build0 = omp_get_wtime();
    CSR A = coo_to_csr(m, {});
    P.bw_before = bandwidth(A);
    P.bw_after = P.bw_before;
    const std::string reorder = env_str("REORDER", "rcm");
    P.bw_note = "  (no reordering)";
    if (reorder == "rcm" || reorder == "auto")
    {
        P.perm = rcm_order(A);
        CSR B = coo_to_csr(m, P.perm);
        const long long bw_rcm = bandwidth(B);
        if (reorder == "auto" && bw_rcm >= P.bw_before)
        {
            P.perm.clear();
            P.bw_note = "  (RCM skipped: it would give " + std::to_string(bw_rcm) + ")";
        }
        else
        {
            A = std::move(B);
            P.bw_after = bw_rcm;
            P.bw_note = "  (after RCM)";
        }
    }
    P.A = std::move(A);

    // RHS=file: use <stem>_b.mtx instead of b = 1, in the same (reordered) numbering as A
    if (env_str("RHS", "ones") == "file")
    {
        std::string err;
        fs::path used;
        std::vector<double> b = read_rhs(file_path, P.n, err, used);
        if (b.empty()) { P.skip_reason = "SKIPPED: " + err; return P; }
        if (P.perm.empty()) P.b = std::move(b);
        else
        {
            P.b.assign(P.n, 0.0);
            for (int i = 0; i < P.n; ++i) P.b[P.perm[i]] = b[i];
        }
        P.rhs_note = "b from " + used.filename().string();
    }
    P.build_s = omp_get_wtime() - t_build0;
    return P;
}

