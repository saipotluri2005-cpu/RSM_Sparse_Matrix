#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

struct FreeDeleter { void operator()(double *p) const { std::free(p); } };
using Buf = std::unique_ptr<double[], FreeDeleter>;

static Buf alloc_buf(size_t n)
{
    void *p = nullptr;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0 || !p)
        throw std::bad_alloc();
    return Buf(static_cast<double *>(p));
}

// Allocate rows*cols doubles and zero them in parallel (NUMA first-touch).
// Uses the SAME static row partition as the matvec.
static Buf alloc_matrix_first_touch(size_t rows, size_t cols)
{
    Buf a = alloc_buf(rows * cols);
    double *ptr = a.get();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)rows; ++i)
        std::memset(ptr + (size_t)i * cols, 0, cols * sizeof(double));
    return a;
}

static inline double row_dot(const double *__restrict row,
                             const double *__restrict v, int N)
{
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (int j = 0; j < N; ++j)
        s += row[j] * v[j];
    return s;
}

// Conjugate Gradient, entire loop inside ONE parallel region.
int solve_dense_cg(const double *A, const double *b, double *x,
                   int N, int max_iters, double tol)
{
    Buf r = alloc_buf(N), p = alloc_buf(N), Ap = alloc_buf(N);
    double *R = r.get(), *P = p.get(), *AP = Ap.get();

    double b_norm2 = 0.0, rs0 = 0.0, pAp = 0.0, rsnew = 0.0;
    int iters = max_iters;

#pragma omp parallel
    {
#pragma omp for reduction(+ : b_norm2) schedule(static)
        for (int i = 0; i < N; ++i)
            b_norm2 += b[i] * b[i];
        double b_norm = std::sqrt(b_norm2);
        if (b_norm == 0.0) b_norm = 1.0;

        // r = b - A*x ; p = r
#pragma omp for reduction(+ : rs0) schedule(static)
        for (int i = 0; i < N; ++i)
        {
            double ri = b[i] - row_dot(A + (size_t)i * N, x, N);
            R[i] = ri;
            P[i] = ri;
            rs0 += ri * ri;
        }
        double rs_old = rs0; // thread-private copy

        for (int it = 0; it < max_iters; ++it)
        {
#pragma omp single
            {
                pAp = 0.0;
                rsnew = 0.0;
            }

            // Pass 1: Ap = A*p, fused with p . Ap
#pragma omp for reduction(+ : pAp) schedule(static)
            for (int i = 0; i < N; ++i)
            {
                double s = row_dot(A + (size_t)i * N, P, N);
                AP[i] = s;
                pAp += P[i] * s;
            }
            double alpha = rs_old / pAp;

            // Pass 2: x += alpha p ; r -= alpha Ap ; rsnew = r . r
#pragma omp for reduction(+ : rsnew) schedule(static)
            for (int i = 0; i < N; ++i)
            {
                x[i] += alpha * P[i];
                double ri = R[i] - alpha * AP[i];
                R[i] = ri;
                rsnew += ri * ri;
            }

            if (std::sqrt(rsnew) / b_norm < tol)
            {
#pragma omp single
                {
                    iters = it + 1;
                }
                break; // all threads see the same rsnew, so all break together
            }

            double beta = rsnew / rs_old;
            rs_old = rsnew;   // read now, before Pass 3's barrier

            // Pass 3: p = r + beta p
#pragma omp for schedule(static)
            for (int i = 0; i < N; ++i)
                P[i] = R[i] + beta * P[i];
        }
    }
    return iters;
}

// ---------------------------------------------------------------------------
// File discovery and reading
// ---------------------------------------------------------------------------
static bool ends_with(const std::string &s, const std::string &suf)
{
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static bool is_gz(const fs::path &p) { return ends_with(p.filename().string(), ".gz"); }

// Accepts both "name.mtx" and "name.mtx.gz"
static bool is_matrix_file(const fs::path &p)
{
    std::string n = p.filename().string();
    return ends_with(n, ".mtx") || ends_with(n, ".mtx.gz");
}

// "west0167.mtx.gz" -> "west0167"
static std::string matrix_stem(const fs::path &p)
{
    std::string n = p.filename().string();
    if (ends_with(n, ".gz"))  n.resize(n.size() - 3);
    if (ends_with(n, ".mtx")) n.resize(n.size() - 4);
    return n;
}

std::vector<fs::path> find_all_mtx_files(const fs::path &folder_path)
{
    std::vector<fs::path> files;
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
    {
        std::cerr << "Error: Folder " << folder_path << " does not exist or is not a directory.\n";
        return files;
    }
    for (const auto &entry : fs::recursive_directory_iterator(folder_path))
        if (entry.is_regular_file() && is_matrix_file(entry.path()))
            files.push_back(entry.path());
    return files;
}

static std::string slurp_plain(const fs::path &path)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::rewind(f);
    std::string s(sz, '\0');
    size_t got = std::fread(&s[0], 1, sz, f);
    (void)got;
    std::fclose(f);
    return s;
}

// Read a .gz file by piping it through `gzip -dc` (no unzipping needed).
static std::string slurp_gz(const fs::path &path)
{
    std::string cmd = "gzip -dc '" + path.string() + "'";
    FILE *f = popen(cmd.c_str(), "r");
    if (!f) return {};
    std::string s;
    char chunk[1 << 16];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0)
        s.append(chunk, n);
    pclose(f);
    return s;
}

static std::string slurp(const fs::path &path)
{
    return is_gz(path) ? slurp_gz(path) : slurp_plain(path);
}

// ---------------------------------------------------------------------------
// One matrix
// ---------------------------------------------------------------------------
void process_matrix_file(const fs::path &file_path)
{
    std::cout << "\n==================================================\n";
    std::cout << "Processing File: " << file_path.string() << "\n";
    std::cout << "==================================================\n";

    std::string buf = slurp(file_path);
    if (buf.empty())
    {
        std::cerr << "Error opening/reading file " << file_path << std::endl;
        return;
    }

    size_t eol = buf.find('\n');
    if (eol == std::string::npos)
    {
        std::cerr << "SKIPPED: no header line.\n";
        return;
    }
    std::string header = buf.substr(0, eol);

    // CG here needs a real, square, sparse (coordinate) matrix.
    if (header.find("coordinate") == std::string::npos ||
        header.find("complex") != std::string::npos)
    {
        std::cerr << "SKIPPED: not a real coordinate-format matrix. Header: " << header << "\n";
        return;
    }
    bool is_symmetric = header.find("symmetric") != std::string::npos;
    bool is_pattern = header.find("pattern") != std::string::npos;

    const char *c = buf.c_str() + eol + 1;
    while (*c == '%') // skip comment lines
    {
        while (*c && *c != '\n') ++c;
        if (*c) ++c;
    }

    char *end;
    long long num_rows = std::strtoll(c, &end, 10); c = end;
    long long num_cols = std::strtoll(c, &end, 10); c = end;
    long long num_entries = std::strtoll(c, &end, 10); c = end;

    if (num_rows != num_cols || num_rows <= 0)
    {
        std::cerr << "SKIPPED: matrix is not square (" << num_rows << " x " << num_cols << ").\n";
        return;
    }
    int N = (int)num_rows;

    double gb = (double)num_rows * num_cols * 8 / 1e9;
    const char *lim = std::getenv("MAX_DENSE_GB");
    double max_gb = lim ? std::atof(lim) : 100.0;
    std::cout << "Allocating Dense Matrix (" << num_rows << " x " << num_cols << ") = "
              << gb << " GB..." << std::endl;
    if (gb > max_gb)
    {
        std::cerr << "SKIPPED: needs " << gb << " GB, above MAX_DENSE_GB=" << max_gb << ".\n";
        return;
    }

    Buf A = alloc_matrix_first_touch((size_t)num_rows, (size_t)num_cols);
    double *Ap_ = A.get();

    for (long long k = 0; k < num_entries; ++k)
    {
        long long r = std::strtoll(c, &end, 10); c = end;
        long long cc = std::strtoll(c, &end, 10); c = end;
        double v = 1.0;
        if (!is_pattern) { v = std::strtod(c, &end); c = end; }
        --r; --cc;
        if (r < 0 || r >= num_rows || cc < 0 || cc >= num_cols)
        {
            std::cerr << "SKIPPED: bad entry index at entry " << k << ".\n";
            return;
        }
        Ap_[(size_t)r * num_cols + cc] = v;
        if (is_symmetric && r != cc)
            Ap_[(size_t)cc * num_cols + r] = v;
    }
    buf.clear(); buf.shrink_to_fit();

    std::vector<double> b(N, 1.0), x(N, 0.0);

    double t0 = omp_get_wtime();
    int iterations = solve_dense_cg(Ap_, b.data(), x.data(), N, 5000, 1e-5);
    double duration_ms = (omp_get_wtime() - t0) * 1000.0;

    // Accuracy check (parallel)
    double res2 = 0.0, b2 = 0.0;
#pragma omp parallel for reduction(+ : res2, b2) schedule(static)
    for (int i = 0; i < N; ++i)
    {
        double d = row_dot(Ap_ + (size_t)i * N, x.data(), N) - b[i];
        res2 += d * d;
        b2 += b[i] * b[i];
    }
    double rel_res = std::sqrt(res2) / std::sqrt(b2);

    std::cout << "Threads:            " << omp_get_max_threads() << std::endl;
    std::cout << "Matrix Size:        " << num_rows << " x " << num_cols << std::endl;
    std::cout << "Iterations Taken:   " << iterations << std::endl;
    std::cout << "Solve Time:         " << duration_ms << " ms" << std::endl;
    std::cout << "Relative Residual:  " << rel_res << std::endl;

    std::string out_name = matrix_stem(file_path) + "_dense_solution.txt";
    std::ofstream out(out_name);
    for (int i = 0; i < N; ++i)
        out << i << " " << x[i] << "\n";
    std::cout << "Solution saved to:  " << out_name << std::endl;
}

int main(int argc, char *argv[])
{
    // Default input: the Matrices folder (run from the repo directory).
    // Pass another folder, or a single .mtx / .mtx.gz file, to override.
    fs::path target = (argc > 1) ? argv[1] : "Matrices";

    std::cout << "Reading matrices from: " << fs::absolute(target) << std::endl;

    std::vector<fs::path> mtx_files;
    if (fs::is_regular_file(target))
        mtx_files.push_back(target);
    else
        mtx_files = find_all_mtx_files(target);

    if (mtx_files.empty())
    {
        std::cout << "No .mtx or .mtx.gz files found in: " << target << std::endl;
        return 0;
    }
    std::cout << "Found " << mtx_files.size() << " matrix file(s) to process." << std::endl;

    for (const auto &f : mtx_files)
        process_matrix_file(f);
    return 0;
}