// rcm_study.cpp -- compare orderings of sparse matrices.
//
//   rcm_study [options] MATRIX [MATRIX ...]
//
// For every matrix and ordering, reports one CSV row:
//   matrix,N,nnz,ordering,bandwidth,profile,mean_dist,order_ms,spmv_ms,threads,checksum
//
// Orderings (--orderings a,b,...; default: all):
//   original      as stored in the file
//   shuffled      random renumbering (--seed)
//   rcm           Reverse Cuthill-McKee as in cpu_sparse.cpp (pseudo-peripheral start)
//   cm            Cuthill-McKee, not reversed
//   rcm_mindeg    RCM started from the lowest-degree node of each component
//   rcm_first     RCM started from the lowest-numbered node of each component
//   shuffled_rcm  RCM applied to the shuffled matrix
//
// bandwidth = max |i-j|; profile = sum over rows of (i - first column);
// mean_dist = mean |i-j| over nonzeros; order_ms = time to compute the
// permutation; spmv_ms = median time of y = A x over --reps runs with
// OMP_NUM_THREADS threads (static row split); checksum = OK when sum(y)
// matches the original ordering's (same product, only renumbered).
//
// Options:
//   --orderings LIST  orderings to run
//   --seed S          shuffle seed (default 1)
//   --reps R          SpMV repetitions (default 20; 0 skips timing)
//   --csv FILE        write the CSV here instead of stdout
//   --outdir DIR      where --spy and --write files go (default .)
//   --spy G           write a G x G nonzero-count grid per ordering:
//                     DIR/spy_<matrix>_<ordering>.csv
//   --write LIST      write these orderings as DIR/<matrix>_<ordering>.mtx
//
// Generated with assistance of claude code

#include "rcm.hpp"
#include "sparse_io.hpp"

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) out.push_back(t);
    return out;
}

bool contains(const std::vector<std::string> &v, const std::string &s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

std::vector<int> random_perm(int n, unsigned long long seed)
{
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::mt19937_64 rng(seed);
    for (int i = n - 1; i > 0; --i) std::swap(p[i], p[(int)(rng() % (unsigned long long)(i + 1))]);
    return p;
}

// y = A x, static row split like the solvers
void spmv(const sio::Csr &A, const double *x, double *y)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.n; ++i)
    {
        double s = 0.0;
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k) s += A.val[k] * x[A.ci[k]];
        y[i] = s;
    }
}

double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void write_spy(const std::string &path, const sio::Csr &A, int G)
{
    std::vector<long long> grid((size_t)G * G, 0);
    const double scale = (double)G / A.n;
    for (int i = 0; i < A.n; ++i)
    {
        const int gi = std::min(G - 1, (int)(i * scale));
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            grid[(size_t)gi * G + std::min(G - 1, (int)(A.ci[k] * scale))]++;
    }
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    for (int r = 0; r < G; ++r)
        for (int c = 0; c < G; ++c) std::fprintf(f, "%lld%c", grid[(size_t)r * G + c], c + 1 < G ? ',' : '\n');
    std::fclose(f);
}

} // namespace

int main(int argc, char **argv)
{
    const std::vector<std::string> all = {"original", "shuffled", "rcm", "cm",
                                          "rcm_mindeg", "rcm_first", "shuffled_rcm"};
    std::vector<std::string> orderings = all, write_list, files;
    unsigned long long seed = 1;
    int reps = 20, spy = 0;
    std::string csv_path, outdir = ".";
    for (int a = 1; a < argc; ++a)
    {
        std::string s = argv[a];
        auto next = [&]() -> std::string {
            if (a + 1 >= argc) { std::cerr << s << " needs a value\n"; std::exit(2); }
            return argv[++a];
        };
        if (s == "--orderings") orderings = split(next());
        else if (s == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (s == "--reps") reps = std::atoi(next().c_str());
        else if (s == "--csv") csv_path = next();
        else if (s == "--outdir") outdir = next();
        else if (s == "--spy") spy = std::atoi(next().c_str());
        else if (s == "--write") write_list = split(next());
        else if (!s.empty() && s[0] == '-') { std::cerr << "unknown option " << s << "\n"; return 2; }
        else files.push_back(s);
    }
    for (const auto &o : orderings)
        if (!contains(all, o)) { std::cerr << "unknown ordering " << o << "\n"; return 2; }
    if (files.empty())
    {
        std::cerr << "usage: rcm_study [--orderings LIST] [--seed S] [--reps R] [--csv FILE]\n"
                     "                 [--outdir DIR] [--spy G] [--write LIST] MATRIX...\n";
        return 2;
    }

    FILE *csv = csv_path.empty() ? stdout : std::fopen(csv_path.c_str(), "w");
    if (!csv) { std::cerr << "cannot write " << csv_path << "\n"; return 2; }
    std::fprintf(csv, "matrix,N,nnz,ordering,bandwidth,profile,mean_dist,order_ms,spmv_ms,threads,checksum\n");
    std::fflush(csv);
    const int threads = omp_get_max_threads();
    int failures = 0;

    for (const auto &file : files)
    {
        sio::Coo coo;
        try { coo = sio::read_mtx(file); }
        catch (const std::exception &e) { std::cerr << "SKIPPED " << file << ": " << e.what() << "\n"; continue; }
        const std::string name = sio::stem(file);
        const bool symmetric = coo.symmetric;
        sio::Csr A0 = sio::to_csr(coo);
        { sio::Coo tmp; std::swap(tmp, coo); }
        const int n = A0.n;
        std::cerr << name << ": N=" << n << " nnz=" << A0.nnz() << "\n";

        // x in original numbering; each ordering gets the matching relabelled x
        std::vector<double> x0(n), y(n);
        for (int i = 0; i < n; ++i) x0[i] = 1.0 + (i % 7) / 8.0;
        double ref_sum = 0.0;
        {
            spmv(A0, x0.data(), y.data());
            for (double v : y) ref_sum += v;
        }

        std::vector<int> shuf;
        sio::Csr As;
        auto need_shuffled = [&]() -> const sio::Csr & {
            if (shuf.empty()) { shuf = random_perm(n, seed); As = sio::permute(A0, shuf); }
            return As;
        };

        for (const auto &o : orderings)
        {
            std::vector<int> perm; // original index -> new index
            double t0 = sio::now_ms();
            if (o == "original") { perm.resize(n); for (int i = 0; i < n; ++i) perm[i] = i; }
            else if (o == "shuffled") { need_shuffled(); t0 = sio::now_ms(); perm = shuf; }
            else if (o == "rcm") perm = rcm::order(A0);
            else if (o == "cm") perm = rcm::order(A0, {rcm::Start::Peripheral, false});
            else if (o == "rcm_mindeg") perm = rcm::order(A0, {rcm::Start::MinDegree, true});
            else if (o == "rcm_first") perm = rcm::order(A0, {rcm::Start::First, true});
            else // shuffled_rcm: time only the RCM on the shuffled matrix
            {
                const sio::Csr &S = need_shuffled();
                t0 = sio::now_ms();
                std::vector<int> r = rcm::order(S);
                perm.resize(n);
                for (int i = 0; i < n; ++i) perm[i] = r[shuf[i]];
            }
            const double order_ms = sio::now_ms() - t0;
            if (!sio::is_permutation(perm, n))
            {
                std::cerr << "ERROR: " << o << " is not a permutation for " << name << "\n";
                ++failures;
                continue;
            }

            sio::Csr B = (o == "original") ? sio::Csr(A0) : sio::permute(A0, perm);
            const sio::Shape sh = sio::shape(B);

            std::vector<double> x(n);
            for (int i = 0; i < n; ++i) x[perm[i]] = x0[i];
            double spmv_ms = 0.0;
            std::vector<double> times;
            for (int r = 0; r < reps + 2; ++r)
            {
                const double s0 = sio::now_ms();
                spmv(B, x.data(), y.data());
                if (r >= 2) times.push_back(sio::now_ms() - s0);
            }
            spmv_ms = median(times);
            double sum = 0.0;
            for (double v : y) sum += v;
            const bool ok = std::fabs(sum - ref_sum) <= 1e-9 * std::max(1.0, std::fabs(ref_sum));
            if (!ok) ++failures;

            std::fprintf(csv, "%s,%d,%lld,%s,%lld,%lld,%.6g,%.3f,%.6g,%d,%s\n", name.c_str(), n, B.nnz(),
                         o.c_str(), sh.bandwidth, sh.profile, sh.mean_dist, order_ms, spmv_ms, threads,
                         ok ? "OK" : "MISMATCH");
            std::fflush(csv);

            if (spy > 0) write_spy(outdir + "/spy_" + name + "_" + o + ".csv", B, spy);
            if (contains(write_list, o))
                sio::write_csr(outdir + "/" + name + "_" + o + ".mtx", B, symmetric,
                               "rcm_study: " + name + " ordering=" + o + (o.find("shuffled") != std::string::npos
                                                                            ? " seed=" + std::to_string(seed)
                                                                            : ""));
        }
    }
    if (csv != stdout) std::fclose(csv);
    return failures ? 1 : 0;
}
