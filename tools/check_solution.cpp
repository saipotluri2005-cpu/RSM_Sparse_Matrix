// check_solution.cpp -- independent check of a solver's saved answer.
//
//   check_solution MATRIX SOLUTION [--ref REFERENCE] [--tol T] [--ref-tol T]
//
// Reads the matrix (.mtx / .mtx.gz) and a solution file of "index value" lines
// (the format cpu_only, cpu_sparse and cpu_sparse_v2 write, 0-based, original
// row numbering), and recomputes the relative residual ||b - A x|| / ||b||
// with b = 1 itself, instead of trusting the number the solver printed.
// With --ref it also reports ||x - ref|| / ||ref|| and the largest |x - ref|.
//
// Exit status: 0 if residual <= tol (default 1e-4) and, with --ref, the
// relative difference <= ref-tol (default 1e-6); 1 otherwise; 2 on bad input.
//
// Generated with assistance of claude code

#include "sparse_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static std::vector<double> read_solution(const std::string &path, int n)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<double> x(n, 0.0);
    std::vector<char> seen(n, 0);
    long long i;
    double v;
    int count = 0;
    while (in >> i >> v)
    {
        if (i < 0 || i >= n) throw std::runtime_error(path + ": index out of range: " + std::to_string(i));
        if (seen[i]) throw std::runtime_error(path + ": index repeated: " + std::to_string(i));
        seen[i] = 1;
        x[i] = v;
        ++count;
    }
    if (count != n)
        throw std::runtime_error(path + ": has " + std::to_string(count) + " values, matrix has " +
                                 std::to_string(n) + " rows");
    return x;
}

int main(int argc, char **argv)
{
    std::string mat, sol, ref;
    double tol = 1e-4, ref_tol = 1e-6;
    for (int a = 1; a < argc; ++a)
    {
        std::string s = argv[a];
        if (s == "--ref" && a + 1 < argc) ref = argv[++a];
        else if (s == "--tol" && a + 1 < argc) tol = std::atof(argv[++a]);
        else if (s == "--ref-tol" && a + 1 < argc) ref_tol = std::atof(argv[++a]);
        else if (mat.empty()) mat = s;
        else if (sol.empty()) sol = s;
        else { std::cerr << "unexpected argument: " << s << "\n"; return 2; }
    }
    if (mat.empty() || sol.empty())
    {
        std::cerr << "usage: check_solution MATRIX SOLUTION [--ref REFERENCE] [--tol T] [--ref-tol T]\n";
        return 2;
    }
    try
    {
        sio::Csr A = sio::to_csr(sio::read_mtx(mat));
        std::vector<double> x = read_solution(sol, A.n);

        long double r2 = 0.0L, b2 = 0.0L;
        for (int i = 0; i < A.n; ++i)
        {
            long double ax = 0.0L;
            for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
                ax += (long double)A.val[k] * x[A.ci[k]];
            long double d = 1.0L - ax;
            r2 += d * d;
            b2 += 1.0L;
        }
        const double resid = (double)std::sqrt(r2 / b2);
        bool ok = std::isfinite(resid) && resid <= tol;
        std::printf("matrix=%s n=%d nnz=%lld solution=%s\n", mat.c_str(), A.n, A.nnz(), sol.c_str());
        std::printf("relative_residual=%.6e tol=%.1e %s\n", resid, tol, ok ? "OK" : "FAIL");

        if (!ref.empty())
        {
            std::vector<double> y = read_solution(ref, A.n);
            long double d2 = 0.0L, y2 = 0.0L;
            double dmax = 0.0;
            for (int i = 0; i < A.n; ++i)
            {
                double d = x[i] - y[i];
                d2 += (long double)d * d;
                y2 += (long double)y[i] * y[i];
                dmax = std::max(dmax, std::fabs(d));
            }
            const double rel = y2 > 0 ? (double)std::sqrt(d2 / y2) : (double)std::sqrt(d2);
            const bool ref_ok = std::isfinite(rel) && rel <= ref_tol;
            std::printf("relative_difference=%.6e max_abs_difference=%.6e ref_tol=%.1e %s\n", rel, dmax,
                        ref_tol, ref_ok ? "OK" : "FAIL");
            ok = ok && ref_ok;
        }
        return ok ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
