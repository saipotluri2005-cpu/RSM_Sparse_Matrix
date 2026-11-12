// gen_fem.cpp -- SPD test matrices of any size, written as symmetric
// Matrix Market files (lower triangle).
//
//   gen_fem poisson1d N              [options]   tridiagonal (2, -1), N unknowns
//   gen_fem poisson2d n              [options]   5-point grid, n^2 unknowns
//   gen_fem poisson3d n              [options]   7-point grid, n^3 unknowns
//   gen_fem elasticity nx [ny nz]    [options]   3D linear elasticity
//
// elasticity: a box of nx*ny*nz unit hexahedral (8-node, trilinear) elements,
// E = 1, nu = 0.3, 2x2x2 Gauss quadrature, all nodes on the x = 0 face clamped.
// Three unknowns (ux, uy, uz) per free node, numbered node by node; up to 81
// nonzeros per row. Free unknowns: 3 * nx * (ny+1) * (nz+1).
//
// Options:
//   --shuffle SEED   renumber the nodes randomly (a node's 3 unknowns stay
//                    together), like a mesh generator's numbering
//   -o FILE          output path (default: <problem>_<size>[_shufSEED].mtx)
//   --check          elasticity: verify the element stiffness is symmetric and
//                    that rigid-body motions produce no force; exit 1 if not
//
// Generated with assistance of claude code

#include "sparse_io.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// Fisher-Yates with a fixed generator, so a seed gives the same file everywhere.
std::vector<int> random_perm(int n, unsigned long long seed)
{
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::mt19937_64 rng(seed);
    for (int i = n - 1; i > 0; --i) std::swap(p[i], p[(int)(rng() % (unsigned long long)(i + 1))]);
    return p;
}

// ---------------------------------------------------------------------------
// Hexahedral element stiffness. Local node a has corner (a&1, a>>1&1, a>>2&1).
// ---------------------------------------------------------------------------
using Ke24 = std::array<std::array<double, 24>, 24>;

Ke24 hex_stiffness(double E, double nu)
{
    const double lam = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
    double D[6][6] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) D[i][j] = lam + (i == j ? 2 * mu : 0.0);
    for (int i = 3; i < 6; ++i) D[i][i] = mu;

    Ke24 K{};
    const double g = 1.0 / std::sqrt(3.0);
    const double detJ = 0.125; // unit cube: x = (xi + 1) / 2
    for (int gi = 0; gi < 8; ++gi)
    {
        const double xi = (gi & 1 ? g : -g), et = (gi >> 1 & 1 ? g : -g), ze = (gi >> 2 & 1 ? g : -g);
        double B[6][24] = {};
        for (int a = 0; a < 8; ++a)
        {
            const double sa = (a & 1) ? 1 : -1, ta = (a >> 1 & 1) ? 1 : -1, ua = (a >> 2 & 1) ? 1 : -1;
            // dN/dx = 2 dN/dxi on the unit cube
            const double nx = 2 * 0.125 * sa * (1 + ta * et) * (1 + ua * ze);
            const double ny = 2 * 0.125 * ta * (1 + sa * xi) * (1 + ua * ze);
            const double nz = 2 * 0.125 * ua * (1 + sa * xi) * (1 + ta * et);
            B[0][3 * a] = nx;
            B[1][3 * a + 1] = ny;
            B[2][3 * a + 2] = nz;
            B[3][3 * a] = ny; B[3][3 * a + 1] = nx;
            B[4][3 * a + 1] = nz; B[4][3 * a + 2] = ny;
            B[5][3 * a] = nz; B[5][3 * a + 2] = nx;
        }
        double DB[6][24] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 24; ++j)
                for (int k = 0; k < 6; ++k) DB[i][j] += D[i][k] * B[k][j];
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
            {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += B[k][i] * DB[k][j];
                K[i][j] += s * detJ;
            }
    }
    return K;
}

bool check_element(const Ke24 &K)
{
    double kmax = 0.0, asym = 0.0, rigid = 0.0, dmin = 1e300;
    for (int i = 0; i < 24; ++i)
    {
        dmin = std::min(dmin, K[i][i]);
        for (int j = 0; j < 24; ++j)
        {
            kmax = std::max(kmax, std::fabs(K[i][j]));
            asym = std::max(asym, std::fabs(K[i][j] - K[j][i]));
        }
    }
    // 3 translations and 3 rotations about the origin
    for (int mode = 0; mode < 6; ++mode)
    {
        double u[24];
        for (int a = 0; a < 8; ++a)
        {
            const double X = a & 1, Y = a >> 1 & 1, Z = a >> 2 & 1;
            double v[3] = {0, 0, 0};
            if (mode < 3) v[mode] = 1;
            else if (mode == 3) { v[0] = -Y; v[1] = X; }  // about z
            else if (mode == 4) { v[1] = -Z; v[2] = Y; }  // about x
            else { v[0] = Z; v[2] = -X; }                 // about y
            for (int c = 0; c < 3; ++c) u[3 * a + c] = v[c];
        }
        for (int i = 0; i < 24; ++i)
        {
            double f = 0.0;
            for (int j = 0; j < 24; ++j) f += K[i][j] * u[j];
            rigid = std::max(rigid, std::fabs(f));
        }
    }
    const bool ok = asym <= 1e-14 * kmax && rigid <= 1e-12 * kmax && dmin > 0;
    std::printf("element check: max|K|=%.6g  max|K-K^T|/max|K|=%.2e  max|K*rigid|/max|K|=%.2e  "
                "min diag=%.6g  %s\n",
                kmax, asym / kmax, rigid / kmax, dmin, ok ? "OK" : "FAIL");
    return ok;
}

// Calls emit(row, col, value) for every nonzero with col <= row (natural
// numbering). Each problem knows its own structure, so nothing is stored.
struct Problem
{
    std::string kind;
    int nx = 0, ny = 0, nz = 0;
    int dofs_per_node = 1;
    long long nodes = 0; // unknown-carrying nodes
    Ke24 K{};
    double drop = 0.0;   // couplings below this are exact cancellations

    long long unknowns() const { return nodes * dofs_per_node; }

    template <class F> void for_each_lower(F &&emit) const
    {
        if (kind == "poisson1d")
        {
            for (long long i = 0; i < nodes; ++i)
            {
                if (i > 0) emit(i, i - 1, -1.0);
                emit(i, i, 2.0);
            }
        }
        else if (kind == "poisson2d")
        {
            const long long n = nx;
            for (long long i = 0; i < n; ++i)
                for (long long j = 0; j < n; ++j)
                {
                    long long k = i * n + j;
                    if (i > 0) emit(k, k - n, -1.0);
                    if (j > 0) emit(k, k - 1, -1.0);
                    emit(k, k, 4.0);
                }
        }
        else if (kind == "poisson3d")
        {
            const long long n = nx;
            for (long long i = 0; i < n; ++i)
                for (long long j = 0; j < n; ++j)
                    for (long long l = 0; l < n; ++l)
                    {
                        long long k = (i * n + j) * n + l;
                        if (i > 0) emit(k, k - n * n, -1.0);
                        if (j > 0) emit(k, k - n, -1.0);
                        if (l > 0) emit(k, k - 1, -1.0);
                        emit(k, k, 6.0);
                    }
        }
        else // elasticity: free nodes have i >= 1; free index f = (i-1) + nx*(j + (ny+1)*k)
        {
            auto fidx = [&](long long i, long long j, long long k) { return (i - 1) + nx * (j + (long long)(ny + 1) * k); };
            double blk[27][3][3];
            for (long long k = 0; k <= nz; ++k)
                for (long long j = 0; j <= ny; ++j)
                    for (long long i = 1; i <= nx; ++i)
                    {
                        std::fill(&blk[0][0][0], &blk[0][0][0] + 27 * 9, 0.0);
                        for (long long ek = k - 1; ek <= k; ++ek)
                            for (long long ej = j - 1; ej <= j; ++ej)
                                for (long long ei = i - 1; ei <= i; ++ei)
                                {
                                    if (ei < 0 || ei >= nx || ej < 0 || ej >= ny || ek < 0 || ek >= nz) continue;
                                    const int a = (int)((i - ei) + 2 * (j - ej) + 4 * (k - ek));
                                    for (int b = 0; b < 8; ++b)
                                    {
                                        const long long qi = ei + (b & 1), qj = ej + (b >> 1 & 1), qk = ek + (b >> 2 & 1);
                                        const int o = (int)((qi - i + 1) + 3 * ((qj - j + 1) + 3 * (qk - k + 1)));
                                        for (int cp = 0; cp < 3; ++cp)
                                            for (int cq = 0; cq < 3; ++cq) blk[o][cp][cq] += K[3 * a + cp][3 * b + cq];
                                    }
                                }
                        const long long fp = fidx(i, j, k);
                        for (int cp = 0; cp < 3; ++cp)
                            for (int o = 0; o < 27; ++o)
                            {
                                const long long qi = i + o % 3 - 1, qj = j + o / 3 % 3 - 1, qk = k + o / 9 - 1;
                                if (qi < 1 || qi > nx || qj < 0 || qj > ny || qk < 0 || qk > nz) continue;
                                const long long fq = fidx(qi, qj, qk);
                                for (int cq = 0; cq < 3; ++cq)
                                {
                                    if (fq > fp || (fq == fp && cq > cp)) continue; // lower triangle only
                                    const double v = blk[o][cp][cq];
                                    if (std::fabs(v) > drop) emit(3 * fp + cp, 3 * fq + cq, v);
                                }
                            }
                    }
        }
    }
};

[[noreturn]] void usage()
{
    std::fprintf(stderr,
                 "usage: gen_fem poisson1d N | poisson2d n | poisson3d n | elasticity nx [ny nz]\n"
                 "               [--shuffle SEED] [-o FILE] [--check]\n");
    std::exit(2);
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 3) usage();
    Problem P;
    P.kind = argv[1];
    std::vector<long long> sizes;
    std::string out;
    bool shuffle = false, check = false;
    unsigned long long seed = 1;
    for (int a = 2; a < argc; ++a)
    {
        std::string s = argv[a];
        if (s == "--shuffle" && a + 1 < argc) { shuffle = true; seed = std::strtoull(argv[++a], nullptr, 10); }
        else if (s == "-o" && a + 1 < argc) out = argv[++a];
        else if (s == "--check") check = true;
        else if (!s.empty() && s[0] != '-') sizes.push_back(std::atoll(s.c_str()));
        else usage();
    }
    if (sizes.empty() || sizes[0] < 1) usage();

    std::string tag;
    if (P.kind == "poisson1d") { P.nodes = sizes[0]; tag = std::to_string(sizes[0]); }
    else if (P.kind == "poisson2d") { P.nx = (int)sizes[0]; P.nodes = sizes[0] * sizes[0]; tag = std::to_string(sizes[0]); }
    else if (P.kind == "poisson3d") { P.nx = (int)sizes[0]; P.nodes = sizes[0] * sizes[0] * sizes[0]; tag = std::to_string(sizes[0]); }
    else if (P.kind == "elasticity")
    {
        P.nx = (int)sizes[0];
        P.ny = (int)(sizes.size() > 1 ? sizes[1] : sizes[0]);
        P.nz = (int)(sizes.size() > 2 ? sizes[2] : sizes[0]);
        if (P.ny < 1 || P.nz < 1) usage();
        P.dofs_per_node = 3;
        P.nodes = (long long)P.nx * (P.ny + 1) * (P.nz + 1);
        P.K = hex_stiffness(1.0, 0.3);
        double kmax = 0.0;
        for (auto &row : P.K)
            for (double v : row) kmax = std::max(kmax, std::fabs(v));
        P.drop = 1e-12 * kmax;
        tag = std::to_string(P.nx) + "x" + std::to_string(P.ny) + "x" + std::to_string(P.nz);
        if (check && !check_element(P.K)) return 1;
    }
    else usage();

    const long long N = P.unknowns();
    if (N > 2147483647LL) { std::fprintf(stderr, "too many unknowns for 32-bit indices\n"); return 2; }
    if (out.empty()) out = P.kind + "_" + tag + (shuffle ? "_shuf" + std::to_string(seed) : "") + ".mtx";

    std::vector<int> node_perm;
    if (shuffle) node_perm = random_perm((int)P.nodes, seed);
    const int d = P.dofs_per_node;
    auto map = [&](long long u) -> long long {
        return shuffle ? (long long)node_perm[u / d] * d + u % d : u;
    };

    const double t0 = sio::now_ms();
    long long count = 0;
    P.for_each_lower([&](long long, long long, double) { ++count; });

    std::string comment = "gen_fem " + P.kind + " " + tag;
    if (P.kind == "elasticity") comment += " E=1 nu=0.3 hex8 2x2x2-Gauss clamped-x0";
    if (shuffle) comment += " node-shuffle-seed=" + std::to_string(seed);
    sio::MtxWriter w(out, N, count, true, comment);
    P.for_each_lower([&](long long r, long long c, double v) {
        long long R = map(r), C = map(c);
        if (C > R) std::swap(R, C);
        w.entry(R + 1, C + 1, v);
    });
    w.close();
    std::printf("wrote %s  N=%lld  stored entries=%lld  (%.1f s)\n", out.c_str(), N, count,
                (sio::now_ms() - t0) / 1000.0);
    return 0;
}
