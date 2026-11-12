// rcm.hpp -- Cuthill-McKee orderings for the tools/ programs.
//
// With default options this is the same algorithm as rcm_order() in
// cpu_sparse.cpp / cpu_sparse_v2.cpp (same adjacency, start node, neighbour
// order and tie-breaking), so it produces the same permutation; qa/run_tests.sh
// checks that. The options select the variants compared by rcm_study.
//
// Generated with assistance of claude code

#pragma once

#include "sparse_io.hpp"

#include <algorithm>
#include <vector>

namespace rcm {

enum class Start
{
    Peripheral, // pseudo-peripheral node (repeated BFS), as in cpu_sparse
    MinDegree,  // lowest-degree node of the component
    First       // lowest-numbered node of the component
};

struct Options
{
    Start start = Start::Peripheral;
    bool reverse = true; // false gives plain Cuthill-McKee
};

// Returns perm with perm[old] = new.
inline std::vector<int> order(const sio::Csr &A, Options opt = {})
{
    const int n = A.n;
    // symmetric adjacency: pattern of A + A^T without the diagonal, deduplicated
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

    std::vector<int> ord, level(n, -1), q, nb;
    ord.reserve(n);
    std::vector<char> done(n, 0);

    auto bfs = [&](int s) { // BFS over unvisited nodes; returns eccentricity of s
        q.clear();
        q.push_back(s);
        level[s] = 0;
        int ecc = 0;
        for (size_t h = 0; h < q.size(); ++h)
        {
            int u = q[h];
            for (long long k = ap[u]; k < ap[u] + deg[u]; ++k)
                if (int w = adj[k]; !done[w] && level[w] < 0)
                {
                    level[w] = level[u] + 1;
                    ecc = level[w];
                    q.push_back(w);
                }
        }
        return ecc;
    };

    for (int start = 0; start < n; ++start)
    {
        if (done[start]) continue;
        int s = start;
        if (opt.start == Start::Peripheral)
        {
            int ecc = bfs(s);
            for (int tries = 0; tries < 8; ++tries)
            {
                int best = -1;
                for (int u : q)
                    if (level[u] == ecc && (best < 0 || deg[u] < deg[best])) best = u;
                for (int u : q) level[u] = -1;
                int e2 = bfs(best);
                if (e2 <= ecc) break;
                s = best;
                ecc = e2;
            }
            for (int u : q) level[u] = -1;
        }
        else if (opt.start == Start::MinDegree)
        {
            bfs(s);
            for (int u : q)
                if (deg[u] < deg[s]) s = u;
            for (int u : q) level[u] = -1;
        }

        size_t head = ord.size();
        ord.push_back(s);
        done[s] = 1;
        while (head < ord.size())
        {
            int u = ord[head++];
            nb.clear();
            for (long long k = ap[u]; k < ap[u] + deg[u]; ++k)
                if (!done[adj[k]]) { nb.push_back(adj[k]); done[adj[k]] = 1; }
            std::sort(nb.begin(), nb.end(), [&](int a, int b) { return deg[a] < deg[b]; });
            ord.insert(ord.end(), nb.begin(), nb.end());
        }
    }
    std::vector<int> perm(n);
    for (int k = 0; k < n; ++k)
        perm[ord[opt.reverse ? n - 1 - k : k]] = k;
    return perm;
}

} // namespace rcm
