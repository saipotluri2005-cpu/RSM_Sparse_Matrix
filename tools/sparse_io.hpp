// sparse_io.hpp -- Matrix Market reading/writing, CSR building and ordering
// metrics shared by the programs in tools/.
//
// Generated with assistance of claude code

#pragma once

#include <omp.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sio {

// Entries as stored in the file (one triangle if symmetric), 0-based.
struct Coo
{
    int n = 0;
    bool symmetric = false;
    bool pattern = false;
    std::vector<int> r, c;
    std::vector<double> v;
};

// Full matrix (both triangles), each row sorted by column, duplicates summed.
struct Csr
{
    int n = 0;
    std::vector<long long> rp;
    std::vector<int> ci;
    std::vector<double> val;
    long long nnz() const { return rp.empty() ? 0 : rp[n]; }
};

inline double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline bool ends_with(const std::string &s, const std::string &suf)
{
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// "dir/name.mtx.gz" -> "name"
inline std::string stem(const std::string &path)
{
    std::string n = path.substr(path.find_last_of('/') + 1);
    if (ends_with(n, ".gz")) n.resize(n.size() - 3);
    if (ends_with(n, ".mtx")) n.resize(n.size() - 4);
    return n;
}

inline std::string slurp(const std::string &path)
{
    std::string s;
    if (ends_with(path, ".gz"))
    {
        std::string cmd = "gzip -dc '" + path + "'";
        FILE *f = popen(cmd.c_str(), "r");
        if (!f) throw std::runtime_error("cannot run gzip on " + path);
        char chunk[1 << 16];
        size_t k;
        while ((k = std::fread(chunk, 1, sizeof chunk, f)) > 0) s.append(chunk, k);
        if (pclose(f) != 0) throw std::runtime_error("gzip failed on " + path);
        return s;
    }
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::rewind(f);
    s.resize(sz > 0 ? sz : 0);
    s.resize(std::fread(s.data(), 1, s.size(), f));
    std::fclose(f);
    return s;
}

// Square, real/integer/pattern, general/symmetric coordinate files only.
inline Coo read_mtx(const std::string &path)
{
    std::string buf = slurp(path);
    size_t eol = buf.find('\n');
    if (eol == std::string::npos) throw std::runtime_error(path + ": no header line");
    std::string header = buf.substr(0, eol);
    for (char &ch : header) ch = (char)std::tolower((unsigned char)ch);
    std::istringstream hs(header);
    std::string magic, object, format, field, symmetry;
    hs >> magic >> object >> format >> field >> symmetry;
    if (magic != "%%matrixmarket" || object != "matrix" || format != "coordinate")
        throw std::runtime_error(path + ": not a coordinate Matrix Market file");
    if (field != "real" && field != "integer" && field != "pattern")
        throw std::runtime_error(path + ": unsupported field '" + field + "'");
    if (symmetry != "general" && symmetry != "symmetric")
        throw std::runtime_error(path + ": unsupported symmetry '" + symmetry + "'");

    Coo m;
    m.symmetric = symmetry == "symmetric";
    m.pattern = field == "pattern";
    const char *p = buf.c_str() + eol + 1;
    while (*p == '%')
    {
        while (*p && *p != '\n') ++p;
        if (*p) ++p;
    }
    char *end;
    long long rows = std::strtoll(p, &end, 10); p = end;
    long long cols = std::strtoll(p, &end, 10); p = end;
    long long ents = std::strtoll(p, &end, 10); p = end;
    if (rows <= 0 || rows != cols || rows > 2147483647LL || ents < 0)
        throw std::runtime_error(path + ": matrix must be square with 32-bit size");
    m.n = (int)rows;
    m.r.resize(ents); m.c.resize(ents); m.v.resize(ents);
    for (long long k = 0; k < ents; ++k)
    {
        long long i = std::strtoll(p, &end, 10);
        if (end == p) throw std::runtime_error(path + ": truncated at entry " + std::to_string(k));
        p = end;
        long long j = std::strtoll(p, &end, 10); p = end;
        double v = 1.0;
        if (!m.pattern) { v = std::strtod(p, &end); p = end; }
        if (i < 1 || i > rows || j < 1 || j > cols)
            throw std::runtime_error(path + ": bad index at entry " + std::to_string(k));
        m.r[k] = (int)(i - 1); m.c[k] = (int)(j - 1); m.v[k] = v;
    }
    return m;
}

// Sort each row by column and sum duplicates; rows given by rp/ci/val with
// capacity per row, compacted in place.
inline void sort_and_merge(Csr &A)
{
    const int n = A.n;
    std::vector<long long> len(n);
#pragma omp parallel
    {
        std::vector<std::pair<int, double>> tmp;
#pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < n; ++i)
        {
            tmp.clear();
            for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k) tmp.emplace_back(A.ci[k], A.val[k]);
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            long long w = A.rp[i];
            for (const auto &t : tmp)
            {
                if (w > A.rp[i] && A.ci[w - 1] == t.first) A.val[w - 1] += t.second;
                else { A.ci[w] = t.first; A.val[w] = t.second; ++w; }
            }
            len[i] = w - A.rp[i];
        }
    }
    long long w = 0;
    for (int i = 0; i < n; ++i)
    {
        const long long b = A.rp[i];
        if (w != b)
        {
            std::memmove(&A.ci[w], &A.ci[b], len[i] * sizeof(int));
            std::memmove(&A.val[w], &A.val[b], len[i] * sizeof(double));
        }
        A.rp[i] = w;
        w += len[i];
    }
    A.rp[n] = w;
    A.ci.resize(w);
    A.val.resize(w);
}

// COO -> full CSR. perm (new = perm[old]) is optional.
inline Csr to_csr(const Coo &m, const std::vector<int> *perm = nullptr)
{
    auto P = [&](int i) { return perm ? (*perm)[i] : i; };
    Csr A;
    A.n = m.n;
    A.rp.assign(m.n + 1, 0);
    const size_t E = m.r.size();
    for (size_t k = 0; k < E; ++k)
    {
        A.rp[P(m.r[k]) + 1]++;
        if (m.symmetric && m.r[k] != m.c[k]) A.rp[P(m.c[k]) + 1]++;
    }
    for (int i = 0; i < m.n; ++i) A.rp[i + 1] += A.rp[i];
    A.ci.resize(A.rp[m.n]);
    A.val.resize(A.rp[m.n]);
    std::vector<long long> fill(A.rp.begin(), A.rp.end() - 1);
    for (size_t k = 0; k < E; ++k)
    {
        int r = P(m.r[k]), c = P(m.c[k]);
        long long q = fill[r]++;
        A.ci[q] = c; A.val[q] = m.v[k];
        if (m.symmetric && m.r[k] != m.c[k]) { q = fill[c]++; A.ci[q] = r; A.val[q] = m.v[k]; }
    }
    sort_and_merge(A);
    return A;
}

// Relabel rows and columns: new index = perm[old].
inline Csr permute(const Csr &A, const std::vector<int> &perm)
{
    const int n = A.n;
    Csr B;
    B.n = n;
    B.rp.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) B.rp[perm[i] + 1] = A.rp[i + 1] - A.rp[i];
    for (int i = 0; i < n; ++i) B.rp[i + 1] += B.rp[i];
    B.ci.resize(A.nnz());
    B.val.resize(A.nnz());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        long long w = B.rp[perm[i]];
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k, ++w)
        {
            B.ci[w] = perm[A.ci[k]];
            B.val[w] = A.val[k];
        }
    }
    sort_and_merge(B);
    return B;
}

// True if perm is a bijection on [0, n).
inline bool is_permutation(const std::vector<int> &perm, int n)
{
    if ((int)perm.size() != n) return false;
    std::vector<char> seen(n, 0);
    for (int p : perm)
    {
        if (p < 0 || p >= n || seen[p]) return false;
        seen[p] = 1;
    }
    return true;
}

struct Shape
{
    long long bandwidth = 0;   // max |i - j|
    long long profile = 0;     // sum over rows of (i - first column in row i), lower part
    double mean_dist = 0.0;    // mean |i - j| over nonzeros
};

inline Shape shape(const Csr &A)
{
    Shape s;
    long long bw = 0, prof = 0;
    double dist = 0.0;
#pragma omp parallel for reduction(max : bw) reduction(+ : prof, dist) schedule(static)
    for (int i = 0; i < A.n; ++i)
    {
        if (A.rp[i] == A.rp[i + 1]) continue;
        const long long first = std::min<long long>(A.ci[A.rp[i]], i);
        prof += i - first;
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
        {
            long long d = std::llabs((long long)A.ci[k] - i);
            bw = std::max(bw, d);
            dist += (double)d;
        }
    }
    s.bandwidth = bw;
    s.profile = prof;
    s.mean_dist = A.nnz() ? dist / (double)A.nnz() : 0.0;
    return s;
}

// Buffered Matrix Market writer (1-based indices, shortest round-trip values).
class MtxWriter
{
  public:
    MtxWriter(const std::string &path, long long n, long long entries, bool symmetric,
              const std::string &comment)
        : f_(std::fopen(path.c_str(), "wb")), buf_(1 << 22)
    {
        if (!f_) throw std::runtime_error("cannot write " + path);
        std::string h = std::string("%%MatrixMarket matrix coordinate real ") +
                        (symmetric ? "symmetric" : "general") + "\n";
        if (!comment.empty()) h += "% " + comment + "\n";
        h += std::to_string(n) + " " + std::to_string(n) + " " + std::to_string(entries) + "\n";
        put(h.data(), h.size());
    }
    ~MtxWriter() { close(); }

    void entry(long long r1, long long c1, double v)
    {
        if (pos_ + 80 > buf_.size()) flush();
        char *p = buf_.data() + pos_, *e = buf_.data() + buf_.size();
        p = std::to_chars(p, e, r1).ptr; *p++ = ' ';
        p = std::to_chars(p, e, c1).ptr; *p++ = ' ';
        p = std::to_chars(p, e, v).ptr; *p++ = '\n';
        pos_ = p - buf_.data();
        ++written_;
    }
    long long written() const { return written_; }
    void close()
    {
        if (!f_) return;
        flush();
        std::fclose(f_);
        f_ = nullptr;
    }

  private:
    void put(const char *s, size_t k)
    {
        if (pos_ + k > buf_.size()) flush();
        std::memcpy(buf_.data() + pos_, s, k);
        pos_ += k;
    }
    void flush()
    {
        if (pos_ && std::fwrite(buf_.data(), 1, pos_, f_) != pos_)
            throw std::runtime_error("write failed");
        pos_ = 0;
    }
    FILE *f_;
    std::vector<char> buf_;
    size_t pos_ = 0;
    long long written_ = 0;
};

// Write a full CSR; symmetric output keeps the lower triangle only.
inline void write_csr(const std::string &path, const Csr &A, bool symmetric,
                      const std::string &comment)
{
    long long count = 0;
    for (int i = 0; i < A.n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            if (!symmetric || A.ci[k] <= i) ++count;
    MtxWriter w(path, A.n, count, symmetric, comment);
    for (int i = 0; i < A.n; ++i)
        for (long long k = A.rp[i]; k < A.rp[i + 1]; ++k)
            if (!symmetric || A.ci[k] <= i) w.entry(i + 1, A.ci[k] + 1, A.val[k]);
    w.close();
}

} // namespace sio
