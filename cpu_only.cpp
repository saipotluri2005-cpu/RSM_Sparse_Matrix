#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Dense Matrix-Vector Multiplication: y = A * x
void dense_matvec(const std::vector<std::vector<double>> &A,
                  const std::vector<double> &x,
                  std::vector<double> &y)
{
    int N = A.size();
    for (int i = 0; i < N; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
        {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }
}

// Vector Dot Product: a . b
double dot_product(const std::vector<double> &a, const std::vector<double> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

// Vector L2 Norm: ||v||
double vector_norm(const std::vector<double> &v)
{
    return std::sqrt(dot_product(v, v));
}

// Dense Conjugate Gradient Solver for Ax = b
int solve_dense_cg(const std::vector<std::vector<double>> &A,
                   const std::vector<double> &b,
                   std::vector<double> &x,
                   int max_iters, double tol)
{
    int N = A.size();
    std::vector<double> r(N), p(N), Ap(N);

    // Initial residual: r = b - A * x
    dense_matvec(A, x, Ap);
    for (int i = 0; i < N; ++i)
    {
        r[i] = b[i] - Ap[i];
        p[i] = r[i];
    }

    double rsold = dot_product(r, r);
    double b_norm = vector_norm(b);
    if (b_norm == 0.0)
        b_norm = 1.0;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        dense_matvec(A, p, Ap);
        double alpha = rsold / dot_product(p, Ap);

        for (int i = 0; i < N; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rsnew = dot_product(r, r);
        double relative_residual = std::sqrt(rsnew) / b_norm;

        if (relative_residual < tol)
        {
            return iter + 1;
        }

        for (int i = 0; i < N; ++i)
        {
            p[i] = r[i] + (rsnew / rsold) * p[i];
        }
        rsold = rsnew;
    }

    return max_iters;
}

// Helper to recursively find all .mtx files in a folder
std::vector<fs::path> find_all_mtx_files(const fs::path &folder_path)
{
    std::vector<fs::path> mtx_files;
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path))
    {
        std::cerr << "Error: Folder " << folder_path << " does not exist or is not a directory.\n";
        return mtx_files;
    }

    for (const auto &entry : fs::recursive_directory_iterator(folder_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".mtx")
        {
            mtx_files.push_back(entry.path());
        }
    }
    return mtx_files;
}

// Function to process a single .mtx file
void process_matrix_file(const fs::path &file_path)
{
    std::cout << "\n==================================================\n";
    std::cout << "Processing File: " << file_path.string() << "\n";
    std::cout << "==================================================\n";

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << file_path << std::endl;
        return;
    }

    std::string line;
    bool is_symmetric = false;

    // Check header
    std::getline(file, line);
    if (line.find("symmetric") != std::string::npos)
    {
        is_symmetric = true;
    }

    // Skip comments
    while (file.peek() == '%')
    {
        std::getline(file, line);
    }

    // Read dimensions
    int num_rows, num_cols, num_entries;
    file >> num_rows >> num_cols >> num_entries;

    std::cout << "Allocating Dense Matrix (" << num_rows << " x " << num_cols << ")..." << std::endl;

    // Create a 2D dense matrix initialized to zero
    std::vector<std::vector<double>> A(num_rows, std::vector<double>(num_cols, 0.0));

    // Read entries
    for (int i = 0; i < num_entries; ++i)
    {
        int r, c;
        double v;
        file >> r >> c >> v;
        r--;
        c--; // 1-based to 0-based index

        A[r][c] = v;
        if (is_symmetric && r != c)
        {
            A[c][r] = v;
        }
    }
    file.close();

    // Setup vectors
    std::vector<double> b(num_rows, 1.0);
    std::vector<double> x(num_rows, 0.0);

    // Run Dense Solver
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = solve_dense_cg(A, b, x, 5000, 1e-5);
    auto end = std::chrono::high_resolution_clock::now();

    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Accuracy check
    std::vector<double> Ax(num_rows);
    dense_matvec(A, x, Ax);
    std::vector<double> residual(num_rows);
    for (int i = 0; i < num_rows; ++i)
        residual[i] = Ax[i] - b[i];
    double rel_res = vector_norm(residual) / vector_norm(b);

    // Print Results
    std::cout << "Matrix Size:        " << num_rows << " x " << num_cols << std::endl;
    std::cout << "Iterations Taken:   " << iterations << std::endl;
    std::cout << "Solve Time:         " << duration_ms << " ms" << std::endl;
    std::cout << "Relative Residual:  " << rel_res << std::endl;

    // Save output file using filename
    std::string out_name = file_path.stem().string() + "_dense_solution.txt";
    std::ofstream out(out_name);
    for (int i = 0; i < num_rows; ++i)
    {
        out << i << " " << x[i] << "\n";
    }
    out.close();
    std::cout << "Solution saved to:  " << out_name << std::endl;
}

int main(int argc, char *argv[])
{
    // Default search directory is the current folder (.) or user provided argument
    fs::path target_dir = (argc > 1) ? argv[1] : ".";

    std::cout << "Recursively searching for .mtx files in: " << fs::absolute(target_dir) << std::endl;

    std::vector<fs::path> mtx_files = find_all_mtx_files(target_dir);

    if (mtx_files.empty())
    {
        std::cout << "No .mtx files found in directory: " << target_dir << std::endl;
        return 0;
    }

    std::cout << "Found " << mtx_files.size() << " .mtx file(s) to process." << std::endl;

    for (const auto &file_path : mtx_files)
    {
        process_matrix_file(file_path);
    }

    return 0;
}