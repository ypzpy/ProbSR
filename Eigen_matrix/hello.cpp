#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

// 构建离散化的拉普拉斯算子
SparseMatrix<double> build_laplacian(int N) {
    int N3 = N * N * N;
    SparseMatrix<double> A(N3, N3);
    std::vector<Triplet<double>> triplets;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = i * N * N + j * N + k;

                // 边界条件：Dirichlet 边界条件，边界上的值为 0
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
                    triplets.emplace_back(idx, idx, 1.0);
                } else {
                    triplets.emplace_back(idx, idx, -6.0);
                    triplets.emplace_back(idx, idx - N * N, 1.0);
                    triplets.emplace_back(idx, idx + N * N, 1.0);
                    triplets.emplace_back(idx, idx - N, 1.0);
                    triplets.emplace_back(idx, idx + N, 1.0);
                    triplets.emplace_back(idx, idx - 1, 1.0);
                    triplets.emplace_back(idx, idx + 1, 1.0);
                }
            }
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// 构建右手边的forcing项
VectorXd build_rhs(int N, double a, double b, double c) {
    VectorXd b_vec(N * N * N);
    double pi = M_PI;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = static_cast<double>(i) / (N - 1);
                double y = static_cast<double>(j) / (N - 1);
                double z = static_cast<double>(k) / (N - 1);
                int idx = i * N * N + j * N + k;

                // 边界条件：Dirichlet 边界条件，边界上的值为 0
                if (i == 0 || i == N-1 || j == 0 || j == N-1 || k == 0 || k == N-1) {
                    b_vec[idx] = 0.0;
                } else {
                    b_vec[idx] = std::sin(a * pi * x) * std::cos(b * pi * y) * std::exp(-c * z) +
                                 std::sin(b * pi * y) * std::cos(c * pi * z) * std::exp(-a * x) +
                                 std::sin(c * pi * z) * std::cos(a * pi * x) * std::exp(-b * y);
                }
            }
        }
    }

    return b_vec;
}

// 将解保存到文件
void save_solution_to_file(const VectorXd& solution, int N, const std::string& filename) {
    std::ofstream outfile(filename);

    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = i * N * N + j * N + k;
                outfile << solution[idx] << " ";
            }
            outfile << std::endl;
        }
        outfile << std::endl;
    }

    outfile.close();
}

int main() {
    int N = 60;  // 网格点数 (x, y, z 方向)
    double a = 1.0, b = 1.0, c = 1.0;  // 参数

    // 构建离散化的拉普拉斯算子
    SparseMatrix<double> A = build_laplacian(N);

    // 构建右手边的forcing项
    VectorXd b_vec = build_rhs(N, a, b, c);

    // 使用 Eigen 求解线性方程
    SparseLU<SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return -1;
    }

    VectorXd solution = solver.solve(b_vec);
    if (solver.info() != Success) {
        std::cerr << "Solving failed" << std::endl;
        return -1;
    }

    // 将解保存到文件
    // save_solution_to_file(solution, N, "poisson_solution.txt");

    return 0;
}
