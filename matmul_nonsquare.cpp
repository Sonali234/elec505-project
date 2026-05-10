#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// Modified local matmul to handle non-square (M x K) * (K x N)
void local_matmul_rectangular(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& R, 
                              int m_tile, int k_tile, int n_tile) {
    for (int i = 0; i < m_tile; ++i) {
        for (int k = 0; k < k_tile; ++k) {
            for (int j = 0; j < n_tile; ++j) {
                // R[i][j] += A[i][k] * B[k][j]
                R[i * n_tile + j] += A[i * k_tile + k] * B[k * n_tile + j];
            }
        }
    }
}

int main() {
    // Experimenting with different aspect ratios
    // Aspect ratios: Square, Tall-Skinny (K is large), Short-Wide (M/N are large)
    struct Shape { int M, K, N; std::string label; };
    std::vector<Shape> shapes = {
        {2048, 2048, 2048, "Square"},
        {512, 8192, 512,   "Tall-Skinny"},
        {4096, 512, 4096,  "Short-Wide"}
    };

    int c = 2; // Fixed replication factor
    int p = 32; // Fixed processor count

    std::cout << "==============================================================================\n";
    std::cout << "Simulating Maunam 2.5D MM: Matrix Aspect Ratio Analysis (p=" << p << ", c=" << c << ")\n";
    std::cout << "==============================================================================\n";
    std::cout << std::left << std::setw(15) << "Label" 
              << std::setw(20) << "Dimensions (MxKxN)" 
              << std::setw(20) << "Tile (m x k x n)" 
              << "Compute Time (s)\n";
    std::cout << "------------------------------------------------------------------------------\n";

    for (const auto& s : shapes) {
        // Maunam Grid Logic
        int grid_dim = std::round(std::sqrt(p / c));
        
        // Local tile dimensions
        int m_tile = s.M / grid_dim;
        int n_tile = s.N / grid_dim;
        int k_tile = s.K / c; // K is partitioned by c in 2.5D algorithms

        // Allocate memory
        std::vector<double> A(m_tile * k_tile, 1.0);
        std::vector<double> B(k_tile * n_tile, 2.0);
        std::vector<double> R(m_tile * n_tile, 0.0);

        // Start Timing
        auto start = std::chrono::high_resolution_clock::now();

        // Number of systolic steps is the grid dimension
        for (int step = 0; step < grid_dim; ++step) {
            local_matmul_rectangular(A, B, R, m_tile, k_tile, n_tile);
        }

        // Stop Timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::string dim_str = std::to_string(s.M) + "x" + std::to_string(s.K) + "x" + std::to_string(s.N);
        std::string tile_str = std::to_string(m_tile) + "x" + std::to_string(k_tile) + "x" + std::to_string(n_tile);

        std::cout << std::left << std::setw(15) << s.label 
                  << std::setw(20) << dim_str 
                  << std::setw(20) << tile_str 
                  << std::fixed << std::setprecision(6) << duration.count() << "\n";
    }

    std::cout << "------------------------------------------------------------------------------\n";
    return 0;
}