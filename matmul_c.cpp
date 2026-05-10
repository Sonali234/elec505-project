#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// The exact same local tile multiplication used in the MPI version
void local_matmul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& R, int tile_size) {
    for (int i = 0; i < tile_size; ++i) {
        for (int k = 0; k < tile_size; ++k) {
            for (int j = 0; j < tile_size; ++j) {
                R[i * tile_size + j] += A[i * tile_size + k] * B[k * tile_size + j];
            }
        }
    }
}

int main() {
    int n = 2048; // Global matrix size
    
    // We will test multiple replication factors (c)
    // Note: c must be a factor of p, and p/c must be a perfect square for this simulation logic
    std::vector<int> c_values = {1, 2, 4, 8}; 
    std::vector<int> p_values = {16, 32, 64};

    std::cout << "============================================================================\n";
    std::cout << "Simulating Maunam 2.5D MM: Variation of Replication Factor (c) for N=" << n << "\n";
    std::cout << "============================================================================\n";
    std::cout << std::left << std::setw(5) << "c" 
              << std::setw(10) << "p" 
              << std::setw(15) << "Tile Size" 
              << std::setw(15) << "Steps" 
              << "Compute Time (s)\n";
    std::cout << "----------------------------------------------------------------------------\n";

    for (int c : c_values) {
        for (int p : p_values) {
            // Ensure p/c is at least 1 to avoid math errors
            if (p < c) continue;

            // Calculate the grid dimension: sqrt(p/c)
            // In a real 2.5D grid, this represents the processors per row/column in one layer
            int sqrt_p_c = std::round(std::sqrt(p / c));
            
            // Recalculate tile size based on the current c
            int tile_size = n / sqrt_p_c;
            int num_elements = tile_size * tile_size;

            // Allocate memory for the current tile
            std::vector<double> A(num_elements, 1.0);
            std::vector<double> B(num_elements, 2.0);
            std::vector<double> R(num_elements, 0.0);

            // Start Timing
            auto start = std::chrono::high_resolution_clock::now();

            // The number of systolic shift steps is also sqrt(p/c)
            for (int step = 0; step < sqrt_p_c; ++step) {
                local_matmul(A, B, R, tile_size);
            }

            // Stop Timing
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            std::cout << std::left << std::setw(5) << c 
                      << std::setw(10) << p 
                      << std::setw(15) << (std::to_string(tile_size) + "x" + std::to_string(tile_size)) 
                      << std::setw(15) << sqrt_p_c 
                      << std::fixed << std::setprecision(6) << duration.count() << "\n";
        }
        std::cout << "----------------------------------------------------------------------------\n";
    }
    
    return 0;
}