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
    int c = 2; // Replication factor
    int base_tile_size = 512; // We fix the tile size to keep work per processor constant
    
    // Processor counts to test
    int p_values[] = {2, 8, 18, 32, 50, 72}; 

    std::cout << "================================================================================\n";
    std::cout << "Simulating Maunam 2.5D MM: Weak Scaling Analysis (Constant Work per Processor)\n";
    std::cout << "Target Local Tile Size: " << base_tile_size << "x" << base_tile_size << "\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(10) << "p" 
              << std::setw(15) << "Global N" 
              << std::setw(15) << "Grid Dim" 
              << std::setw(15) << "Total Ops" 
              << "Compute Time (s)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    for (int p : p_values) {
        // In weak scaling, N increases with p.
        // Grid dimension (sqrt(p/c)) determines how many tiles make up the global N.
        int sqrt_p_c = std::round(std::sqrt(p / c));
        int n_global = base_tile_size * sqrt_p_c;
        
        int num_elements = base_tile_size * base_tile_size;

        std::vector<double> A(num_elements, 1.0);
        std::vector<double> B(num_elements, 2.0);
        std::vector<double> R(num_elements, 0.0);

        // Start Timing
        auto start = std::chrono::high_resolution_clock::now();

        // Each processor performs 'sqrt_p_c' steps of the local tile multiplication
        for (int step = 0; step < sqrt_p_c; ++step) {
            local_matmul(A, B, R, base_tile_size);
        }

        // Stop Timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Calculate total operations (N^3)
        long long total_ops = (long long)n_global * n_global * n_global;

        std::cout << std::left << std::setw(10) << p 
                  << std::setw(15) << n_global 
                  << std::setw(15) << sqrt_p_c 
                  << std::setw(15) << total_ops 
                  << std::fixed << std::setprecision(6) << duration.count() << "\n";
    }

    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Note: In ideal weak scaling, Compute Time should remain constant as p increases.\n";
    return 0;
}