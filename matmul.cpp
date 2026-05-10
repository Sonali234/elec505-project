#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

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
    int c = 2;    // Maunam's 2.5D replication factor
    
    // The processor counts we want to test
    int p_values[] = {2, 8, 16, 32, 64};

    std::cout << "========================================================\n";
    std::cout << "Simulating Maunam 2.5D MM Compute Workload for N=" << n << "\n";
    std::cout << "========================================================\n";

    for (int p : p_values) {
        // Calculate the tile size exactly as Maunam would
        int sqrt_p_c = std::round(std::sqrt(p / c));
        int tile_size = n / sqrt_p_c;
        int num_elements = tile_size * tile_size;

        std::vector<double> A(num_elements, 1.0);
        std::vector<double> B(num_elements, 2.0);
        std::vector<double> R(num_elements, 0.0);

        // Start Timing
        auto start = std::chrono::high_resolution_clock::now();

        // In 2.5D MM, each processor performs 'sqrt(p/c)' systolic shift steps
        for (int step = 0; step < sqrt_p_c; ++step) {
            local_matmul(A, B, R, tile_size);
        }

        // Stop Timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Processors (p): " << p 
                  << "\t| Tile Size: " << tile_size << "x" << tile_size 
                  << "\t| Compute Time: " << duration.count() << " seconds\n";
    }
    
    std::cout << "========================================================\n";
    return 0;
}