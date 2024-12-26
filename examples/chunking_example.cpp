#include "tensorstore_dll/tensorstore_dll.h"
#include "tensorstore_dll/version.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <filesystem>

// Helper function to check and print errors
void checkError(const TSError* error) {
    if (error && error->message) {
        std::cerr << "Error: " << error->message << std::endl;
        TSClearError(const_cast<TSError*>(error));
        exit(1);
    }
}

// Timer class for benchmarking
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    TimePoint start_time;
    std::string operation;
    double* elapsed_time;

public:
    Timer(const std::string& op, double* elapsed = nullptr) 
        : operation(op), elapsed_time(elapsed) {
        start_time = Clock::now();
    }

    ~Timer() {
        Duration elapsed = Clock::now() - start_time;
        if (elapsed_time) {
            *elapsed_time = elapsed.count();
        }
        std::cout << std::left << std::setw(40) << operation 
                  << ": " << std::fixed << std::setprecision(3) 
                  << elapsed.count() << " seconds" << std::endl;
    }
};

// Function to get file size in MB
double getFileSizeMB(const std::string& filename) {
    std::filesystem::path path(filename);
    if (!std::filesystem::exists(path)) return 0.0;
    
    auto bytes = std::filesystem::file_size(path);
    return static_cast<double>(bytes) / (1024 * 1024);
}

// Compression configuration struct
struct CompressionConfig {
    const char* name;           // Configuration name
    const char* compressor;     // Main compressor (zstd/blosc)
    const char* blosc_subcode;  // Blosc-specific subcompressor
    int compression_level;      // Compression level
    int blosc_blocksize;       // Blosc block size in bytes
    int shuffle;               // Shuffle mode (0=none, 1=byte, 2=bit)
    int num_threads;           // Number of threads for compression
};

// Function to test different read patterns
void testReadPattern(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                    double* elapsed_time, TSError* error) {
    Timer timer("Read", elapsed_time);
    
    size_t total_elements = shape[0] * shape[1] * shape[2];
    std::vector<uint16_t> data(total_elements);
    
    TSReadUInt16(dataset, origin, shape, data.data(), error);
}

// Function to test different write patterns
void testWritePattern(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                     double* elapsed_time, TSError* error) {
    Timer timer("Write", elapsed_time);
    
    size_t total_elements = shape[0] * shape[1] * shape[2];
    std::vector<uint16_t> data(total_elements);

    // Create test pattern that's more compressible
    for (size_t i = 0; i < total_elements; ++i) {
        // Create repeating patterns for better compression
        data[i] = static_cast<uint16_t>((i % 16) * 4096);
    }
    
    TSWriteUInt16(dataset, origin, shape, data.data(), error);
}

int main() {
    TSError error = {nullptr, 0};
    
    // Create context
    TSContext* context = TSCreateContext();
    if (!context) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // Dataset dimensions
    const int64_t volume_shape[] = {256, 256, 256};  // 256³ volume
    const int rank = 3;
    const int64_t chunks[] = {32, 32, 32};  // Use consistent chunk size for comparison
    const int shard_size_mb = 16;

    std::cout << "Testing different compression configurations...\n" << std::endl;

    // Define compression configurations
    CompressionConfig configs[] = {
        // No compression
        {"No compression", "none", nullptr, 0, 0, 0, 1},
        
        // ZSTD configurations
        {"ZSTD light", "zstd", nullptr, 1, 0, 0, 1},
        {"ZSTD balanced", "zstd", nullptr, 3, 0, 0, 1},
        {"ZSTD heavy", "zstd", nullptr, 9, 0, 0, 1},
        
        // Blosc with LZ4
        {"Blosc-LZ4 light", "blosc", "lz4", 1, 256*1024, 1, 4},
        {"Blosc-LZ4 balanced", "blosc", "lz4", 5, 256*1024, 2, 4},
        {"Blosc-LZ4 heavy", "blosc", "lz4", 9, 256*1024, 2, 4},
        
        // Blosc with ZSTD
        {"Blosc-ZSTD light", "blosc", "zstd", 1, 256*1024, 1, 4},
        {"Blosc-ZSTD balanced", "blosc", "zstd", 3, 256*1024, 2, 4},
        {"Blosc-ZSTD heavy", "blosc", "zstd", 9, 256*1024, 2, 4},
        
        // Blosc with BLOSCLZ
        {"Blosc-BLOSCLZ light", "blosc", "blosclz", 1, 256*1024, 1, 4},
        {"Blosc-BLOSCLZ balanced", "blosc", "blosclz", 5, 256*1024, 2, 4},
        {"Blosc-BLOSCLZ heavy", "blosc", "blosclz", 9, 256*1024, 2, 4}
    };

    // Results table header
    std::cout << std::left 
              << std::setw(30) << "Configuration"
              << std::setw(15) << "Size (MB)"
              << std::setw(15) << "Write (s)"
              << std::setw(15) << "Read (s)"
              << std::setw(15) << "Ratio"
              << "\n" << std::string(90, '-') << std::endl;

    // Calculate uncompressed size
    double uncompressed_size = static_cast<double>(volume_shape[0]) * 
                              volume_shape[1] * volume_shape[2] * 
                              sizeof(uint16_t) / (1024 * 1024);

    for (const auto& config : configs) {
        // Create dataset with current configuration
        std::string filename = std::string("compression_test_") + 
                             (config.compressor == std::string("blosc") 
                              ? config.blosc_subcode 
                              : config.compressor) + "_" +
                             std::to_string(config.compression_level) + ".zarr";

        TSDataset* dataset = TSCreateZarrCompressed(
            context,
            filename.c_str(),
            TS_UINT16,
            volume_shape,
            rank,
            chunks,
            shard_size_mb,
            config.compressor,
            config.compression_level,
            &error,
            config.blosc_subcode,    // Blosc-specific parameters
            config.blosc_blocksize,
            config.shuffle,
            config.num_threads
        );
        checkError(&error);

        // Test write performance
        double write_time = 0.0;
        const int64_t origin[] = {0, 0, 0};
        const int64_t shape[] = {128, 128, 128};  // Write a subset of the volume
        testWritePattern(dataset, origin, shape, &write_time, &error);

        // Test read performance
        double read_time = 0.0;
        testReadPattern(dataset, origin, shape, &read_time, &error);

        // Get file size and calculate compression ratio
        double size_mb = getFileSizeMB(filename);
        double compression_ratio = uncompressed_size / size_mb;

        // Print results
        std::cout << std::left 
                  << std::setw(30) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << size_mb
                  << std::setw(15) << std::fixed << std::setprecision(3) << write_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << read_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << compression_ratio
                  << std::endl;

        TSCloseDataset(dataset);
    }

    // Print recommendations
    std::cout << "\nCompression Recommendations:\n" << std::string(50, '-') << std::endl;
    
    std::cout << "1. For fastest write performance:\n"
              << "   - Use Blosc-LZ4 with light compression\n"
              << "   - Enable shuffle mode 1 (byte shuffle)\n"
              << "   - Use 4-8 threads depending on CPU\n";
    
    std::cout << "\n2. For best compression ratio:\n"
              << "   - Use Blosc-ZSTD with heavy compression\n"
              << "   - Enable shuffle mode 2 (bit shuffle)\n"
              << "   - Increase block size for better compression\n";
    
    std::cout << "\n3. For fastest read performance:\n"
              << "   - Use Blosc-LZ4 with balanced compression\n"
              << "   - Enable shuffle mode 1\n"
              << "   - Match block size to common read patterns\n";
    
    std::cout << "\n4. For memory-constrained systems:\n"
              << "   - Use ZSTD (without Blosc) for lower memory usage\n"
              << "   - Reduce block size and number of threads\n"
              << "   - Use lighter compression levels\n";

    std::cout << "\n5. Blosc-specific tips:\n"
              << "   - LZ4: Best for fast compression/decompression\n"
              << "   - ZSTD: Best for high compression ratio\n"
              << "   - BLOSCLZ: Good balance of speed and compression\n"
              << "   - Shuffle improves compression of structured data\n"
              << "   - Block size affects both speed and ratio\n";

    // Cleanup
    TSDestroyContext(context);

    std::cout << "\nCompression example completed successfully!" << std::endl;
    return 0;
}