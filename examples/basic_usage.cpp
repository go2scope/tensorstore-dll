#include "tensorstore_dll/tensorstore_dll.h"
#include "tensorstore_dll/version.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

// Helper function to check and print errors
void checkError(const TSError* error) {
    if (error && error->message) {
        std::cerr << "Error: " << error->message << std::endl;
        TSClearError(const_cast<TSError*>(error));
        exit(1);
    }
}

int main() {
    TSError error = {nullptr, 0};

    // Print version information
    std::cout << "TensorStore DLL Version: " << GetVersionString() << std::endl;

    // Create context
    TSContext* context = TSCreateContext();
    if (!context) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // Define dataset parameters
    const int64_t shape[] = {100, 100, 100};  // 100x100x100 volume
    const int64_t chunks[] = {32, 32, 32};    // 32x32x32 chunks
    const int rank = 3;
    const int shard_size_mb = 64;

    // Create dataset
    std::cout << "Creating dataset..." << std::endl;
    TSDataset* dataset = TSCreateZarr(
        context,
        "test_volume.zarr",
        TS_UINT16,         // 16-bit unsigned integers
        shape,
        rank,
        chunks,
        shard_size_mb,
        &error
    );
    checkError(&error);

    // Add some metadata
    std::cout << "Setting metadata..." << std::endl;
    TSSetMetadata(dataset, "description", "Test volume data", &error);
    checkError(&error);
    TSSetMetadata(dataset, "units", "micrometers", &error);
    checkError(&error);

    // Write some test data
    std::cout << "Writing data..." << std::endl;
    const int64_t write_origin[] = {0, 0, 0};
    const int64_t write_shape[] = {32, 32, 32};
    const size_t write_size = 32 * 32 * 32;
    
    // Create test pattern: ramp from 0 to 65535
    std::vector<uint16_t> write_data(write_size);
    for (size_t i = 0; i < write_size; ++i) {
        write_data[i] = static_cast<uint16_t>((i * 65535) / write_size);
    }

    TSWriteUInt16(dataset, write_origin, write_shape, write_data.data(), &error);
    checkError(&error);

    // Read back the data
    std::cout << "Reading data..." << std::endl;
    std::vector<uint16_t> read_data(write_size);
    TSReadUInt16(dataset, write_origin, write_shape, read_data.data(), &error);
    checkError(&error);

    // Verify some values
    bool data_correct = true;
    for (size_t i = 0; i < write_size; ++i) {
        if (write_data[i] != read_data[i]) {
            std::cout << "Data mismatch at " << i << ": wrote " 
                     << write_data[i] << " but read " << read_data[i] << std::endl;
            data_correct = false;
            break;
        }
    }
    if (data_correct) {
        std::cout << "Data verification successful!" << std::endl;
    }

    // Read metadata
    char metadata_value[256];
    TSGetMetadata(dataset, "description", metadata_value, sizeof(metadata_value), &error);
    checkError(&error);
    std::cout << "Description: " << metadata_value << std::endl;

    TSGetMetadata(dataset, "units", metadata_value, sizeof(metadata_value), &error);
    checkError(&error);
    std::cout << "Units: " << metadata_value << std::endl;

    // Get dataset shape
    int64_t actual_shape[3];
    int actual_rank;
    TSGetShape(dataset, actual_shape, &actual_rank, &error);
    checkError(&error);

    std::cout << "Dataset shape: [";
    for (int i = 0; i < actual_rank; ++i) {
        std::cout << actual_shape[i];
        if (i < actual_rank - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get chunk shape
    int64_t actual_chunks[3];
    TSGetChunkShape(dataset, actual_chunks, &actual_rank, &error);
    checkError(&error);

    std::cout << "Chunk shape: [";
    for (int i = 0; i < actual_rank; ++i) {
        std::cout << actual_chunks[i];
        if (i < actual_rank - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Cleanup
    TSCloseDataset(dataset);
    TSDestroyContext(context);

    std::cout << "Example completed successfully!" << std::endl;
    return 0;
}