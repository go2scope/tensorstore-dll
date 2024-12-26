#include "tensorstore_dll/tensorstore_dll.h"
#include "tensorstore_dll/version.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>

// Helper function to check and print errors
void checkError(const TSError* error) {
    if (error && error->message) {
        std::cerr << "Error: " << error->message << std::endl;
        TSClearError(const_cast<TSError*>(error));
        exit(1);
    }
}

// Helper to get current timestamp as string
std::string getCurrentTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

int main() {
    TSError error = {nullptr, 0};

    // Create context
    TSContext* context = TSCreateContext();
    if (!context) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }

    // Create a small dataset for metadata examples
    const int64_t shape[] = {10, 10, 10};
    const int64_t chunks[] = {5, 5, 5};
    const int rank = 3;
    const int shard_size_mb = 1;

    std::cout << "Creating dataset with metadata..." << std::endl;
    TSDataset* dataset = TSCreateZarr(
        context,
        "metadata_example.zarr",
        TS_UINT16,
        shape,
        rank,
        chunks,
        shard_size_mb,
        &error
    );
    checkError(&error);

    // Set basic metadata
    std::cout << "\nSetting basic metadata..." << std::endl;
    TSSetMetadata(dataset, "title", "Metadata Example Dataset", &error);
    checkError(&error);
    TSSetMetadata(dataset, "created", getCurrentTimestamp().c_str(), &error);
    checkError(&error);
    TSSetMetadata(dataset, "version", "1.0", &error);
    checkError(&error);

    // Set dimensional metadata
    std::cout << "\nSetting dimensional metadata..." << std::endl;
    TSSetMetadata(dataset, "dimension.x.units", "micrometers", &error);
    checkError(&error);
    TSSetMetadata(dataset, "dimension.x.scale", "0.5", &error);
    checkError(&error);
    TSSetMetadata(dataset, "dimension.y.units", "micrometers", &error);
    checkError(&error);
    TSSetMetadata(dataset, "dimension.y.scale", "0.5", &error);
    checkError(&error);
    TSSetMetadata(dataset, "dimension.z.units", "micrometers", &error);
    checkError(&error);
    TSSetMetadata(dataset, "dimension.z.scale", "1.0", &error);
    checkError(&error);

    // Set acquisition metadata
    std::cout << "\nSetting acquisition metadata..." << std::endl;
    TSSetMetadata(dataset, "acquisition.instrument", "Example Microscope", &error);
    checkError(&error);
    TSSetMetadata(dataset, "acquisition.operator", "John Doe", &error);
    checkError(&error);
    TSSetMetadata(dataset, "acquisition.date", getCurrentTimestamp().c_str(), &error);
    checkError(&error);
    TSSetMetadata(dataset, "acquisition.exposure", "100ms", &error);
    checkError(&error);

    // Set processing metadata
    std::cout << "\nSetting processing metadata..." << std::endl;
    TSSetMetadata(dataset, "processing.software", "TensorStore DLL Example", &error);
    checkError(&error);
    TSSetMetadata(dataset, "processing.version", GetVersionString(), &error);
    checkError(&error);
    TSSetMetadata(dataset, "processing.date", getCurrentTimestamp().c_str(), &error);
    checkError(&error);

    // Read back and verify metadata
    std::cout << "\nReading metadata...\n" << std::endl;
    char value[256];

    // Function to read and print metadata
    auto printMetadata = [&](const char* key) {
        TSGetMetadata(dataset, key, value, sizeof(value), &error);
        if (!error.message) {
            std::cout << std::left << std::setw(30) << key << ": " << value << std::endl;
        }
        TSClearError(&error);
    };

    // Print basic metadata
    std::cout << "Basic Metadata:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    printMetadata("title");
    printMetadata("created");
    printMetadata("version");
    std::cout << std::endl;

    // Print dimensional metadata
    std::cout << "Dimensional Metadata:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    printMetadata("dimension.x.units");
    printMetadata("dimension.x.scale");
    printMetadata("dimension.y.units");
    printMetadata("dimension.y.scale");
    printMetadata("dimension.z.units");
    printMetadata("dimension.z.scale");
    std::cout << std::endl;

    // Print acquisition metadata
    std::cout << "Acquisition Metadata:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    printMetadata("acquisition.instrument");
    printMetadata("acquisition.operator");
    printMetadata("acquisition.date");
    printMetadata("acquisition.exposure");
    std::cout << std::endl;

    // Print processing metadata
    std::cout << "Processing Metadata:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    printMetadata("processing.software");
    printMetadata("processing.version");
    printMetadata("processing.date");
    std::cout << std::endl;

    // List all metadata keys
    std::cout << "Listing all metadata keys:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    char* keys[100];  // Assume maximum 100 metadata keys
    size_t num_keys = 0;
    TSListMetadata(dataset, keys, &num_keys, &error);
    checkError(&error);

    for (size_t i = 0; i < num_keys; ++i) {
        TSGetMetadata(dataset, keys[i], value, sizeof(value), &error);
        if (!error.message) {
            std::cout << std::left << std::setw(30) << keys[i] << ": " << value << std::endl;
        }
        TSClearError(&error);
    }

    // Cleanup
    TSCloseDataset(dataset);
    TSDestroyContext(context);

    std::cout << "\nMetadata example completed successfully!" << std::endl;
    return 0;
}