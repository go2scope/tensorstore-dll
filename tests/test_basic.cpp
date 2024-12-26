#include "gtest/gtest.h"
#include "tensorstore_dll/tensorstore_dll.h"
#include "tensorstore_dll/version.h"
#include <vector>
#include <cstdint>
#include <filesystem>
#include <memory>

// Custom deleter for RAII handling of TensorStore resources
struct TSContextDeleter {
    void operator()(TSContext* ctx) { TSDestroyContext(ctx); }
};

struct TSDatasetDeleter {
    void operator()(TSDataset* ds) { TSCloseDataset(ds); }
};

using TSContextPtr = std::unique_ptr<TSContext, TSContextDeleter>;
using TSDatasetPtr = std::unique_ptr<TSDataset, TSDatasetDeleter>;

class TensorStoreDLLTest : public ::testing::Test {
protected:
    TSError error{nullptr, 0};
    TSContextPtr context;
    const std::string test_file = "test_basic.zarr";

    void SetUp() override {
        // Create context
        context.reset(TSCreateContext());
        ASSERT_NE(context, nullptr) << "Failed to create context";

        // Remove test file if it exists
        std::filesystem::remove_all(test_file);
    }

    void TearDown() override {
        // Clean up test file
        std::filesystem::remove_all(test_file);
    }

    // Helper to create a basic dataset
    TSDatasetPtr createTestDataset(const int64_t* shape, int rank) {
        const int64_t chunks[] = {32, 32, 32};  // Default chunk size
        const int shard_size_mb = 8;

        TSDataset* dataset = TSCreateZarr(
            context.get(),
            test_file.c_str(),
            TS_UINT16,
            shape,
            rank,
            chunks,
            shard_size_mb,
            &error
        );
        EXPECT_EQ(error.message, nullptr) << "Error creating dataset: " << error.message;
        return TSDatasetPtr(dataset);
    }
};

// Test version information
TEST_F(TensorStoreDLLTest, VersionInfo) {
    const char* version = GetVersionString();
    ASSERT_NE(version, nullptr);
    EXPECT_STRNE(version, "");

    int major, minor, patch;
    GetVersion(&major, &minor, &patch);
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
    EXPECT_GE(patch, 0);
}

// Test context creation and destruction
TEST_F(TensorStoreDLLTest, ContextLifecycle) {
    TSContext* ctx = TSCreateContext();
    ASSERT_NE(ctx, nullptr);
    TSDestroyContext(ctx);
}

// Test dataset creation
TEST_F(TensorStoreDLLTest, DatasetCreation) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    // Verify shape
    int64_t actual_shape[3];
    int actual_rank;
    ASSERT_EQ(TSGetShape(dataset.get(), actual_shape, &actual_rank, &error), 0);
    EXPECT_EQ(error.message, nullptr);
    EXPECT_EQ(actual_rank, 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(actual_shape[i], shape[i]);
    }
}

// Test data writing and reading
TEST_F(TensorStoreDLLTest, DataWriteRead) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    // Create test data
    const int64_t write_origin[] = {0, 0, 0};
    const int64_t write_shape[] = {32, 32, 32};
    const size_t num_elements = 32 * 32 * 32;
    std::vector<uint16_t> write_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        write_data[i] = static_cast<uint16_t>(i % 65536);
    }

    // Write data
    ASSERT_EQ(TSWriteUInt16(dataset.get(), write_origin, write_shape, 
                           write_data.data(), &error), 0);
    EXPECT_EQ(error.message, nullptr);

    // Read data back
    std::vector<uint16_t> read_data(num_elements);
    ASSERT_EQ(TSReadUInt16(dataset.get(), write_origin, write_shape,
                          read_data.data(), &error), 0);
    EXPECT_EQ(error.message, nullptr);

    // Verify data
    for (size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(write_data[i], read_data[i]) 
            << "Data mismatch at index " << i;
    }
}

// Test metadata operations
TEST_F(TensorStoreDLLTest, MetadataOperations) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    // Write metadata
    const char* test_key = "test_key";
    const char* test_value = "test_value";
    ASSERT_EQ(TSSetMetadata(dataset.get(), test_key, test_value, &error), 0);
    EXPECT_EQ(error.message, nullptr);

    // Read metadata
    char read_value[256];
    ASSERT_EQ(TSGetMetadata(dataset.get(), test_key, read_value, 
                           sizeof(read_value), &error), 0);
    EXPECT_EQ(error.message, nullptr);
    EXPECT_STREQ(read_value, test_value);
}

// Test error handling
TEST_F(TensorStoreDLLTest, ErrorHandling) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    // Try to read from invalid coordinates
    const int64_t invalid_origin[] = {100, 100, 100};  // Outside dataset bounds
    const int64_t read_shape[] = {32, 32, 32};
    std::vector<uint16_t> data(32 * 32 * 32);

    EXPECT_NE(TSReadUInt16(dataset.get(), invalid_origin, read_shape,
                          data.data(), &error), 0);
    EXPECT_NE(error.message, nullptr);
    TSClearError(&error);
}

// Test data type handling
TEST_F(TensorStoreDLLTest, DataTypeHandling) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    TSDataType dtype;
    ASSERT_EQ(TSGetDataType(dataset.get(), &dtype, &error), 0);
    EXPECT_EQ(error.message, nullptr);
    EXPECT_EQ(dtype, TS_UINT16);
}

// Test chunk shape retrieval
TEST_F(TensorStoreDLLTest, ChunkShape) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    int64_t chunk_shape[3];
    int chunk_rank;
    ASSERT_EQ(TSGetChunkShape(dataset.get(), chunk_shape, &chunk_rank, &error), 0);
    EXPECT_EQ(error.message, nullptr);
    EXPECT_EQ(chunk_rank, 3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(chunk_shape[i], 32);  // Default chunk size from createTestDataset
    }
}

// Test partial reads and writes
TEST_F(TensorStoreDLLTest, PartialIO) {
    const int64_t shape[] = {64, 64, 64};
    auto dataset = createTestDataset(shape, 3);
    ASSERT_NE(dataset, nullptr);

    // Write to different regions
    const int64_t regions[][3] = {
        {0, 0, 0},    // Corner
        {32, 32, 32}, // Middle
        {63, 63, 63}  // Far corner
    };
    const int64_t small_shape[] = {1, 1, 1};
    
    for (const auto& origin : regions) {
        uint16_t write_value = 42;
        ASSERT_EQ(TSWriteUInt16(dataset.get(), origin, small_shape,
                               &write_value, &error), 0);
        EXPECT_EQ(error.message, nullptr);

        uint16_t read_value = 0;
        ASSERT_EQ(TSReadUInt16(dataset.get(), origin, small_shape,
                              &read_value, &error), 0);
        EXPECT_EQ(error.message, nullptr);
        EXPECT_EQ(read_value, write_value);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}