#define TENSORSTORE_DLL_EXPORTS
#include "tensorstore_dll.h"

#include "tensorstore/context.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/driver/zarr/driver.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/kvstore/file/driver.h"
#include "tensorstore/index_space/dim_expression.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace tensorstore;

struct TSContext {
    Context ctx;
};

struct TSDataset {
    TensorStore<> store;
    Transaction transaction;
    json metadata;  // Custom metadata storage
};

namespace {

DataType GetTensorStoreType(TSDataType dtype) {
    switch (dtype) {
        case TS_UINT8: return dtype_v<uint8_t>;
        case TS_UINT16: return dtype_v<uint16_t>;
        case TS_UINT32: return dtype_v<uint32_t>;
        default: throw std::runtime_error("Unsupported data type");
    }
}

void SetError(TSError* error, const std::string& message, int code = -1) {
    if (error) {
        error->message = strdup(message.c_str());
        error->code = code;
    }
}

} // namespace

extern "C" {

TSDataset* TSCreateZarr(TSContext* context, const char* path, TSDataType dtype,
                        const int64_t* shape, int rank, const int64_t* chunks,
                        int shard_size_mb, TSError* error) {
    try {
        auto dataset = new TSDataset;
        
        // Create shape and chunks vectors
        std::vector<Index> shape_vec(shape, shape + rank);
        std::vector<Index> chunks_vec(chunks, chunks + rank);
        
        // Configure Zarr options with chunking and sharding
        json zarr_spec = {
            {"driver", "zarr"},
            {"kvstore", {
                {"driver", "file"},
                {"path", path}
            }},
            {"dtype", GetTensorStoreType(dtype).name()},
            {"shape", shape_vec},
            {"chunks", chunks_vec},
            {"metadata", {
                {"zarr_format", 2},
                {"shard_size_mb", shard_size_mb}
            }}
        };
        
        // Create the dataset
        auto future = tensorstore::Open(zarr_spec, context->ctx,
                                      OpenMode::create | OpenMode::delete_existing).result();
                                      
        if (!future.ok()) {
            SetError(error, future.status().ToString());
            delete dataset;
            return nullptr;
        }
        
        dataset->store = future.value();
        dataset->transaction = tensorstore::Transaction();
        dataset->metadata = json::object();  // Initialize empty metadata
        
        return dataset;
    } catch (const std::exception& e) {
        SetError(error, e.what());
        return nullptr;
    }
}

template<typename T>
int TSReadTyped(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                T* data, TSError* error) {
    try {
        auto rank = dataset->store.rank();
        std::vector<Index> origin_vec(origin, origin + rank);
        std::vector<Index> shape_vec(shape, shape + rank);
        
        auto read_future = tensorstore::Read(
            dataset->store | dataset->transaction,
            BoxView<>(origin_vec, shape_vec),
            ElementPointer<T>(data)
        ).result();
        
        if (!read_future.ok()) {
            SetError(error, read_future.status().ToString());
            return -1;
        }
        return 0;
    } catch (const std::exception& e) {
        SetError(error, e.what());
        return -1;
    }
}

template<typename T>
int TSWriteTyped(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                 const T* data, TSError* error) {
    try {
        auto rank = dataset->store.rank();
        std::vector<Index> origin_vec(origin, origin + rank);
        std::vector<Index> shape_vec(shape, shape + rank);
        
        auto write_future = tensorstore::Write(
            ElementPointer<const T>(data),
            dataset->store | dataset->transaction,
            BoxView<>(origin_vec, shape_vec)
        ).result();
        
        if (!write_future.ok()) {
            SetError(error, write_future.status().ToString());
            return -1;
        }
        return 0;
    } catch (const std::exception& e) {
        SetError(error, e.what());
        return -1;
    }
}

// Implement type-specific read operations
int TSReadUInt8(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                unsigned char* data, TSError* error) {
    return TSReadTyped(dataset, origin, shape, data, error);
}

int TSReadUInt16(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                 unsigned short* data, TSError* error) {
    return TSReadTyped(dataset, origin, shape, data, error);
}

int TSReadUInt32(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                 uint32_t* data, TSError* error) {
    return TSReadTyped(dataset, origin, shape, data, error);
}

// Implement type-specific write operations
int TSWriteUInt8(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                 const unsigned char* data, TSError* error) {
    return TSWriteTyped(dataset, origin, shape, data, error);
}

int TSWriteUInt16(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                  const unsigned short* data, TSError* error) {
    return TSWriteTyped(dataset, origin, shape, data, error);
}

int TSWriteUInt32(TSDataset* dataset, const int64_t* origin, const int64_t* shape,
                  const uint32_t* data, TSError* error) {
    return TSWriteTyped(dataset, origin, shape, data, error);
}

// Metadata operations
int TSSetMetadata(TSDataset* dataset, const char* key, const char* value, TSError* error) {
    try {
        dataset->metadata[key] = value;
        
        // Write metadata to .zattrs file
        json attrs = {
            {"custom_metadata", dataset->metadata}
        };
        
        auto kvstore = dataset->store.kvstore();
        auto write_future = kvstore.Write(".zattrs", tensorstore::serialize_json(attrs)).result();
        
        if (!write_future.ok()) {
            SetError(error, write_future.status().ToString());
            return -1;
        }
        return 0;
    } catch (const std::exception& e) {
        SetError(error, e.what());
        return -1;
    }
}

int TSGetMetadata(TSDataset* dataset, const char* key, char* value,
                  size_t value_size, TSError* error) {
    try {
        auto it = dataset->metadata.find(key);
        if (it == dataset->metadata.end()) {
            SetError(error, "Metadata key not found");
            return -1;
        }
        
        std::string str_value = it->get<std::string>();
        if (str_value.size() >= value_size) {
            SetError(error, "Buffer too small for metadata value");
            return -1;
        }
        
        strncpy(value, str_value.c_str(), value_size);
        return 0;
    } catch (const std::exception& e) {
        SetError(error, e.what());
        return -1;
    }
}

// ... (rest of the implementation remains the same)

} // extern "C"