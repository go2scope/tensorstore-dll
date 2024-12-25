#ifndef TENSORSTORE_DLL_H_
#define TENSORSTORE_DLL_H_

#ifdef TENSORSTORE_DLL_EXPORTS
    #define TENSORSTORE_DLL_API __declspec(dllexport)
#else
    #define TENSORSTORE_DLL_API __declspec(dllimport)
#endif

#include <cstdint>

extern "C" {

// Supported data types enum
typedef enum {
    TS_UINT8,   // unsigned char
    TS_UINT16,  // unsigned short
    TS_UINT32   // uint32_t
} TSDataType;

// Error handling
struct TSError {
    const char* message;
    int code;
};

// Opaque handle types
typedef struct TSContext TSContext;
typedef struct TSDataset TSDataset;

// Context management
TENSORSTORE_DLL_API TSContext* TSCreateContext();
TENSORSTORE_DLL_API void TSDestroyContext(TSContext* context);

// Dataset creation with chunking and sharding
TENSORSTORE_DLL_API TSDataset* TSCreateZarr(
    TSContext* context,
    const char* path,
    TSDataType dtype,
    const int64_t* shape,
    int rank,
    const int64_t* chunks,  // Chunk size for each dimension
    int shard_size_mb,      // Target shard size in MB
    TSError* error
);

// Dataset opening
TENSORSTORE_DLL_API TSDataset* TSOpenZarr(
    TSContext* context,
    const char* path,
    const char* mode,  // "r" for read, "w" for write
    TSError* error
);

TENSORSTORE_DLL_API void TSCloseDataset(TSDataset* dataset);

// Type-specific read operations
TENSORSTORE_DLL_API int TSReadUInt8(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    unsigned char* data,
    TSError* error
);

TENSORSTORE_DLL_API int TSReadUInt16(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    unsigned short* data,
    TSError* error
);

TENSORSTORE_DLL_API int TSReadUInt32(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    uint32_t* data,
    TSError* error
);

// Type-specific write operations
TENSORSTORE_DLL_API int TSWriteUInt8(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    const unsigned char* data,
    TSError* error
);

TENSORSTORE_DLL_API int TSWriteUInt16(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    const unsigned short* data,
    TSError* error
);

TENSORSTORE_DLL_API int TSWriteUInt32(
    TSDataset* dataset,
    const int64_t* origin,
    const int64_t* shape,
    const uint32_t* data,
    TSError* error
);

// Metadata operations
TENSORSTORE_DLL_API int TSSetMetadata(
    TSDataset* dataset,
    const char* key,
    const char* value,
    TSError* error
);

TENSORSTORE_DLL_API int TSGetMetadata(
    TSDataset* dataset,
    const char* key,
    char* value,
    size_t value_size,
    TSError* error
);

TENSORSTORE_DLL_API int TSListMetadata(
    TSDataset* dataset,
    char** keys,
    size_t* num_keys,
    TSError* error
);

// Dataset information
TENSORSTORE_DLL_API int TSGetShape(
    TSDataset* dataset,
    int64_t* shape,
    int* rank,
    TSError* error
);

TENSORSTORE_DLL_API int TSGetChunkShape(
    TSDataset* dataset,
    int64_t* chunks,
    int* rank,
    TSError* error
);

TENSORSTORE_DLL_API int TSGetDataType(
    TSDataset* dataset,
    TSDataType* dtype,
    TSError* error
);

// Error handling
TENSORSTORE_DLL_API void TSClearError(TSError* error);

} // extern "C"

#endif  // TENSORSTORE_DLL_H_