#ifndef TENSORSTORE_DLL_H_
#define TENSORSTORE_DLL_H_

#ifdef _WIN32
    #ifdef TENSORSTORE_DLL_EXPORTS
        #define TENSORSTORE_DLL_API __declspec(dllexport)
    #else
        #define TENSORSTORE_DLL_API __declspec(dllimport)
    #endif
#else
    #define TENSORSTORE_DLL_API
#endif

#include <cstdint>

extern "C" {

// Error handling
struct TSError {
    const char* message;
    int code;
};

// Opaque handle types
typedef struct TSContext TSContext;
typedef struct TSDataset TSDataset;

// Data types
typedef enum {
    TS_UINT8,   // unsigned char
    TS_UINT16,  // unsigned short
    TS_UINT32   // uint32_t
} TSDataType;

// Context management
TENSORSTORE_DLL_API TSContext* TSCreateContext();
TENSORSTORE_DLL_API void TSDestroyContext(TSContext* context);

// Error handling
TENSORSTORE_DLL_API void TSClearError(TSError* error);

} // extern "C"

#endif // TENSORSTORE_DLL_H_
