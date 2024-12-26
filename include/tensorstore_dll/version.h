#ifndef TENSORSTORE_DLL_VERSION_H_
#define TENSORSTORE_DLL_VERSION_H_

// Version numbering
#define TENSORSTORE_DLL_VERSION_MAJOR 1
#define TENSORSTORE_DLL_VERSION_MINOR 0
#define TENSORSTORE_DLL_VERSION_PATCH 0

// Version string
#define TENSORSTORE_DLL_VERSION "1.0.0"

// API declaration
#ifdef _WIN32
    #ifdef TENSORSTORE_DLL_EXPORTS
        #define TENSORSTORE_DLL_API __declspec(dllexport)
    #else
        #define TENSORSTORE_DLL_API __declspec(dllimport)
    #endif
#else
    #define TENSORSTORE_DLL_API
#endif

// Version functions
extern "C" {
    TENSORSTORE_DLL_API const char* GetVersionString();
    TENSORSTORE_DLL_API void GetVersion(int* major, int* minor, int* patch);
}

#endif  // TENSORSTORE_DLL_VERSION_H_