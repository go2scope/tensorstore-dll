
#include "tensorstore_dll/span_fix.h"  // Must come first
#include "tensorstore_dll/tensorstore_dll.h"
#include "tensorstore_dll/version.h"
#include "error_handling.h"

#include "tensorstore/context.h"
#include "tensorstore/driver/zarr/driver.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

struct TSContext {
    tensorstore::Context ctx;
};

extern "C" {

const char* GetVersionString() {
    return TENSORSTORE_DLL_VERSION;
}

void GetVersion(int* major, int* minor, int* patch) {
    if (major) *major = TENSORSTORE_DLL_VERSION_MAJOR;
    if (minor) *minor = TENSORSTORE_DLL_VERSION_MINOR;
    if (patch) *patch = TENSORSTORE_DLL_VERSION_PATCH;
}

TSContext* TSCreateContext() {
    try {
        auto context = new TSContext;
        context->ctx = tensorstore::Context::Default();
        return context;
    } catch (const std::exception&) {
        return nullptr;
    }
}

void TSDestroyContext(TSContext* context) {
    delete context;
}

void TSClearError(TSError* error) {
    if (error && error->message) {
        free((void*)error->message);
        error->message = nullptr;
        error->code = 0;
    }
}

} // extern "C"