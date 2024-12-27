#include "error_handling.h"
#include <cstring>

void SetError(TSError* error, const std::string& message, int code) {
    if (error) {
        #ifdef _WIN32
        error->message = _strdup(message.c_str());
        #else
        error->message = strdup(message.c_str());
        #endif
        error->code = code;
    }
}

void SetError(TSError* error, const absl::Status& status) {
    if (error) {
        #ifdef _WIN32
        error->message = _strdup(status.ToString().c_str());
        #else
        error->message = strdup(status.ToString().c_str());
        #endif
        error->code = static_cast<int>(status.code());
    }
}