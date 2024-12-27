#ifndef TENSORSTORE_DLL_ERROR_HANDLING_H_
#define TENSORSTORE_DLL_ERROR_HANDLING_H_

#include "tensorstore_dll/tensorstore_dll.h"
#include "absl/status/status.h"
#include <string>

void SetError(TSError* error, const std::string& message, int code = -1);
void SetError(TSError* error, const absl::Status& status);

#endif // TENSORSTORE_DLL_ERROR_HANDLING_H_