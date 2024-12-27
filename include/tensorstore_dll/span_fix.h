#ifndef TENSORSTORE_DLL_SPAN_FIX_H_
#define TENSORSTORE_DLL_SPAN_FIX_H_

#if defined(_MSC_VER)
#include <gsl/gsl>
namespace tensorstore {
template <typename T, std::ptrdiff_t Extent = gsl::dynamic_extent>
using span = gsl::span<T, Extent>;
}
#define TENSORSTORE_INTERNAL_SPAN_INCLUDED 1
#endif

#endif // TENSORSTORE_DLL_SPAN_FIX_H_