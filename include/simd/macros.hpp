#pragma once

#if defined(_MSC_VER)
#   define SIMD_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#   define SIMD_INLINE __inline__ __attribute__((__always_inline__))
#else
#   define SIMD_INLINE inline
#endif
