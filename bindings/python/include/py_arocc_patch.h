#ifndef ZIGNAL_PY_AROCC_PATCH_H
#define ZIGNAL_PY_AROCC_PATCH_H

// Helper shims that paper over arocc translate-c incompatibilities in glibc & Python headers.
// Keep everything in this single header so we can delete it when arocc fully supports these constructs.

#include <stdbool.h>

#ifndef ZIGNAL_APPLY_AROCC_PATCHES
#  ifdef ZIGNAL_FORCE_AROCC_PATCHES
#    define ZIGNAL_APPLY_AROCC_PATCHES 1
#  else
#    define ZIGNAL_APPLY_AROCC_PATCHES 0
#  endif
#endif

#if ZIGNAL_APPLY_AROCC_PATCHES

#if defined(ZIGNAL_REQUIRE_FLOATN_TYPES)
// -----------------------------------------------------------------------------
// FloatN fallbacks
// -----------------------------------------------------------------------------
// arocc currently lacks support for the ISO/IEC TS 18661 _FloatN keywords.
// Provide cheap typedefs so headers like <wchar.h> and <stdlib.h> can declare
// their conversion helpers.
typedef float _Float32;
typedef double _Float64;
typedef double _Float32x;
typedef long double _Float64x;
#endif

// -----------------------------------------------------------------------------
// Atomic built-in fallbacks
// -----------------------------------------------------------------------------
// arocc's translate-c currently treats __atomic_* built-ins as void-returning,
// which breaks Python's inline helpers. Provide simple, non-atomic replacements
// so the headers remain translatable. These macros intentionally sacrifice
// memory-ordering guarantees; the generated Zig bindings never call these paths.

#ifndef ZIGNAL_ATOMIC_BARRIER
#define ZIGNAL_ATOMIC_BARRIER(order) ((void)(order))
#endif

#ifndef __atomic_load_n
#define __atomic_load_n(ptr, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __val = *__ptr; \
        ZIGNAL_ATOMIC_BARRIER(order); \
        __val; \
    })
#endif

#ifndef __atomic_store_n
#define __atomic_store_n(ptr, value, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        *__ptr = (value); \
        ZIGNAL_ATOMIC_BARRIER(order); \
    })
#endif

#ifndef __atomic_exchange_n
#define __atomic_exchange_n(ptr, value, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __old = *__ptr; \
        *__ptr = (value); \
        ZIGNAL_ATOMIC_BARRIER(order); \
        __old; \
    })
#endif

#ifndef __atomic_fetch_add
#define __atomic_fetch_add(ptr, value, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __old = *__ptr; \
        *__ptr = __old + (value); \
        ZIGNAL_ATOMIC_BARRIER(order); \
        __old; \
    })
#endif

#ifndef __atomic_fetch_and
#define __atomic_fetch_and(ptr, value, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __old = *__ptr; \
        *__ptr = __old & (value); \
        ZIGNAL_ATOMIC_BARRIER(order); \
        __old; \
    })
#endif

#ifndef __atomic_fetch_or
#define __atomic_fetch_or(ptr, value, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __old = *__ptr; \
        *__ptr = __old | (value); \
        ZIGNAL_ATOMIC_BARRIER(order); \
        __old; \
    })
#endif

#ifndef __atomic_compare_exchange_n
#define __atomic_compare_exchange_n(ptr, expected, desired, weak, success, failure) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __exp_ptr = (expected); \
        __auto_type __desired_val = (desired); \
        (void)(weak); \
        bool __match = (*__ptr == *__exp_ptr); \
        if (__match) { \
            *__ptr = __desired_val; \
        } else { \
            *__exp_ptr = *__ptr; \
        } \
        ZIGNAL_ATOMIC_BARRIER(success); \
        ZIGNAL_ATOMIC_BARRIER(failure); \
        __match; \
    })
#endif

#ifndef __atomic_thread_fence
#define __atomic_thread_fence(order) ZIGNAL_ATOMIC_BARRIER(order)
#endif

#ifndef __atomic_store
#define __atomic_store(ptr, value_ptr, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __val_ptr = (value_ptr); \
        *__ptr = *__val_ptr; \
        ZIGNAL_ATOMIC_BARRIER(order); \
    })
#endif

#ifndef __atomic_load
#define __atomic_load(ptr, result_ptr, order) \
    __extension__ ({ \
        __auto_type __ptr = (ptr); \
        __auto_type __res_ptr = (result_ptr); \
        *__res_ptr = *__ptr; \
        ZIGNAL_ATOMIC_BARRIER(order); \
    })
#endif

#endif // ZIGNAL_APPLY_AROCC_PATCHES

#endif // ZIGNAL_PY_AROCC_PATCH_H
