#ifndef ZIGNAL_PYTHON_VISIBILITY_SHIM_H
#define ZIGNAL_PYTHON_VISIBILITY_SHIM_H

#ifndef __has_warning
#  define __has_warning(x) 0
#endif

#if defined(_WIN32)

// Prevent exports.h from redefining the PyAPI macros with __declspec decorations
#define Py_EXPORTS_H
#define Py_IMPORTED_SYMBOL
#define Py_EXPORTED_SYMBOL
#define Py_LOCAL_SYMBOL
#ifdef __cplusplus
#  define PyAPI_FUNC(RTYPE) RTYPE
#  define PyAPI_DATA(RTYPE) extern RTYPE
#  define PyMODINIT_FUNC extern "C" PyObject*
#else
#  define PyAPI_FUNC(RTYPE) RTYPE
#  define PyAPI_DATA(RTYPE) extern RTYPE
#  define PyMODINIT_FUNC PyObject*
#endif

#include <Python.h>

#else

#include <stdbool.h>

// Skip Python's exports.h so we can provide attribute-free linkage macros.
#ifndef Py_EXPORTS_H
#define Py_EXPORTS_H
#  define Py_IMPORTED_SYMBOL
#  define Py_EXPORTED_SYMBOL
#  define Py_LOCAL_SYMBOL
#  ifndef PyAPI_FUNC
#    define PyAPI_FUNC(RTYPE) RTYPE
#  endif
#  ifndef PyAPI_DATA
#    define PyAPI_DATA(RTYPE) extern RTYPE
#  endif
#  ifndef PyMODINIT_FUNC
#    ifdef __cplusplus
#      define PyMODINIT_FUNC extern "C" PyObject*
#    else
#      define PyMODINIT_FUNC PyObject*
#    endif
#  endif
#endif

// Provide soft-fallbacks for GCC atomic builtins that Zig's translator cannot
// model yet. These degrade to plain loads/stores but keep signatures intact.
#define __zignal_atomic_fetch_add(obj, value) \
    ({ __auto_type _obj = (obj); __auto_type _val = (value); \
       __auto_type _old = *_obj; *_obj = _old + _val; _old; })

#define __zignal_atomic_fetch_and(obj, value) \
    ({ __auto_type _obj = (obj); __auto_type _val = (value); \
       __auto_type _old = *_obj; *_obj = _old & _val; _old; })

#define __zignal_atomic_fetch_or(obj, value) \
    ({ __auto_type _obj = (obj); __auto_type _val = (value); \
       __auto_type _old = *_obj; *_obj = _old | _val; _old; })

#define __zignal_atomic_exchange_n(obj, value) \
    ({ __auto_type _obj = (obj); __auto_type _val = (value); \
       __auto_type _old = *_obj; *_obj = _val; _old; })

#define __zignal_atomic_load_n(obj) \
    ({ __auto_type _obj = (obj); *_obj; })

#define __zignal_atomic_store_n(obj, value) \
    do { __auto_type _obj = (obj); *_obj = (value); } while (0)

#define __zignal_atomic_compare_exchange_n(obj, expected, desired) \
    ({ __auto_type _obj = (obj); __auto_type _expected = (expected); \
       __auto_type _desired = (desired); \
       __auto_type _old = *_obj; bool _match = (_old == *_expected); \
       if (_match) { *_obj = _desired; } else { *_expected = _old; } \
       _match; })

#define __atomic_fetch_add(obj, value, order) __zignal_atomic_fetch_add((obj), (value))
#define __atomic_fetch_and(obj, value, order) __zignal_atomic_fetch_and((obj), (value))
#define __atomic_fetch_or(obj, value, order) __zignal_atomic_fetch_or((obj), (value))
#define __atomic_exchange_n(obj, value, order) __zignal_atomic_exchange_n((obj), (value))
#define __atomic_load_n(obj, order) __zignal_atomic_load_n((obj))
#define __atomic_store_n(obj, value, order) __zignal_atomic_store_n((obj), (value))
#define __atomic_compare_exchange_n(obj, expected, desired, weak, success, failure) \
    __zignal_atomic_compare_exchange_n((obj), (expected), (desired))
#define __atomic_thread_fence(order) ((void)0)

#include <Python.h>

#endif // _WIN32

#endif // ZIGNAL_PYTHON_VISIBILITY_SHIM_H
