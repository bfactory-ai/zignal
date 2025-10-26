#ifndef ZIGNAL_FORCE_AROCC_PATCHES
#  if defined(__has_include)
#    if __has_include_next(<math.h>)
#      include_next <math.h>
#    else
#      include "/usr/include/math.h"
#    endif
#  else
#    include "/usr/include/math.h"
#  endif
#else

#ifndef ZIGNAL_AROCC_STUB_MATH_H
#define ZIGNAL_AROCC_STUB_MATH_H 1
#define _MATH_H 1

// Minimal math header to satisfy Python's usage during translate-c when arocc
// struggles with glibc's macro-heavy <math.h>.
// This intentionally forgoes full math declarations; it only supplies the
// pieces referenced by the Python headers we import.

#include <float.h>

#ifndef HUGE_VAL
#define HUGE_VAL (__extension__ (1e308 * 1e308))
#endif

#ifndef HUGE_VALF
#define HUGE_VALF ((float)HUGE_VAL)
#endif

#ifndef HUGE_VALL
#define HUGE_VALL ((long double)HUGE_VAL)
#endif

#ifndef INFINITY
#define INFINITY ((float)HUGE_VALF)
#endif

#ifndef NAN
#define NAN (__extension__ ((float)(0.0f / 0.0f)))
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef isnan
#define isnan(x) ((x) != (x))
#endif

#ifndef isinf
#define isinf(x) ((x) == INFINITY || (x) == -INFINITY)
#endif

#ifndef isfinite
#define isfinite(x) (!isnan(x) && !isinf(x))
#endif

#ifndef signbit
#define signbit(x) ((x) < 0)
#endif

#ifndef fmax
#define fmax(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef fmin
#define fmin(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef fabs
#define fabs(x) ((x) < 0 ? -(x) : (x))
#endif

typedef double double_t;
typedef float float_t;

#endif // ZIGNAL_AROCC_STUB_MATH_H

#endif // ZIGNAL_FORCE_AROCC_PATCHES
