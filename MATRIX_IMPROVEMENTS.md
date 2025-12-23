# Matrix Implementation Improvements

This document tracks planned improvements and refactorings for the dynamic `Matrix` implementation in `src/matrix/Matrix.zig`.

## Planned Tasks

- [ ] **SIMD Alignment Optimization**: In `simdGemmKernel`, check for memory alignment. Consider using aligned loads if the underlying `items` slice is guaranteed to be aligned, or handle unaligned edges explicitly for better performance.

## Completed Tasks

- [x] **Unify or Clarify Type Conversion**: Unified `as()` and `cast()` into `as()` by updating `meta.as()` to handle rounding when converting from float to integer. Removed redundant `cast()` method from `Matrix.zig` and updated `Point.zig` to use the unified `meta.as()`.
- [x] **Robust LU Decomposition**: Replace the `assert(n == self.cols)` in `lu()` with a runtime check that returns `MatrixError.NotSquare` to maintain consistency with the sticky error chaining pattern. Improved `determinant()`, `qr()`, `rank()`, and `svd()` as well.
- [x] **Standardize Permutation Types**: Reconciled the difference between `LuResult` (uses `Matrix(T)` for permutations) and `QrResult` (uses `[]usize`). Introduced a unified `Permutation` type with `toMatrix()` support and updated both decomposition methods to use it.
- [x] **Standard Library Formatting Integration**: Investigated the usage of `*std.Io.Writer`. Confirmed that in this environment's standard library, `std.Io.Writer` is the standard writer type and `format(self, *Writer)` is the expected signature for `std.fmt`. The current implementation is already correctly integrated.
- [x] **Safe Python Bindings**: Fixed potential null pointer crashes in LU and QR decomposition bindings by restoring null checks and improving cleanup logic when creating the result dictionary.

