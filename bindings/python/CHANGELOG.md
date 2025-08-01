# Python Bindings Changelog

## [0.2.0] - 2025-07-25

### Breaking Changes
- **Image Internal Storage**: The `Image` class (formerly `ImageRgb`) now uses RGBA internal storage
  - Enables SIMD optimizations for 2-5x performance improvements
  - Increases memory usage by ~33% (4 bytes per pixel instead of 3)
  - External API remains unchanged - RGB images are automatically converted

### Added
- **Feature Distribution Matching (FDM)**: New module for advanced color transfer algorithms
  - Transfer color distributions between images
  - Domain adaptation capabilities
- **Zero-Copy NumPy Integration**: Efficient conversion without data copying
  - `to_numpy()` method for Image objects
  - `from_numpy()` coming in future release
- **Image Interpolation**: Access to all Zignal interpolation methods
  - Methods: nearest, bilinear, bicubic, catmull_rom, lanczos, mitchell
- **Rich Display Support**: `__format__` method for terminal and notebook display
  - ANSI full/half-block, Sixel, Kitty, Braille, and formats
- **Type Stubs**: Comprehensive `.pyi` files for IDE support
  - Full type hints for all classes and methods
  - Improved autocomplete and static analysis

### Changed
- Image class can now save to PNG format (`.png` extension required)
- Smart versioning system using `build.zig.zon` as source of truth
- Optimized wheel builds with stripped debug symbols (50% size reduction)

### Distribution
- Automated wheel building for Python 3.8-3.13
- Support for Linux, macOS (Intel & ARM), and Windows
- CI/CD pipeline for automated PyPI releases

## [0.1.0] - 2024-07-21

### Initial Release

### Core Features
- **Complete Color Space Support**: All 12 Zignal color spaces
  - RGB types: `Rgb`, `Rgba`
  - Perceptual: `Hsl`, `Hsv`
  - Lab family: `Lab`, `Lch`
  - Modern: `Oklab`, `Oklch`
  - Device: `Xyz`, `Lms`
  - Specialized: `Xyb`, `Ycbcr`
- **Image Class**: Basic image I/O and manipulation
  - Load PNG and JPEG images
  - Save to PNG format
  - Access image dimensions via `rows` and `cols` properties
- **Type Safety**: Proper validation and error handling
- **Pythonic API**: Natural property access and method calls
- **High Performance**: Zero-copy operations where possible

### Architecture
- Pure Zig implementation using Python C API
- No Python runtime dependencies
- Efficient memory management
- Cross-platform support
