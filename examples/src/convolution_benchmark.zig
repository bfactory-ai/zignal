const std = @import("std");
const zignal = @import("zignal");
const Image = zignal.Image;
const Rgb = zignal.Rgb;
const Rgba = zignal.Rgba;

const Timer = std.time.Timer;

fn benchmarkConvolve3x3(comptime T: type, allocator: std.mem.Allocator, width: usize, height: usize, iterations: usize) !void {
    // Create test image
    var img = try Image(T).initAlloc(allocator, height, width);
    defer img.deinit(allocator);

    // Fill with random data
    var rng = std.Random.DefaultPrng.init(12345);
    const rand = rng.random();

    switch (@typeInfo(T)) {
        .int, .float => {
            for (img.data) |*pixel| {
                pixel.* = switch (@typeInfo(T)) {
                    .int => rand.int(T),
                    .float => rand.float(T),
                    else => unreachable,
                };
            }
        },
        .@"struct" => {
            for (img.data) |*pixel| {
                inline for (std.meta.fields(T)) |field| {
                    @field(pixel, field.name) = switch (@typeInfo(field.type)) {
                        .int => rand.int(field.type),
                        .float => rand.float(field.type),
                        else => unreachable,
                    };
                }
            }
        },
        else => unreachable,
    }

    // Define a simple 3x3 kernel (Gaussian blur approximation)
    const kernel = [3][3]f32{
        .{ 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 },
        .{ 2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0 },
        .{ 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 },
    };

    var out: Image(T) = try .initAlloc(allocator, height, width);
    defer out.deinit(allocator);

    // Warm-up run
    try img.convolve(allocator, kernel, &out, .replicate);

    // Benchmark
    var timer = try Timer.start();
    const start = timer.read();

    for (0..iterations) |_| {
        try img.convolve(allocator, kernel, &out, .replicate);
    }

    const elapsed = timer.read() - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (ms_per_iter / 1000.0);

    std.debug.print("  {}x{} {s}: {d:.2} ms/iter, {d:.2} Mpixels/sec\n", .{ width, height, @typeName(T), ms_per_iter, pixels_per_sec / 1_000_000.0 });
}

fn benchmarkSobel(allocator: std.mem.Allocator, width: usize, height: usize, iterations: usize) !void {
    // Create RGB test image
    var img = try Image(Rgb).initAlloc(allocator, height, width);
    defer img.deinit(allocator);

    // Fill with random data
    var rng = std.Random.DefaultPrng.init(12345);
    const rand = rng.random();

    for (img.data) |*pixel| {
        pixel.r = rand.int(u8);
        pixel.g = rand.int(u8);
        pixel.b = rand.int(u8);
    }

    var out: Image(u8) = try .initAlloc(allocator, height, width);
    defer out.deinit(allocator);

    // Warm-up run
    try img.sobel(allocator, &out);

    // Benchmark
    var timer = try Timer.start();
    const start = timer.read();

    for (0..iterations) |_| {
        try img.sobel(allocator, &out);
    }

    const elapsed = timer.read() - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const pixels_per_sec = @as(f64, @floatFromInt(width * height)) / (ms_per_iter / 1000.0);

    std.debug.print("  {}x{} Sobel: {d:.2} ms/iter, {d:.2} Mpixels/sec\n", .{ width, height, ms_per_iter, pixels_per_sec / 1_000_000.0 });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Convolution 3x3 Benchmark (SIMD Optimized) ===\n\n", .{});

    const sizes = [_]struct { w: usize, h: usize }{
        .{ .w = 256, .h = 256 },
        .{ .w = 512, .h = 512 },
        .{ .w = 1024, .h = 1024 },
        .{ .w = 2048, .h = 2048 },
    };

    for (sizes) |size| {
        std.debug.print("Image size: {}x{}\n", .{ size.w, size.h });

        // Determine iteration count based on image size
        const base_iters = 100;
        const iter_scale = @max(1, (512 * 512) / (size.w * size.h));
        const iterations = base_iters * iter_scale;

        // Benchmark different pixel types
        try benchmarkConvolve3x3(u8, allocator, size.w, size.h, iterations);
        try benchmarkConvolve3x3(f32, allocator, size.w, size.h, iterations);
        try benchmarkConvolve3x3(Rgb, allocator, size.w, size.h, iterations);
        try benchmarkConvolve3x3(Rgba, allocator, size.w, size.h, iterations);

        // Benchmark Sobel (which uses convolve3x3 internally)
        try benchmarkSobel(allocator, size.w, size.h, iterations);

        std.debug.print("\n", .{});
    }

    std.debug.print("Benchmark complete!\n", .{});
    std.debug.print("\nNote: The convolve3x3 function now uses SIMD optimizations for:\n", .{});
    std.debug.print("  - Scalar types (u8, f32): Process multiple pixels in parallel\n", .{});
    std.debug.print("  - RGBA structs: Use 4-channel SIMD for parallel channel processing\n", .{});
    std.debug.print("  - RGB structs: Standard optimized path\n", .{});
}
