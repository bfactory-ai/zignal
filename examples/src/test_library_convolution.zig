const std = @import("std");
const zignal = @import("zignal");
const Image = zignal.Image;
const Rgba = zignal.Rgba;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test kernel (edge detection)
    const kernel = [3][3]f32{
        .{ -1, -1, -1 },
        .{ -1, 8, -1 },
        .{ -1, -1, -1 },
    };

    // Test sizes
    const sizes = [_]usize{ 256, 512, 1024, 2048 };

    std.debug.print("\n=== Testing Library Convolution with Integer SIMD Optimization ===\n", .{});
    std.debug.print("Using CPU's optimal SIMD width for u32: {} pixels per vector\n", .{std.simd.suggestVectorLength(u32) orelse 1});
    std.debug.print("\n", .{});

    for (sizes) |size| {
        // Create test image
        var img = try Image(Rgba).initAlloc(allocator, size, size);
        defer img.deinit(allocator);

        // Fill with test pattern
        for (0..size) |r| {
            for (0..size) |c| {
                const val = @as(u8, @intCast((r + c) % 256));
                img.at(r, c).* = Rgba{ .r = val, .g = val, .b = val, .a = 255 };
            }
        }

        var output = try Image(Rgba).initAlloc(allocator, size, size);
        defer output.deinit(allocator);

        // Warm-up
        try img.convolve(allocator, kernel, &output, .zero);

        // Benchmark
        const iterations = 5;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            try img.convolve(allocator, kernel, &output, .zero);
        }

        const elapsed = timer.read();
        const ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

        std.debug.print("{}x{}: {:.2} ms/iteration\n", .{ size, size, ms });
    }

    // Also test scalar types for comparison
    std.debug.print("\n=== Scalar (u8) Performance for Comparison ===\n", .{});
    for (sizes) |size| {
        var img = try Image(u8).initAlloc(allocator, size, size);
        defer img.deinit(allocator);

        // Fill with test pattern
        for (0..size) |r| {
            for (0..size) |c| {
                img.at(r, c).* = @intCast((r + c) % 256);
            }
        }

        var output = try Image(u8).initAlloc(allocator, size, size);
        defer output.deinit(allocator);

        // Warm-up
        try img.convolve(allocator, kernel, &output, .zero);

        // Benchmark
        const iterations = 5;
        var timer = try std.time.Timer.start();

        for (0..iterations) |_| {
            try img.convolve(allocator, kernel, &output, .zero);
        }

        const elapsed = timer.read();
        const ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0 / @as(f64, @floatFromInt(iterations));

        std.debug.print("{}x{}: {:.2} ms/iteration\n", .{ size, size, ms });
    }
}
