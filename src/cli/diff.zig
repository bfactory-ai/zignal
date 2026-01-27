const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const zignal = @import("zignal");

const args = @import("args.zig");

const Args = struct {
    output: ?[]const u8 = null,
    scale: ?f32 = null,
    threshold: ?u8 = null,
    binary: bool = false,

    pub const meta = .{
        .output = .{ .help = "Path to save the difference image", .metavar = "path" },
        .scale = .{ .help = "Scale factor for difference visibility (default: 1.0)", .metavar = "float" },
        .threshold = .{ .help = "Ignore differences smaller than this value (0-255)", .metavar = "int" },
    };
};

pub const description = "Compute the visual difference between two images.";

pub const help = args.generateHelp(
    Args,
    "zignal diff <image1> <image2> [options]",
    description,
);

pub fn run(io: Io, writer: *std.Io.Writer, gpa: Allocator, iterator: *std.process.Args.Iterator) !void {
    const parsed = try args.parse(Args, gpa, iterator);
    defer parsed.deinit(gpa);

    if (parsed.help) {
        try args.printHelp(writer, help);
        return;
    }

    if (parsed.positionals.len != 2) {
        std.log.err("Expected exactly two input images.", .{});
        try args.printHelp(writer, help);
        return;
    }

    const path1 = parsed.positionals[0];
    const path2 = parsed.positionals[1];

    // Load images
    var img1 = zignal.Image(zignal.Rgba(u8)).load(io, gpa, path1) catch |err| {
        std.log.err("Failed to load image '{s}': {}", .{ path1, err });
        return;
    };
    defer img1.deinit(gpa);

    var img2 = zignal.Image(zignal.Rgba(u8)).load(io, gpa, path2) catch |err| {
        std.log.err("Failed to load image '{s}': {}", .{ path2, err });
        return;
    };
    defer img2.deinit(gpa);

    if (img1.rows != img2.rows or img1.cols != img2.cols) {
        std.log.err("Dimension mismatch: {d}x{d} vs {d}x{d}", .{
            img1.cols, img1.rows, img2.cols, img2.rows,
        });
        return;
    }

    const output_path = parsed.options.output orelse "diff.png";
    const scale = parsed.options.scale orelse 1.0;
    const threshold = parsed.options.threshold orelse 0;
    const binary = parsed.options.binary;

    std.log.info("Computing difference...", .{});

    var diff_img = try zignal.Image(zignal.Rgba(u8)).init(gpa, img1.rows, img1.cols);
    defer diff_img.deinit(gpa);

    // Compute difference using the library method
    const DiffMode = zignal.Image(zignal.Rgba(u8)).DiffMode;
    const mode: DiffMode = if (binary) .{ .binary = @floatFromInt(threshold) } else .absolute;

    try img1.diff(img2, diff_img, mode);

    var max_diff: u8 = 0;
    var total_diff_pixels: usize = 0;

    // Process for statistics and visualization
    for (0..diff_img.rows) |r| {
        for (0..diff_img.cols) |c| {
            const pixel = diff_img.at(r, c);
            const dr = pixel.r;
            const dg = pixel.g;
            const db = pixel.b;
            const da = pixel.a; // Note: for binary mode, this might be 0 or 255 depending on diff logic

            const local_max = @max(@max(dr, dg), @max(db, da));
            if (local_max > max_diff) max_diff = local_max;

            if (binary) {
                // In binary mode, any non-zero channel means difference > threshold
                if (local_max > 0) {
                    total_diff_pixels += 1;
                    // Ensure alpha is 255 for visibility if it was set to 0 by diff logic
                    pixel.a = 255;
                } else {
                    // Make sure background is opaque black
                    pixel.a = 255;
                }
            } else {
                // In absolute mode, we need to check threshold manually for stats
                if (dr > threshold or dg > threshold or db > threshold or da > threshold) {
                    total_diff_pixels += 1;
                }

                // Apply scaling for visualization
                if (scale != 1.0) {
                    pixel.r = zignal.meta.clamp(u8, @as(f32, @floatFromInt(dr)) * scale);
                    pixel.g = zignal.meta.clamp(u8, @as(f32, @floatFromInt(dg)) * scale);
                    pixel.b = zignal.meta.clamp(u8, @as(f32, @floatFromInt(db)) * scale);
                }

                // Force opaque for visualization so the difference color is visible
                pixel.a = 255;
            }
        }
    }

    std.log.info("Max difference found: {d}", .{max_diff});
    std.log.info("Pixels differing > {d}: {d}", .{ threshold, total_diff_pixels });
    std.log.info("Saving difference image to '{s}'...", .{output_path});

    try diff_img.save(io, gpa, output_path);
}
