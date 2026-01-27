const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const zignal = @import("zignal");

const args = @import("args.zig");
const display = @import("display.zig");
const common = @import("common.zig");

const Args = struct {
    output: ?[]const u8 = null,
    scale: ?f32 = null,
    threshold: ?u8 = null,
    binary: bool = false,
    display: bool = false,
    width: ?u32 = null,
    height: ?u32 = null,
    protocol: ?[]const u8 = null,
    filter: ?[]const u8 = null,

    pub const meta = .{
        .output = .{ .help = "Path to save the difference image", .metavar = "path" },
        .scale = .{ .help = "Scale factor for difference visibility (default: 1.0)", .metavar = "float" },
        .threshold = .{ .help = "Ignore differences smaller than this value (0-255)", .metavar = "int" },
        .binary = .{ .help = "Produce a binary output (white for difference, black for match)" },
        .display = .{ .help = "Display the result in the terminal (default if no output file)" },
        .width = .{ .help = "Width of each sub-image for display", .metavar = "N" },
        .height = .{ .help = "Height of each sub-image for display", .metavar = "N" },
        .protocol = .{ .help = "Force display protocol: kitty, sixel, sgr, braille, auto", .metavar = "p" },
        .filter = .{ .help = "Interpolation filter for display resizing", .metavar = "name" },
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

    const should_display = parsed.options.display or parsed.options.output == null;

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

    if (parsed.options.output) |output_path| {
        std.log.info("Saving difference image to '{s}'...", .{output_path});
        try diff_img.save(io, gpa, output_path);
    }

    if (should_display) {
        // Calculate proportional scale factor based on user constraints
        // Using img1 dimensions as reference (img2 and diff_img are same size)
        const user_scale = zignal.terminal.aspectScale(
            parsed.options.width,
            parsed.options.height,
            img1.rows,
            img1.cols,
        );

        const w = @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(img1.cols)) * user_scale)));
        const h = @as(u32, @intFromFloat(@round(@as(f32, @floatFromInt(img1.rows)) * user_scale)));

        const filter = try common.resolveFilter(parsed.options.filter);

        // 3 images side-by-side
        const canvas_w = 3 * w;
        const canvas_h = h;

        var canvas = try zignal.Image(zignal.Rgba(u8)).init(gpa, canvas_h, canvas_w);
        defer canvas.deinit(gpa);
        canvas.fill(zignal.Rgba(u8).black);

        const wf = @as(f32, @floatFromInt(w));
        const hf = @as(f32, @floatFromInt(h));

        // Insert Image 1
        canvas.insert(img1, .{ .l = 0, .t = 0, .r = wf, .b = hf }, 0, filter, .none);

        // Insert Image 2
        canvas.insert(img2, .{ .l = wf, .t = 0, .r = 2 * wf, .b = hf }, 0, filter, .none);

        // Insert Difference
        canvas.insert(diff_img, .{ .l = 2 * wf, .t = 0, .r = 3 * wf, .b = hf }, 0, filter, .none);

        try display.displayCanvas(io, writer, &canvas, parsed.options.protocol, filter);
    }
}
