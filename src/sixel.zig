//! Sixel graphics protocol support for image rendering
//!
//! This module provides functionality to convert images to sixel format,
//! which is supported by various terminal emulators for displaying graphics.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Image = @import("image.zig").Image;
const color = @import("color.zig");
const Rgb = color.Rgb;

/// Available palette modes for sixel encoding
pub const PaletteMode = enum {
    /// Fixed 252-color palette with 6x7x6 RGB distribution
    fixed_6x7x6,
    /// Fixed 16-color VGA palette
    fixed_vga16,
    /// Fixed 216-color web-safe palette (6x6x6 RGB cube)
    fixed_web216,
    /// Adaptive palette using median cut algorithm
    adaptive,
};

/// Dithering modes for color quantization
pub const DitherMode = enum {
    /// No dithering, direct color quantization
    none,
    /// Floyd-Steinberg error diffusion dithering
    floyd_steinberg,
    /// Atkinson dithering (used by original Macintosh)
    atkinson,
    /// Automatic selection based on palette size
    auto,
};

/// Options for sixel encoding
pub const SixelOptions = struct {
    /// Palette generation mode
    palette_mode: PaletteMode = .fixed_6x7x6,
    /// Dithering algorithm to use
    dither_mode: DitherMode = .auto,
    /// Maximum colors for adaptive palette (1-256)
    max_colors: u16 = 256,
    /// Maximum output width (image will be scaled if larger)
    max_width: u32 = 800,
    /// Maximum output height (image will be scaled if larger)
    max_height: u32 = 600,
};

/// Result of sixel encoding operation
pub const SixelResult = struct {
    /// The encoded sixel data
    data: []u8,
    /// Final width after scaling
    width: u32,
    /// Final height after scaling
    height: u32,
    /// Number of colors in the palette
    palette_size: u16,
};

/// Converts an image to sixel format
pub fn imageToSixel(
    comptime T: type,
    image: Image(T),
    allocator: Allocator,
    options: SixelOptions,
) ![]u8 {
    // Calculate scaling if needed
    var width = image.cols;
    var height = image.rows;
    var scale_x: f32 = 1.0;
    var scale_y: f32 = 1.0;

    if (width > options.max_width) {
        scale_x = @as(f32, @floatFromInt(options.max_width)) / @as(f32, @floatFromInt(width));
    }
    if (height > options.max_height) {
        scale_y = @as(f32, @floatFromInt(options.max_height)) / @as(f32, @floatFromInt(height));
    }
    const scale = @min(scale_x, scale_y);

    if (scale < 1.0) {
        width = @intFromFloat(@as(f32, @floatFromInt(width)) * scale);
        height = @intFromFloat(@as(f32, @floatFromInt(height)) * scale);
    }

    // Prepare palette based on mode
    var palette: [256]Rgb = undefined;
    var palette_size: usize = 0;

    switch (options.palette_mode) {
        .fixed_6x7x6 => {
            generateFixed6x7x6Palette(&palette);
            palette_size = 252;
        },
        .fixed_vga16 => {
            @memcpy(palette[0..16], &vga16_palette);
            palette_size = 16;
        },
        .fixed_web216 => {
            generateWeb216Palette(&palette);
            palette_size = 216;
        },
        .adaptive => {
            palette_size = try generateAdaptivePalette(T, image, allocator, &palette, options.max_colors);
            if (palette_size == 0) {
                // Fallback to fixed palette if adaptive fails
                generateFixed6x7x6Palette(&palette);
                palette_size = 252;
            }
        },
    }

    // Determine dithering mode
    const dither_mode = switch (options.dither_mode) {
        .auto => if (palette_size <= 16) DitherMode.atkinson else DitherMode.floyd_steinberg,
        else => options.dither_mode,
    };

    // Prepare working data for dithering if needed
    var working_data: ?[]u8 = null;
    if (dither_mode != .none) {
        working_data = try allocator.alloc(u8, width * height * 3);

        // Copy scaled image data to working buffer
        for (0..height) |row| {
            for (0..width) |col| {
                const src_r = if (scale < 1.0) @as(usize, @intFromFloat(@as(f32, @floatFromInt(row)) / scale)) else row;
                const src_c = if (scale < 1.0) @as(usize, @intFromFloat(@as(f32, @floatFromInt(col)) / scale)) else col;

                if (src_r < image.rows and src_c < image.cols) {
                    const pixel = image.at(src_r, src_c).*;
                    const rgb = color.convertColor(Rgb, pixel);
                    const pos = (row * width + col) * 3;
                    working_data.?[pos] = rgb.r;
                    working_data.?[pos + 1] = rgb.g;
                    working_data.?[pos + 2] = rgb.b;
                }
            }
        }

        // Apply dithering
        switch (dither_mode) {
            .floyd_steinberg => FloydSteinbergDither.apply(working_data.?, width, height, palette[0..palette_size], palette_size),
            .atkinson => AtkinsonDither.apply(working_data.?, width, height, palette[0..palette_size], palette_size),
            else => {},
        }
    }
    defer if (working_data) |data| allocator.free(data);

    // Create output buffer - estimate size generously
    var output = std.ArrayList(u8).init(allocator);
    defer output.deinit();

    // Start sixel sequence with DCS, then add raster dimensions
    try output.writer().print("\x1bPq\"1;1;{d};{d}", .{ width, height });

    // Define palette
    switch (options.palette_mode) {
        .fixed_6x7x6 => {
            // For fixed palette, we need to define all colors
            var idx: u16 = 0;
            for (0..6) |r| {
                for (0..7) |g| {
                    for (0..6) |b| {
                        const r_val = (@as(u32, @intCast(r)) * 100 + 2) / 5;
                        const g_val = (@as(u32, @intCast(g)) * 100 + 3) / 6;
                        const b_val = (@as(u32, @intCast(b)) * 100 + 2) / 5;
                        try output.writer().print("#{d};2;{d};{d};{d}", .{ idx, r_val, g_val, b_val });
                        idx += 1;
                    }
                }
            }
        },
        .fixed_vga16, .fixed_web216, .adaptive => {
            // Define only the colors in the palette
            for (0..palette_size) |i| {
                const p = palette[i];
                const r_val = (@as(u32, p.r) * 100 + 127) / 255;
                const g_val = (@as(u32, p.g) * 100 + 127) / 255;
                const b_val = (@as(u32, p.b) * 100 + 127) / 255;
                try output.writer().print("#{d};2;{d};{d};{d}", .{ i, r_val, g_val, b_val });
            }
        },
    }

    // Encode pixels as sixels
    var row: usize = 0;
    while (row < height) : (row += 6) {
        // Build a map of which colors are used in this sixel row
        var colors_used = try allocator.alloc(bool, palette_size);
        defer allocator.free(colors_used);
        @memset(colors_used, false);

        var color_map = try allocator.alloc([800]u8, palette_size);
        defer allocator.free(color_map);
        for (color_map) |*cm| {
            @memset(cm, 0);
        }

        // First pass: build bitmaps for each color
        for (0..width) |col| {
            var sixel_bits = try allocator.alloc(u8, palette_size);
            defer allocator.free(sixel_bits);
            @memset(sixel_bits, 0);

            // Check all 6 pixels in this column
            for (0..6) |bit| {
                const pixel_row = row + bit;
                if (pixel_row < height) {
                    const color_idx = if (working_data) |data| blk: {
                        // Use dithered data
                        const pos = (pixel_row * width + col) * 3;
                        const rgb = Rgb{
                            .r = data[pos],
                            .g = data[pos + 1],
                            .b = data[pos + 2],
                        };
                        break :blk findNearestColor(palette[0..palette_size], rgb);
                    } else blk: {
                        // Use original data without dithering
                        const src_r = if (scale < 1.0) @as(usize, @intFromFloat(@as(f32, @floatFromInt(pixel_row)) / scale)) else pixel_row;
                        const src_c = if (scale < 1.0) @as(usize, @intFromFloat(@as(f32, @floatFromInt(col)) / scale)) else col;

                        if (src_r < image.rows and src_c < image.cols) {
                            const pixel = image.at(src_r, src_c).*;
                            const rgb = color.convertColor(Rgb, pixel);

                            // Direct quantization for fixed palettes when not dithering
                            switch (options.palette_mode) {
                                .fixed_6x7x6 => break :blk @as(u8, @intCast(quantize6x7x6(rgb.r, rgb.g, rgb.b))),
                                .fixed_web216 => break :blk quantizeWeb216(rgb.r, rgb.g, rgb.b),
                                else => break :blk findNearestColor(palette[0..palette_size], rgb),
                            }
                        } else {
                            continue;
                        }
                    };

                    // Set the bit for this color
                    sixel_bits[color_idx] |= @as(u8, 1) << @intCast(bit);
                    colors_used[color_idx] = true;
                }
            }

            // Store the sixel characters for each color
            for (0..palette_size) |c| {
                if (sixel_bits[c] != 0) {
                    color_map[c][col] = sixel_bits[c] + '?'; // Add to '?' to get sixel char
                }
            }
        }

        // Second pass: output sixels for each color
        var current_color: usize = std.math.maxInt(usize); // Invalid color
        for (0..palette_size) |c| {
            if (!colors_used[c]) continue;

            // Select color if different
            if (c != current_color) {
                current_color = c;
                try output.writer().print("#{d}", .{current_color});
            }

            // Output all sixels for this color
            for (0..width) |col| {
                if (color_map[c][col] != 0) {
                    try output.writer().print("{c}", .{color_map[c][col]});
                } else {
                    try output.writer().print("?", .{}); // Empty sixel
                }
            }

            // Carriage return to go back to start of line (except for last color)
            var more_colors = false;
            for (c + 1..palette_size) |nc| {
                if (colors_used[nc]) {
                    more_colors = true;
                    break;
                }
            }
            if (more_colors) {
                try output.writer().print("$", .{}); // Graphics carriage return
            }
        }

        // Move to next sixel row if not at end
        if (row + 6 < height) {
            try output.writer().print("-", .{}); // Graphics new line
        }
    }

    // End sixel sequence with ST
    try output.writer().print("\x1b\\", .{});

    return output.toOwnedSlice();
}

/// Color histogram entry for adaptive palette generation
const ColorCount = struct {
    r: u8,
    g: u8,
    b: u8,
    count: u32,
};

/// Box structure for median cut algorithm
const ColorBox = struct {
    colors: []ColorCount,
    r_min: u8,
    r_max: u8,
    g_min: u8,
    g_max: u8,
    b_min: u8,
    b_max: u8,
    population: u32,

    fn volume(self: ColorBox) u32 {
        if (self.r_max < self.r_min or self.g_max < self.g_min or self.b_max < self.b_min) {
            return 0;
        }
        const r_size = @as(u32, self.r_max) - @as(u32, self.r_min) + 1;
        const g_size = @as(u32, self.g_max) - @as(u32, self.g_min) + 1;
        const b_size = @as(u32, self.b_max) - @as(u32, self.b_min) + 1;
        return r_size * g_size * b_size;
    }

    fn largestDimension(self: ColorBox) u8 {
        const r_range = if (self.r_max >= self.r_min) self.r_max - self.r_min else 0;
        const g_range = if (self.g_max >= self.g_min) self.g_max - self.g_min else 0;
        const b_range = if (self.b_max >= self.b_min) self.b_max - self.b_min else 0;

        if (g_range >= r_range and g_range >= b_range) return 1; // green
        if (r_range >= b_range) return 0; // red
        return 2; // blue
    }
};

/// Finds the nearest color in a palette to the target color
fn findNearestColor(pal: []const Rgb, target: Rgb) u8 {
    var best_idx: u8 = 0;
    var best_dist: u32 = std.math.maxInt(u32);

    for (pal, 0..) |p, idx| {
        const dr = if (target.r > p.r) target.r - p.r else p.r - target.r;
        const dg = if (target.g > p.g) target.g - p.g else p.g - target.g;
        const db = if (target.b > p.b) target.b - p.b else p.b - target.b;
        const dist = @as(u32, dr) * @as(u32, dr) +
            @as(u32, dg) * @as(u32, dg) +
            @as(u32, db) * @as(u32, db);

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = @intCast(idx);
        }
    }
    return best_idx;
}

/// Floyd-Steinberg error diffusion dithering
const FloydSteinbergDither = struct {
    fn diffuseError(data: []u8, w: usize, h: usize, x: usize, y: usize, ch: usize, err: i16) void {
        // Floyd-Steinberg error distribution:
        //          X   7/16
        //  3/16  5/16  1/16

        const diffusePixel = struct {
            fn apply(d: []u8, dw: usize, dh: usize, px: usize, py: usize, channel: usize, e: i16, weight: i16, divisor: i16) void {
                if (px >= dw or py >= dh) return;

                const pos = (py * dw + px) * 3 + channel;
                const new_val = @as(i16, d[pos]) + @divTrunc(e * weight, divisor);
                d[pos] = @intCast(@min(@max(new_val, 0), 255));
            }
        };

        // Distribute error to neighboring pixels
        if (x + 1 < w) {
            diffusePixel.apply(data, w, h, x + 1, y, ch, err, 7, 16); // right
        }
        if (y + 1 < h) {
            if (x > 0) {
                diffusePixel.apply(data, w, h, x - 1, y + 1, ch, err, 3, 16); // bottom-left
            }
            diffusePixel.apply(data, w, h, x, y + 1, ch, err, 5, 16); // bottom
            if (x + 1 < w) {
                diffusePixel.apply(data, w, h, x + 1, y + 1, ch, err, 1, 16); // bottom-right
            }
        }
    }

    pub fn apply(data: []u8, w: usize, h: usize, pal: []const Rgb, pal_size: usize) void {
        for (0..h) |y| {
            for (0..w) |x| {
                const pos = (y * w + x) * 3;
                const original = Rgb{
                    .r = data[pos],
                    .g = data[pos + 1],
                    .b = data[pos + 2],
                };

                // Find nearest palette color
                const idx = findNearestColor(pal[0..pal_size], original);
                const quantized = pal[idx];

                // Calculate and diffuse error for each channel
                const r_error = @as(i16, original.r) - @as(i16, quantized.r);
                const g_error = @as(i16, original.g) - @as(i16, quantized.g);
                const b_error = @as(i16, original.b) - @as(i16, quantized.b);

                // Update pixel to quantized color
                data[pos] = quantized.r;
                data[pos + 1] = quantized.g;
                data[pos + 2] = quantized.b;

                // Diffuse errors
                diffuseError(data, w, h, x, y, 0, r_error);
                diffuseError(data, w, h, x, y, 1, g_error);
                diffuseError(data, w, h, x, y, 2, b_error);
            }
        }
    }
};

/// Atkinson dithering (used by original Macintosh)
const AtkinsonDither = struct {
    fn diffuseError(data: []u8, w: usize, h: usize, x: usize, y: usize, ch: usize, err: i16) void {
        // Atkinson error distribution (only 75% of error is diffused):
        //          X   1/8  1/8
        //   1/8   1/8  1/8
        //         1/8

        const diffusePixel = struct {
            fn apply(d: []u8, dw: usize, dh: usize, px: usize, py: usize, channel: usize, e: i16) void {
                if (px >= dw or py >= dh) return;

                const pos = (py * dw + px) * 3 + channel;
                const new_val = @as(i16, d[pos]) + @divTrunc(e, 8);
                d[pos] = @intCast(@min(@max(new_val, 0), 255));
            }
        };

        // Distribute error to neighboring pixels
        if (x + 1 < w) {
            diffusePixel.apply(data, w, h, x + 1, y, ch, err); // right
            if (x + 2 < w) {
                diffusePixel.apply(data, w, h, x + 2, y, ch, err); // right+1
            }
        }
        if (y + 1 < h) {
            if (x > 0) {
                diffusePixel.apply(data, w, h, x - 1, y + 1, ch, err); // bottom-left
            }
            diffusePixel.apply(data, w, h, x, y + 1, ch, err); // bottom
            if (x + 1 < w) {
                diffusePixel.apply(data, w, h, x + 1, y + 1, ch, err); // bottom-right
            }
        }
        if (y + 2 < h) {
            diffusePixel.apply(data, w, h, x, y + 2, ch, err); // bottom+1
        }
    }

    pub fn apply(data: []u8, w: usize, h: usize, pal: []const Rgb, pal_size: usize) void {
        for (0..h) |y| {
            for (0..w) |x| {
                const pos = (y * w + x) * 3;
                const original = Rgb{
                    .r = data[pos],
                    .g = data[pos + 1],
                    .b = data[pos + 2],
                };

                // Find nearest palette color
                const idx = findNearestColor(pal[0..pal_size], original);
                const quantized = pal[idx];

                // Calculate and diffuse error for each channel
                const r_error = @as(i16, original.r) - @as(i16, quantized.r);
                const g_error = @as(i16, original.g) - @as(i16, quantized.g);
                const b_error = @as(i16, original.b) - @as(i16, quantized.b);

                // Update pixel to quantized color
                data[pos] = quantized.r;
                data[pos + 1] = quantized.g;
                data[pos + 2] = quantized.b;

                // Diffuse errors
                diffuseError(data, w, h, x, y, 0, r_error);
                diffuseError(data, w, h, x, y, 1, g_error);
                diffuseError(data, w, h, x, y, 2, b_error);
            }
        }
    }
};

/// Standard VGA 16-color palette
const vga16_palette = [16]Rgb{
    Rgb{ .r = 0, .g = 0, .b = 0 }, // Black
    Rgb{ .r = 128, .g = 0, .b = 0 }, // Maroon
    Rgb{ .r = 0, .g = 128, .b = 0 }, // Green
    Rgb{ .r = 128, .g = 128, .b = 0 }, // Olive
    Rgb{ .r = 0, .g = 0, .b = 128 }, // Navy
    Rgb{ .r = 128, .g = 0, .b = 128 }, // Purple
    Rgb{ .r = 0, .g = 128, .b = 128 }, // Teal
    Rgb{ .r = 192, .g = 192, .b = 192 }, // Silver
    Rgb{ .r = 128, .g = 128, .b = 128 }, // Gray
    Rgb{ .r = 255, .g = 0, .b = 0 }, // Red
    Rgb{ .r = 0, .g = 255, .b = 0 }, // Lime
    Rgb{ .r = 255, .g = 255, .b = 0 }, // Yellow
    Rgb{ .r = 0, .g = 0, .b = 255 }, // Blue
    Rgb{ .r = 255, .g = 0, .b = 255 }, // Fuchsia
    Rgb{ .r = 0, .g = 255, .b = 255 }, // Cyan
    Rgb{ .r = 255, .g = 255, .b = 255 }, // White
};

/// Quantizes RGB color to fixed 6x7x6 palette index
fn quantize6x7x6(r: u8, g: u8, b: u8) u16 {
    // Quantize to 6 levels for red and blue, 7 for green (most sensitive)
    const r_q = @as(u16, @min(r / 43, 5)); // 0-5 levels
    const g_q = @as(u16, @min(g / 37, 6)); // 0-6 levels
    const b_q = @as(u16, @min(b / 43, 5)); // 0-5 levels
    return r_q * 42 + g_q * 6 + b_q; // 6*7*6 = 252
}

/// Generates the fixed 6x7x6 palette (252 colors)
fn generateFixed6x7x6Palette(palette: []Rgb) void {
    var idx: usize = 0;
    for (0..6) |r| {
        for (0..7) |g| {
            for (0..6) |b| {
                palette[idx] = Rgb{
                    .r = @intCast((r * 255 + 2) / 5),
                    .g = @intCast((g * 255 + 3) / 6),
                    .b = @intCast((b * 255 + 2) / 5),
                };
                idx += 1;
            }
        }
    }
}

/// Quantizes RGB color to web-safe 6x6x6 palette index
fn quantizeWeb216(r: u8, g: u8, b: u8) u8 {
    const r_q = @as(u8, @min(r / 51, 5)); // 0-5 levels
    const g_q = @as(u8, @min(g / 51, 5)); // 0-5 levels
    const b_q = @as(u8, @min(b / 51, 5)); // 0-5 levels
    return r_q * 36 + g_q * 6 + b_q; // 6*6*6 = 216
}

/// Generates the web-safe 216-color palette
fn generateWeb216Palette(palette: []Rgb) void {
    var idx: usize = 0;
    for (0..6) |r| {
        for (0..6) |g| {
            for (0..6) |b| {
                palette[idx] = Rgb{
                    .r = @intCast(r * 51),
                    .g = @intCast(g * 51),
                    .b = @intCast(b * 51),
                };
                idx += 1;
            }
        }
    }
}

/// Generates an adaptive palette using median cut algorithm
fn generateAdaptivePalette(
    comptime T: type,
    image: Image(T),
    allocator: Allocator,
    palette: []Rgb,
    max_colors: u16,
) !usize {
    const HashMap = std.hash_map.HashMap(u16, u32, std.hash_map.AutoContext(u16), 80);
    var histogram = HashMap.init(allocator);
    defer histogram.deinit();

    // Build color histogram (quantized to 5-bit per channel for efficiency)
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            const pixel = image.at(r, c).*;
            const rgb = color.convertColor(Rgb, pixel);

            // Quantize to 5-bit per channel for histogram
            const r5 = rgb.r >> 3;
            const g5 = rgb.g >> 3;
            const b5 = rgb.b >> 3;
            const key = (@as(u16, r5) << 10) | (@as(u16, g5) << 5) | @as(u16, b5);

            const result = try histogram.getOrPut(key);
            if (result.found_existing) {
                result.value_ptr.* += 1;
            } else {
                result.value_ptr.* = 1;
            }
        }
    }

    // Convert histogram to array for median cut
    var color_list = std.ArrayList(ColorCount).init(allocator);
    defer color_list.deinit();

    var iter = histogram.iterator();
    while (iter.next()) |entry| {
        const key = entry.key_ptr.*;
        const count = entry.value_ptr.*;

        const r5 = @as(u8, @intCast((key >> 10) & 0x1F));
        const g5 = @as(u8, @intCast((key >> 5) & 0x1F));
        const b5 = @as(u8, @intCast(key & 0x1F));

        // Convert back to 8-bit with proper scaling
        const r8 = (r5 << 3) | (r5 >> 2);
        const g8 = (g5 << 3) | (g5 >> 2);
        const b8 = (b5 << 3) | (b5 >> 2);

        try color_list.append(.{
            .r = r8,
            .g = g8,
            .b = b8,
            .count = count,
        });
    }

    const palette_size = @min(@min(color_list.items.len, max_colors), palette.len);
    if (palette_size == 0) {
        return 0;
    }

    // Initialize first box with all colors
    var boxes = std.ArrayList(ColorBox).init(allocator);
    defer boxes.deinit();

    var initial_box = ColorBox{
        .colors = color_list.items,
        .r_min = 255,
        .r_max = 0,
        .g_min = 255,
        .g_max = 0,
        .b_min = 255,
        .b_max = 0,
        .population = 0,
    };

    // Find bounds of initial box
    for (color_list.items) |c| {
        initial_box.r_min = @min(initial_box.r_min, c.r);
        initial_box.r_max = @max(initial_box.r_max, c.r);
        initial_box.g_min = @min(initial_box.g_min, c.g);
        initial_box.g_max = @max(initial_box.g_max, c.g);
        initial_box.b_min = @min(initial_box.b_min, c.b);
        initial_box.b_max = @max(initial_box.b_max, c.b);
        initial_box.population += c.count;
    }

    try boxes.append(initial_box);

    // Median cut algorithm
    while (boxes.items.len < palette_size) {
        // Find box with largest volume * population
        var largest_idx: usize = 0;
        var largest_score: u64 = 0;

        for (boxes.items, 0..) |box, i| {
            const score = @as(u64, box.volume()) * @as(u64, box.population);
            if (score > largest_score) {
                largest_score = score;
                largest_idx = i;
            }
        }

        if (largest_score == 0) break; // No more splittable boxes

        // Split the largest box
        var box_to_split = boxes.orderedRemove(largest_idx);

        // Check if box can be split (has different colors)
        if (box_to_split.colors.len <= 1) {
            try boxes.append(box_to_split);
            continue;
        }

        const dim = box_to_split.largestDimension();

        // Sort colors along the chosen dimension
        const SortContext = struct {
            dim: u8,
            pub fn lessThan(ctx: @This(), a: ColorCount, b: ColorCount) bool {
                return switch (ctx.dim) {
                    0 => a.r < b.r,
                    1 => a.g < b.g,
                    else => a.b < b.b,
                };
            }
        };

        std.sort.heap(ColorCount, box_to_split.colors, SortContext{ .dim = dim }, SortContext.lessThan);

        // Find median cut point (weighted by pixel count)
        var total_weight: u64 = 0;
        for (box_to_split.colors) |c| {
            total_weight += c.count;
        }

        const half_weight = total_weight / 2;
        var accumulated_weight: u64 = 0;
        var cut_point: usize = 0;

        for (box_to_split.colors, 0..) |c, i| {
            accumulated_weight += c.count;
            if (accumulated_weight >= half_weight) {
                cut_point = @max(1, @min(i + 1, box_to_split.colors.len - 1));
                break;
            }
        }

        // Create two new boxes
        var box1 = ColorBox{
            .colors = box_to_split.colors[0..cut_point],
            .r_min = 255,
            .r_max = 0,
            .g_min = 255,
            .g_max = 0,
            .b_min = 255,
            .b_max = 0,
            .population = 0,
        };

        var box2 = ColorBox{
            .colors = box_to_split.colors[cut_point..],
            .r_min = 255,
            .r_max = 0,
            .g_min = 255,
            .g_max = 0,
            .b_min = 255,
            .b_max = 0,
            .population = 0,
        };

        // Recalculate bounds for both boxes
        for (box1.colors) |c| {
            box1.r_min = @min(box1.r_min, c.r);
            box1.r_max = @max(box1.r_max, c.r);
            box1.g_min = @min(box1.g_min, c.g);
            box1.g_max = @max(box1.g_max, c.g);
            box1.b_min = @min(box1.b_min, c.b);
            box1.b_max = @max(box1.b_max, c.b);
            box1.population += c.count;
        }

        for (box2.colors) |c| {
            box2.r_min = @min(box2.r_min, c.r);
            box2.r_max = @max(box2.r_max, c.r);
            box2.g_min = @min(box2.g_min, c.g);
            box2.g_max = @max(box2.g_max, c.g);
            box2.b_min = @min(box2.b_min, c.b);
            box2.b_max = @max(box2.b_max, c.b);
            box2.population += c.count;
        }

        if (box1.colors.len > 0 and box1.r_max >= box1.r_min) try boxes.append(box1);
        if (box2.colors.len > 0 and box2.r_max >= box2.r_min) try boxes.append(box2);
    }

    // Generate final palette from boxes (weighted average)
    const actual_size = @min(boxes.items.len, palette.len);
    for (boxes.items[0..actual_size], 0..) |box, i| {
        var r_sum: u64 = 0;
        var g_sum: u64 = 0;
        var b_sum: u64 = 0;
        var weight_sum: u64 = 0;

        for (box.colors) |c| {
            r_sum += @as(u64, c.r) * @as(u64, c.count);
            g_sum += @as(u64, c.g) * @as(u64, c.count);
            b_sum += @as(u64, c.b) * @as(u64, c.count);
            weight_sum += c.count;
        }

        if (weight_sum > 0) {
            palette[i] = .{
                .r = @intCast(@divTrunc(r_sum, weight_sum)),
                .g = @intCast(@divTrunc(g_sum, weight_sum)),
                .b = @intCast(@divTrunc(b_sum, weight_sum)),
            };
        } else {
            // Fallback to center of box if no weight
            palette[i] = .{
                .r = (box.r_min + box.r_max) / 2,
                .g = (box.g_min + box.g_max) / 2,
                .b = (box.b_min + box.b_max) / 2,
            };
        }
    }

    return actual_size;
}

/// Checks if the terminal supports sixel graphics
pub fn isSixelSupported() !bool {
    const allocator = std.heap.page_allocator;

    // Check SIXEL environment variable override
    if (std.process.getEnvVarOwned(allocator, "SIXEL")) |sixel_env| {
        defer allocator.free(sixel_env);
        if (std.mem.eql(u8, sixel_env, "1") or std.mem.eql(u8, sixel_env, "true")) {
            return true;
        }
    } else |_| {}

    // Check TERM environment variable for known sixel-capable terminals
    if (std.process.getEnvVarOwned(allocator, "TERM")) |term| {
        defer allocator.free(term);
        const sixel_terminals = [_][]const u8{
            "xterm-256color",
            "foot",
            "mlterm",
            "yaft-256color",
            "mintty",
            "wezterm",
        };
        for (sixel_terminals) |sixel_term| {
            if (std.mem.indexOf(u8, term, sixel_term) != null) {
                return true;
            }
        }
    } else |_| {}

    return false;
}
