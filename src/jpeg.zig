//! Pure Zig JPEG decoder and baseline encoder implementation.
//! Decoder supports baseline and progressive DCT JPEG images.
//! Encoder implements baseline (SOF0) JPEG with 4:4:4 sampling and adjustable quality.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const convertColor = @import("color.zig").convertColor;
const Image = @import("image.zig").Image;
const Rgb = @import("color.zig").Rgb;
const Ycbcr = @import("color.zig").Ycbcr;

// JPEG signature: 2 bytes that identify a JPEG file (SOI marker)
pub const signature = [_]u8{ 0xFF, 0xD8 };

/// Zigzag scan order for 8x8 DCT blocks
pub const zigzag = [64]u8{
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

// -----------------------------
// Encoder: public API and types
// -----------------------------

pub const JpegSubsampling = enum {
    yuv444,
    // Note: additional subsampling modes can be added later (yuv422, yuv420)
};

pub const EncodeOptions = struct {
    quality: u8 = 90,
    subsampling: JpegSubsampling = .yuv444,
    density_dpi: u16 = 72,
    comment: ?[]const u8 = null,
    pub const default: EncodeOptions = .{};
};

/// Save Image to JPEG file with baseline encoding.
pub fn save(comptime T: type, allocator: Allocator, image: Image(T), file_path: []const u8) !void {
    const bytes = try encodeImage(T, allocator, image, .default);
    defer allocator.free(bytes);

    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    try file.writeAll(bytes);
}

/// Encode an image into baseline JPEG bytes (SOF0, 8-bit, Huffman).
/// Supports grayscale (u8) and RGB (Rgb). Other types are converted to RGB.
pub fn encodeImage(comptime T: type, allocator: Allocator, image: Image(T), options: EncodeOptions) ![]u8 {
    switch (T) {
        u8 => return encodeGrayscale(allocator, image.asBytes(), @intCast(image.cols), @intCast(image.rows), options),
        Rgb => return encodeRgb(allocator, image, options),
        else => {
            var rgb = try Image(Rgb).init(allocator, image.rows, image.cols);
            defer rgb.deinit(allocator);
            for (image.data, 0..) |pix, i| rgb.data[i] = convertColor(Rgb, pix);
            return encodeRgb(allocator, rgb, options);
        },
    }
}

// -----------------------------
// Encoder: internals
// -----------------------------

const StdTables = struct {
    // Base quantization tables (ITU T.81, Annex K)
    const q_luma_base: [64]u8 = .{
        16, 11, 10, 16, 24,  40,  51,  61,
        12, 12, 14, 19, 26,  58,  60,  55,
        14, 13, 16, 24, 40,  57,  69,  56,
        14, 17, 22, 29, 51,  87,  80,  62,
        18, 22, 37, 56, 68,  109, 103, 77,
        24, 35, 55, 64, 81,  104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    };
    const q_chroma_base: [64]u8 = .{
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    };

    // Standard Huffman tables (bits and values)
    // Luminance DC
    const bits_dc_luma: [16]u8 = .{ 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
    const val_dc_luma: [12]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    // Chrominance DC
    const bits_dc_chroma: [16]u8 = .{ 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
    const val_dc_chroma: [12]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

    // Luminance AC
    const bits_ac_luma: [16]u8 = .{ 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125 };
    const val_ac_luma: [162]u8 = .{
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    };
    // Chrominance AC
    const bits_ac_chroma: [16]u8 = .{ 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119 };
    const val_ac_chroma: [162]u8 = .{
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    };
};

const HuffmanEncoder = struct {
    codes: [256]u16 = [1]u16{0} ** 256,
    sizes: [256]u8 = [1]u8{0} ** 256,
};

fn buildHuffmanEncoder(bits: []const u8, vals: []const u8) HuffmanEncoder {
    var enc = HuffmanEncoder{};
    var code: u16 = 0;
    var k: usize = 0;
    for (0..16) |i| {
        const nb = bits[i];
        for (0..nb) |_| {
            const sym = vals[k];
            enc.codes[sym] = code;
            enc.sizes[sym] = @intCast(i + 1);
            code += 1;
            k += 1;
        }
        code = code << 1;
    }
    return enc;
}

const EntropyWriter = struct {
    gpa: Allocator,
    data: std.ArrayList(u8),
    bit_buf: u32 = 0,
    bit_count: u5 = 0,

    pub fn init(gpa: Allocator) EntropyWriter {
        return .{ .gpa = gpa, .data = .empty };
    }
    pub fn deinit(self: *EntropyWriter) void {
        self.data.deinit(self.gpa);
    }

    fn writeByte(self: *EntropyWriter, b: u8) !void {
        try self.data.append(self.gpa, b);
        if (b == 0xFF) try self.data.append(self.gpa, 0x00);
    }
    fn writeBits(self: *EntropyWriter, code: u32, size: u5) !void {
        self.bit_buf = (self.bit_buf << size) | (code & ((@as(u32, 1) << size) - 1));
        self.bit_count += size;
        while (self.bit_count >= 8) {
            const shard = (self.bit_buf >> (self.bit_count - 8)) & 0xFF;
            const out: u8 = @intCast(shard);
            try self.writeByte(out);
            self.bit_count -= 8;
        }
    }
    fn flush(self: *EntropyWriter) !void {
        if (self.bit_count > 0) {
            const pad: u5 = 8 - self.bit_count;
            try self.writeBits((@as(u32, 1) << pad) - 1, pad);
        }
    }
};

fn writeMarker(dst: *std.ArrayList(u8), gpa: Allocator, marker: u16) !void {
    try dst.append(gpa, 0xFF);
    try dst.append(gpa, @intCast(marker & 0xFF));
}

fn writeSegment(dst: *std.ArrayList(u8), gpa: Allocator, marker: u16, payload: []const u8) !void {
    try writeMarker(dst, gpa, marker);
    const len: u16 = @intCast(payload.len + 2);
    try dst.appendSlice(gpa, std.mem.asBytes(&std.mem.nativeTo(u16, len, .big)));
    try dst.appendSlice(gpa, payload);
}

fn scaleQuantTables(quality: u8, ql: *[64]u8, qc: *[64]u8) void {
    const q = @max(@as(u8, 1), @min(@as(u8, 100), quality));
    const scale: i32 = if (q < 50)
        @divTrunc(@as(i32, 5000), @as(i32, q))
    else
        200 - @as(i32, q) * 2;
    for (0..64) |i| {
        const l = @divTrunc((@as(i32, StdTables.q_luma_base[i]) * scale + 50), 100);
        const c = @divTrunc((@as(i32, StdTables.q_chroma_base[i]) * scale + 50), 100);
        ql[i] = @intCast(@max(1, @min(255, l)));
        qc[i] = @intCast(@max(1, @min(255, c)));
    }
}

fn writeDQT(dst: *std.ArrayList(u8), gpa: Allocator, ql: *const [64]u8, qc: *const [64]u8) !void {
    var tmp = std.ArrayList(u8).empty;
    defer tmp.deinit(gpa);

    // Luma table (8-bit precision, id 0)
    try tmp.append(gpa, 0x00);
    for (0..64) |i| try tmp.append(gpa, ql[zigzag[i]]);

    // Chroma table (8-bit precision, id 1)
    try tmp.append(gpa, 0x01);
    for (0..64) |i| try tmp.append(gpa, qc[zigzag[i]]);

    try writeSegment(dst, gpa, 0xFFDB, tmp.items);
}

fn writeSOF0(dst: *std.ArrayList(u8), gpa: Allocator, width: u16, height: u16, grayscale: bool) !void {
    var tmp = std.ArrayList(u8).empty;
    defer tmp.deinit(gpa);
    try tmp.append(gpa, 8); // precision
    try tmp.appendSlice(gpa, std.mem.asBytes(&std.mem.nativeTo(u16, height, .big)));
    try tmp.appendSlice(gpa, std.mem.asBytes(&std.mem.nativeTo(u16, width, .big)));
    if (grayscale) {
        try tmp.append(gpa, 1);
        try tmp.append(gpa, 1); // comp id
        try tmp.append(gpa, 0x11); // sampling 1x1
        try tmp.append(gpa, 0); // quant table id 0
    } else {
        try tmp.append(gpa, 3);
        // Y
        try tmp.append(gpa, 1);
        try tmp.append(gpa, 0x11); // 4:4:4
        try tmp.append(gpa, 0);
        // Cb
        try tmp.append(gpa, 2);
        try tmp.append(gpa, 0x11);
        try tmp.append(gpa, 1);
        // Cr
        try tmp.append(gpa, 3);
        try tmp.append(gpa, 0x11);
        try tmp.append(gpa, 1);
    }
    try writeSegment(dst, gpa, 0xFFC0, tmp.items);
}

fn writeAPP0_JFIF(dst: *std.ArrayList(u8), gpa: Allocator, density_dpi: u16) !void {
    var tmp = std.ArrayList(u8).empty;
    defer tmp.deinit(gpa);
    try tmp.appendSlice(gpa, "JFIF\x00");
    try tmp.append(gpa, 1); // version major
    try tmp.append(gpa, 1); // version minor
    try tmp.append(gpa, 1); // units: dots per inch
    try tmp.appendSlice(gpa, std.mem.asBytes(&std.mem.nativeTo(u16, density_dpi, .big)));
    try tmp.appendSlice(gpa, std.mem.asBytes(&std.mem.nativeTo(u16, density_dpi, .big)));
    try tmp.append(gpa, 0); // x thumbnail
    try tmp.append(gpa, 0); // y thumbnail
    try writeSegment(dst, gpa, 0xFFE0, tmp.items);
}

fn writeCOM(dst: *std.ArrayList(u8), gpa: Allocator, comment: []const u8) !void {
    try writeSegment(dst, gpa, 0xFFFE, comment);
}

fn writeDHT(dst: *std.ArrayList(u8), gpa: Allocator) !void {
    var tmp = std.ArrayList(u8).empty;
    defer tmp.deinit(gpa);

    // DC Luma (class 0, id 0)
    try tmp.append(gpa, 0x00);
    try tmp.appendSlice(gpa, &StdTables.bits_dc_luma);
    try tmp.appendSlice(gpa, &StdTables.val_dc_luma);
    // AC Luma (class 1, id 0)
    try tmp.append(gpa, 0x10);
    try tmp.appendSlice(gpa, &StdTables.bits_ac_luma);
    try tmp.appendSlice(gpa, &StdTables.val_ac_luma);
    // DC Chroma (class 0, id 1)
    try tmp.append(gpa, 0x01);
    try tmp.appendSlice(gpa, &StdTables.bits_dc_chroma);
    try tmp.appendSlice(gpa, &StdTables.val_dc_chroma);
    // AC Chroma (class 1, id 1)
    try tmp.append(gpa, 0x11);
    try tmp.appendSlice(gpa, &StdTables.bits_ac_chroma);
    try tmp.appendSlice(gpa, &StdTables.val_ac_chroma);

    try writeSegment(dst, gpa, 0xFFC4, tmp.items);
}

fn writeSOS(dst: *std.ArrayList(u8), gpa: Allocator, grayscale: bool) !void {
    var tmp = std.ArrayList(u8).empty;
    defer tmp.deinit(gpa);
    if (grayscale) {
        try tmp.append(gpa, 1);
        try tmp.append(gpa, 1); // component id
        try tmp.append(gpa, 0x00); // DC 0, AC 0
    } else {
        try tmp.append(gpa, 3);
        try tmp.append(gpa, 1); // Y
        try tmp.append(gpa, 0x00);
        try tmp.append(gpa, 2); // Cb
        try tmp.append(gpa, 0x11);
        try tmp.append(gpa, 3); // Cr
        try tmp.append(gpa, 0x11);
    }
    try tmp.append(gpa, 0); // Ss
    try tmp.append(gpa, 63); // Se
    try tmp.append(gpa, 0); // Ah/Al
    try writeSegment(dst, gpa, 0xFFDA, tmp.items);
}

fn magnitudeCategory(value: i32) u5 {
    if (value == 0) return 0;
    var v: u32 = @intCast(if (value < 0) -value else value);
    var c: u5 = 0;
    while (v != 0) : (v >>= 1) c += 1;
    return c;
}

fn magnitudeBits(value: i32, mag: u5) u32 {
    if (mag == 0) return 0;
    if (value >= 0) return @intCast(value);
    const base: i32 = (@as(i32, 1) << @intCast(mag)) - 1;
    return @intCast(base + value);
}

fn fdct8x8(src: *const [64]f32, out: *[64]f32) void {
    const pi: f32 = std.math.pi;
    var cosx: [8][8]f32 = undefined;
    var cosy: [8][8]f32 = undefined;
    for (0..8) |u| {
        for (0..8) |x| {
            cosx[u][x] = std.math.cos(@as(f32, (2 * @as(f32, @floatFromInt(x)) + 1) * @as(f32, @floatFromInt(u)) * pi / 16.0));
        }
    }
    for (0..8) |v| {
        for (0..8) |y| {
            cosy[v][y] = std.math.cos(@as(f32, (2 * @as(f32, @floatFromInt(y)) + 1) * @as(f32, @floatFromInt(v)) * pi / 16.0));
        }
    }

    for (0..8) |v| {
        for (0..8) |u| {
            var sum: f32 = 0.0;
            for (0..8) |y| {
                for (0..8) |x| {
                    sum += src[y * 8 + x] * cosx[u][x] * cosy[v][y];
                }
            }
            const cu: f32 = if (u == 0) 1.0 / std.math.sqrt(2.0) else 1.0;
            const cv: f32 = if (v == 0) 1.0 / std.math.sqrt(2.0) else 1.0;
            out[v * 8 + u] = 0.25 * cu * cv * sum;
        }
    }
}

fn encodeBlocksRgb(
    allocator: Allocator,
    image: Image(Rgb),
    ql_f: *const [64]f32,
    qc_f: *const [64]f32,
    writer: *EntropyWriter,
    dc_luma: *const HuffmanEncoder,
    ac_luma: *const HuffmanEncoder,
    dc_chroma: *const HuffmanEncoder,
    ac_chroma: *const HuffmanEncoder,
) !void {
    _ = allocator; // reserved

    const rows: usize = image.rows;
    const cols: usize = image.cols;
    const block_rows: usize = (rows + 7) / 8;
    const block_cols: usize = (cols + 7) / 8;

    var prev_dc_y: i32 = 0;
    var prev_dc_cb: i32 = 0;
    var prev_dc_cr: i32 = 0;

    var block: [64]f32 = undefined;
    var dct: [64]f32 = undefined;

    for (0..block_rows) |br| {
        for (0..block_cols) |bc| {
            // Y
            for (0..8) |y| {
                const iy = @min(rows - 1, br * 8 + y);
                for (0..8) |x| {
                    const ix = @min(cols - 1, bc * 8 + x);
                    const rgb = image.at(iy, ix).*;
                    const ycbcr = convertColor(Ycbcr, rgb);
                    block[y * 8 + x] = ycbcr.y - 128.0;
                }
            }
            fdct8x8(&block, &dct);

            var coeff_y: [64]i32 = undefined;
            for (0..64) |i| coeff_y[i] = @intFromFloat(@round(dct[i] / ql_f[i]));

            // Encode Y block
            const dc_y = coeff_y[0];
            var diff = dc_y - prev_dc_y;
            prev_dc_y = dc_y;
            var mag = magnitudeCategory(diff);
            try writer.writeBits(@intCast(dc_luma.codes[mag]), @as(u5, @intCast(dc_luma.sizes[mag])));
            if (mag > 0) try writer.writeBits(magnitudeBits(diff, mag), mag);

            var run: u8 = 0;
            for (1..64) |k| {
                const v = coeff_y[zigzag[k]];
                if (v == 0) {
                    run += 1;
                    if (run == 16) {
                        try writer.writeBits(@intCast(ac_luma.codes[0xF0]), @as(u5, @intCast(ac_luma.sizes[0xF0])));
                        run = 0;
                    }
                } else {
                    while (run >= 16) : (run -= 16) {
                        try writer.writeBits(@intCast(ac_luma.codes[0xF0]), @as(u5, @intCast(ac_luma.sizes[0xF0])));
                    }
                    const amag = magnitudeCategory(v);
                    const sym: u8 = (run << 4) | @as(u8, amag);
                    try writer.writeBits(@intCast(ac_luma.codes[sym]), @as(u5, @intCast(ac_luma.sizes[sym])));
                    try writer.writeBits(magnitudeBits(v, amag), amag);
                    run = 0;
                }
            }
            if (run > 0) try writer.writeBits(@intCast(ac_luma.codes[0x00]), @as(u5, @intCast(ac_luma.sizes[0x00])));

            // Cb
            for (0..8) |y| {
                const iy = @min(rows - 1, br * 8 + y);
                for (0..8) |x| {
                    const ix = @min(cols - 1, bc * 8 + x);
                    const rgb = image.at(iy, ix).*;
                    const ycbcr = convertColor(Ycbcr, rgb);
                    block[y * 8 + x] = ycbcr.cb - 128.0;
                }
            }
            fdct8x8(&block, &dct);
            var coeff_cb: [64]i32 = undefined;
            for (0..64) |i| coeff_cb[i] = @intFromFloat(@round(dct[i] / qc_f[i]));

            const dc_cb = coeff_cb[0];
            diff = dc_cb - prev_dc_cb;
            prev_dc_cb = dc_cb;
            mag = magnitudeCategory(diff);
            try writer.writeBits(@intCast(dc_chroma.codes[mag]), @as(u5, @intCast(dc_chroma.sizes[mag])));
            if (mag > 0) try writer.writeBits(magnitudeBits(diff, mag), mag);

            run = 0;
            for (1..64) |k| {
                const v = coeff_cb[zigzag[k]];
                if (v == 0) {
                    run += 1;
                    if (run == 16) {
                        try writer.writeBits(@intCast(ac_chroma.codes[0xF0]), @as(u5, @intCast(ac_chroma.sizes[0xF0])));
                        run = 0;
                    }
                } else {
                    while (run >= 16) : (run -= 16) {
                        try writer.writeBits(@intCast(ac_chroma.codes[0xF0]), @as(u5, @intCast(ac_chroma.sizes[0xF0])));
                    }
                    const amag2 = magnitudeCategory(v);
                    const sym2: u8 = (run << 4) | @as(u8, amag2);
                    try writer.writeBits(@intCast(ac_chroma.codes[sym2]), @as(u5, @intCast(ac_chroma.sizes[sym2])));
                    try writer.writeBits(magnitudeBits(v, amag2), amag2);
                    run = 0;
                }
            }
            if (run > 0) try writer.writeBits(@intCast(ac_chroma.codes[0x00]), @as(u5, @intCast(ac_chroma.sizes[0x00])));

            // Cr
            for (0..8) |y| {
                const iy = @min(rows - 1, br * 8 + y);
                for (0..8) |x| {
                    const ix = @min(cols - 1, bc * 8 + x);
                    const rgb = image.at(iy, ix).*;
                    const ycbcr = convertColor(Ycbcr, rgb);
                    block[y * 8 + x] = ycbcr.cr - 128.0;
                }
            }
            fdct8x8(&block, &dct);
            var coeff_cr: [64]i32 = undefined;
            for (0..64) |i| coeff_cr[i] = @intFromFloat(@round(dct[i] / qc_f[i]));

            const dc_cr = coeff_cr[0];
            diff = dc_cr - prev_dc_cr;
            prev_dc_cr = dc_cr;
            mag = magnitudeCategory(diff);
            try writer.writeBits(@intCast(dc_chroma.codes[mag]), @as(u5, @intCast(dc_chroma.sizes[mag])));
            if (mag > 0) try writer.writeBits(magnitudeBits(diff, mag), mag);

            run = 0;
            for (1..64) |k| {
                const v = coeff_cr[zigzag[k]];
                if (v == 0) {
                    run += 1;
                    if (run == 16) {
                        try writer.writeBits(@intCast(ac_chroma.codes[0xF0]), @as(u5, @intCast(ac_chroma.sizes[0xF0])));
                        run = 0;
                    }
                } else {
                    while (run >= 16) : (run -= 16) {
                        try writer.writeBits(@intCast(ac_chroma.codes[0xF0]), @as(u5, @intCast(ac_chroma.sizes[0xF0])));
                    }
                    const amag3 = magnitudeCategory(v);
                    const sym3: u8 = (run << 4) | @as(u8, amag3);
                    try writer.writeBits(@intCast(ac_chroma.codes[sym3]), @as(u5, @intCast(ac_chroma.sizes[sym3])));
                    try writer.writeBits(magnitudeBits(v, amag3), amag3);
                    run = 0;
                }
            }
            if (run > 0) try writer.writeBits(@intCast(ac_chroma.codes[0x00]), @as(u5, @intCast(ac_chroma.sizes[0x00])));
        }
    }
}

fn encodeRgb(allocator: Allocator, image: Image(Rgb), options: EncodeOptions) ![]u8 {
    var out = std.ArrayList(u8).empty;
    defer out.deinit(allocator);

    // SOI
    try out.append(allocator, 0xFF);
    try out.append(allocator, 0xD8);

    try writeAPP0_JFIF(&out, allocator, options.density_dpi);
    if (options.comment) |c| try writeCOM(&out, allocator, c);

    var ql: [64]u8 = undefined;
    var qc: [64]u8 = undefined;
    scaleQuantTables(options.quality, &ql, &qc);
    try writeDQT(&out, allocator, &ql, &qc);
    try writeSOF0(&out, allocator, @intCast(image.cols), @intCast(image.rows), false);
    try writeDHT(&out, allocator);
    try writeSOS(&out, allocator, false);

    // Entropy-coded data
    var ew = EntropyWriter.init(allocator);
    defer ew.deinit();

    // Build Huffman encoders
    const dc_luma = buildHuffmanEncoder(&StdTables.bits_dc_luma, &StdTables.val_dc_luma);
    const ac_luma = buildHuffmanEncoder(&StdTables.bits_ac_luma, &StdTables.val_ac_luma);
    const dc_chroma = buildHuffmanEncoder(&StdTables.bits_dc_chroma, &StdTables.val_dc_chroma);
    const ac_chroma = buildHuffmanEncoder(&StdTables.bits_ac_chroma, &StdTables.val_ac_chroma);

    // Float quant tables for math
    var ql_f: [64]f32 = undefined;
    var qc_f: [64]f32 = undefined;
    for (0..64) |i| {
        ql_f[i] = @floatFromInt(ql[i]);
        qc_f[i] = @floatFromInt(qc[i]);
    }

    try encodeBlocksRgb(allocator, image, &ql_f, &qc_f, &ew, &dc_luma, &ac_luma, &dc_chroma, &ac_chroma);
    try ew.flush();

    // Append entropy-coded bytes into output
    try out.appendSlice(allocator, ew.data.items);

    // EOI
    try out.append(allocator, 0xFF);
    try out.append(allocator, 0xD9);

    return out.toOwnedSlice(allocator);
}

fn encodeGrayscale(allocator: Allocator, bytes: []const u8, width: u32, height: u32, options: EncodeOptions) ![]u8 {
    _ = options.subsampling; // not used for grayscale

    var out = std.ArrayList(u8).empty;
    defer out.deinit(allocator);

    // SOI
    try out.append(allocator, 0xFF);
    try out.append(allocator, 0xD8);

    try writeAPP0_JFIF(&out, allocator, options.density_dpi);
    if (options.comment) |c| try writeCOM(&out, allocator, c);

    var ql: [64]u8 = undefined;
    var dummy: [64]u8 = undefined;
    scaleQuantTables(options.quality, &ql, &dummy);
    // Only luma table used
    var tmp_dqt = std.ArrayList(u8).empty;
    defer tmp_dqt.deinit(allocator);
    try tmp_dqt.append(allocator, 0x00);
    for (0..64) |i| try tmp_dqt.append(allocator, ql[zigzag[i]]);
    try writeSegment(&out, allocator, 0xFFDB, tmp_dqt.items);

    try writeSOF0(&out, allocator, @intCast(width), @intCast(height), true);

    // DHT: only DC/AC luma
    var tmp_dht = std.ArrayList(u8).empty;
    defer tmp_dht.deinit(allocator);
    try tmp_dht.appendSlice(allocator, &[_]u8{0x00});
    try tmp_dht.appendSlice(allocator, &StdTables.bits_dc_luma);
    try tmp_dht.appendSlice(allocator, &StdTables.val_dc_luma);
    try tmp_dht.appendSlice(allocator, &[_]u8{0x10});
    try tmp_dht.appendSlice(allocator, &StdTables.bits_ac_luma);
    try tmp_dht.appendSlice(allocator, &StdTables.val_ac_luma);
    try writeSegment(&out, allocator, 0xFFC4, tmp_dht.items);

    try writeSOS(&out, allocator, true);

    var ew = EntropyWriter.init(allocator);
    defer ew.deinit();

    const dc = buildHuffmanEncoder(&StdTables.bits_dc_luma, &StdTables.val_dc_luma);
    const ac = buildHuffmanEncoder(&StdTables.bits_ac_luma, &StdTables.val_ac_luma);

    // Float quant table
    var ql_f: [64]f32 = undefined;
    for (0..64) |i| ql_f[i] = @floatFromInt(ql[i]);

    var prev_dc: i32 = 0;

    const rows: usize = @intCast(height);
    const cols: usize = @intCast(width);
    const block_rows: usize = (rows + 7) / 8;
    const block_cols: usize = (cols + 7) / 8;
    var block: [64]f32 = undefined;
    var dct: [64]f32 = undefined;

    for (0..block_rows) |br| {
        for (0..block_cols) |bc| {
            for (0..8) |y| {
                const iy = @min(rows - 1, br * 8 + y);
                for (0..8) |x| {
                    const ix = @min(cols - 1, bc * 8 + x);
                    const v: u8 = bytes[iy * cols + ix];
                    block[y * 8 + x] = @as(f32, @floatFromInt(v)) - 128.0;
                }
            }
            fdct8x8(&block, &dct);
            var coeff: [64]i32 = undefined;
            for (0..64) |i| coeff[i] = @intFromFloat(@round(dct[i] / ql_f[i]));

            const dc_coeff = coeff[0];
            const diff = dc_coeff - prev_dc;
            prev_dc = dc_coeff;
            const mag = magnitudeCategory(diff);
            try ew.writeBits(@intCast(dc.codes[mag]), @as(u5, @intCast(dc.sizes[mag])));
            if (mag > 0) try ew.writeBits(magnitudeBits(diff, mag), mag);

            var run: u8 = 0;
            for (1..64) |k| {
                const v = coeff[zigzag[k]];
                if (v == 0) {
                    run += 1;
                    if (run == 16) {
                        try ew.writeBits(@intCast(ac.codes[0xF0]), @as(u5, @intCast(ac.sizes[0xF0])));
                        run = 0;
                    }
                } else {
                    while (run >= 16) : (run -= 16) {
                        try ew.writeBits(@intCast(ac.codes[0xF0]), @as(u5, @intCast(ac.sizes[0xF0])));
                    }
                    const amag = magnitudeCategory(v);
                    const sym: u8 = (run << 4) | @as(u8, amag);
                    try ew.writeBits(@intCast(ac.codes[sym]), @as(u5, @intCast(ac.sizes[sym])));
                    try ew.writeBits(magnitudeBits(v, amag), amag);
                    run = 0;
                }
            }
            if (run > 0) try ew.writeBits(@intCast(ac.codes[0x00]), @as(u5, @intCast(ac.sizes[0x00])));
        }
    }

    try ew.flush();
    try out.appendSlice(allocator, ew.data.items);
    try out.append(allocator, 0xFF);
    try out.append(allocator, 0xD9);
    return out.toOwnedSlice(allocator);
}

/// Lightweight scan to detect number of components declared by the JPEG SOF header.
/// Returns 1 for grayscale, 3 for color, or null on error/unsupported.
pub fn detectComponents(data: []const u8) ?u8 {
    if (data.len < 4) return null;
    if (!std.mem.eql(u8, data[0..2], &signature)) return null;

    var pos: usize = 2;
    while (pos + 3 < data.len) {
        // Find marker prefix 0xFF (skip stuffing bytes)
        if (data[pos] != 0xFF) {
            pos += 1;
            continue;
        }
        while (pos < data.len and data[pos] == 0xFF) pos += 1;
        if (pos >= data.len) break;
        const marker = data[pos];
        pos += 1;

        // Markers without a length field
        if (marker == 0xD8 or marker == 0xD9 or (marker >= 0xD0 and marker <= 0xD7) or marker == 0x01) {
            continue;
        }

        if (pos + 1 >= data.len) break;
        const len: usize = (@as(usize, data[pos]) << 8) | data[pos + 1];
        pos += 2;
        if (len < 2 or pos + len - 2 > data.len) break;

        // Baseline or progressive SOF: return component count
        if (marker == 0xC0 or marker == 0xC2) {
            if (len < 8) break; // need precision(1) + height(2) + width(2) + components(1)
            return data[pos + 5];
        }

        pos += len - 2;
    }
    return null;
}

// JPEG markers
pub const Marker = enum(u16) {
    // Start of Frame markers
    SOF0 = 0xFFC0, // Baseline DCT
    SOF1 = 0xFFC1, // Extended sequential DCT
    SOF2 = 0xFFC2, // Progressive DCT
    SOF3 = 0xFFC3, // Lossless (sequential)

    // Huffman table
    DHT = 0xFFC4,

    // Arithmetic coding conditioning
    DAC = 0xFFCC,

    // Restart interval markers
    RST0 = 0xFFD0,
    RST1 = 0xFFD1,
    RST2 = 0xFFD2,
    RST3 = 0xFFD3,
    RST4 = 0xFFD4,
    RST5 = 0xFFD5,
    RST6 = 0xFFD6,
    RST7 = 0xFFD7,

    // Other markers
    SOI = 0xFFD8, // Start of Image
    EOI = 0xFFD9, // End of Image
    SOS = 0xFFDA, // Start of Scan
    DQT = 0xFFDB, // Quantization Table
    DNL = 0xFFDC, // Number of Lines
    DRI = 0xFFDD, // Restart Interval
    DHP = 0xFFDE, // Hierarchical Progression
    EXP = 0xFFDF, // Expand Reference Component

    // Application segments
    APP0 = 0xFFE0, // JFIF
    APP1 = 0xFFE1, // EXIF
    APP2 = 0xFFE2,
    APP3 = 0xFFE3,
    APP4 = 0xFFE4,
    APP5 = 0xFFE5,
    APP6 = 0xFFE6,
    APP7 = 0xFFE7,
    APP8 = 0xFFE8,
    APP9 = 0xFFE9,
    APP10 = 0xFFEA,
    APP11 = 0xFFEB,
    APP12 = 0xFFEC,
    APP13 = 0xFFED,
    APP14 = 0xFFEE,
    APP15 = 0xFFEF,

    // Comment
    COM = 0xFFFE,

    pub fn fromBytes(bytes: [2]u8) ?Marker {
        const value = (@as(u16, bytes[0]) << 8) | bytes[1];
        return std.meta.intToEnum(Marker, value) catch null;
    }
};

// Component info from SOF
const Component = struct {
    id: u8,
    h_sampling: u4,
    v_sampling: u4,
    quant_table_id: u8,
};

// Scan component info from SOS
const ScanComponent = struct {
    component_id: u8,
    dc_table_id: u4,
    ac_table_id: u4,
};

// Scan information
const ScanInfo = struct {
    components: []ScanComponent,
    start_of_spectral_selection: u8,
    end_of_spectral_selection: u8,
    approximation_high: u4,
    approximation_low: u4,
};

// Frame type to distinguish baseline vs progressive
const FrameType = enum {
    baseline, // SOF0
    progressive, // SOF2
};

// JPEG decoder state
pub const JpegDecoder = struct {
    allocator: Allocator,

    // Image properties
    width: u16,
    height: u16,
    num_components: u8,
    components: [4]Component = undefined,
    frame_type: FrameType = .baseline,

    // Huffman tables (0-3 for DC, 0-3 for AC)
    dc_tables: [4]?HuffmanTable = .{ null, null, null, null },
    ac_tables: [4]?HuffmanTable = .{ null, null, null, null },

    // Quantization tables
    quant_tables: [4]?[64]u16 = .{ null, null, null, null },

    // Scan data
    scan_components: []ScanComponent = undefined,
    restart_interval: u16 = 0,

    // Bit reader for entropy-coded data
    bit_reader: BitReader = undefined,

    // Block dimensions
    block_width: u16 = 0,
    block_height: u16 = 0,
    block_width_actual: u16 = 0,
    block_height_actual: u16 = 0,

    // Block storage for all components (persistent across scans)
    block_storage: ?[][4][64]i32 = null,

    // Separate RGB storage to avoid overwriting chroma data
    rgb_storage: ?[][3][64]u8 = null,

    // Progressive decoding state - persistent across scans
    dc_prediction_values: [4]i32 = @splat(0),
    skip_count: u32 = 0, // For progressive AC scans

    // Restart marker tracking
    expected_rst_marker: u3 = 0, // Cycles 0-7 for RST0-RST7

    // Debug
    debug_block_count: u32 = 0,

    pub fn init(allocator: Allocator) JpegDecoder {
        return .{
            .allocator = allocator,
            .width = 0,
            .height = 0,
            .num_components = 0,
            .scan_components = &[_]ScanComponent{},
            .frame_type = .baseline,
        };
    }

    pub fn deinit(self: *JpegDecoder) void {
        for (&self.dc_tables) |*table| {
            if (table.*) |*t| t.deinit();
        }
        for (&self.ac_tables) |*table| {
            if (table.*) |*t| t.deinit();
        }
        if (self.scan_components.len > 0) {
            self.allocator.free(self.scan_components);
        }
        if (self.block_storage) |storage| {
            self.allocator.free(storage);
        }
        if (self.rgb_storage) |storage| {
            self.allocator.free(storage);
        }
    }

    // Decode a Huffman symbol using the fast lookup table
    pub fn readCode(self: *JpegDecoder, table: *const HuffmanTable) !u8 {
        const fast_bits = 9;
        const fast_index = self.bit_reader.peekBits(fast_bits) catch 0;

        if (self.bit_reader.bit_count >= fast_bits) {
            const value = table.fast_table[fast_index];
            if (value != 255) {
                const length = table.fast_size[fast_index];
                self.bit_reader.consumeBits(length);
                return value;
            }
        }

        var code: u32 = 0;
        var length: u5 = if (self.bit_reader.bit_count < fast_bits) 1 else fast_bits + 1;
        while (length <= 16) : (length += 1) {
            code = self.bit_reader.peekBits(length) catch return error.InvalidHuffmanCode;
            if (table.code_map.get(.{ .length_minus_one = @intCast(length - 1), .code = @intCast(code) })) |value| {
                self.bit_reader.consumeBits(length);
                return value;
            }
        }

        // Debug info for invalid Huffman codes (commented out for now)
        // std.debug.print("DEBUG: Invalid Huffman code: 0x{X} (length tried up to 16)\n", .{code});

        return error.InvalidHuffmanCode;
    }

    // Legacy alias for compatibility
    pub fn decodeHuffmanSymbol(self: *JpegDecoder, table: *const HuffmanTable) !u8 {
        return self.readCode(table);
    }

    // Decode magnitude-coded coefficient (T.81 section F1.2.1)
    pub fn readMagnitudeCoded(self: *JpegDecoder, magnitude: u5) !i32 {
        if (magnitude == 0) return 0;

        var coeff: i32 = @intCast(try self.bit_reader.peekBits(magnitude));
        self.bit_reader.consumeBits(magnitude);

        // Convert from unsigned to signed
        if (coeff < @as(i32, 1) << @intCast(magnitude - 1)) {
            coeff -= (@as(i32, 1) << @intCast(magnitude)) - 1;
        }

        return coeff;
    }

    // Decode AC coefficients (simple baseline implementation)
    pub fn decodeAC(self: *JpegDecoder, table: *const HuffmanTable, block: *[64]i32) !void {
        var k: usize = 1; // Start after DC coefficient

        while (k < 64) {
            const symbol = try self.readCode(table);

            if (symbol == 0) {
                // End of block - zero fill remaining coefficients
                while (k < 64) {
                    block[zigzag[k]] = 0;
                    k += 1;
                }
                return;
            }

            const run = symbol >> 4;
            const size = symbol & 0x0F;

            if (size == 0) {
                if (run == 15) {
                    // ZRL: skip 16 zeros
                    for (0..16) |_| {
                        if (k >= 64) break;
                        block[zigzag[k]] = 0;
                        k += 1;
                    }
                } else {
                    break; // Invalid symbol, exit gracefully
                }
            } else {
                // Skip 'run' zeros
                for (0..run) |_| {
                    if (k >= 64) break;
                    block[zigzag[k]] = 0;
                    k += 1;
                }

                if (k >= 64) break;

                // Decode AC coefficient using magnitude decoder
                const value = try self.readMagnitudeCoded(@intCast(size));
                block[zigzag[k]] = value;
                k += 1;
            }
        }
    }

    // Parse Start of Frame (SOF0/SOF2) marker
    pub fn parseSOF(self: *JpegDecoder, data: []const u8, frame_type: FrameType) !void {
        self.frame_type = frame_type;
        if (data.len < 8) return error.InvalidSOF;

        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidSOF;

        const precision = data[2];
        // Provide specific error messages for different precision values
        switch (precision) {
            8 => {}, // Supported
            12 => return error.Unsupported12BitPrecision,
            16 => return error.Unsupported16BitPrecision,
            else => return error.UnsupportedPrecision,
        }

        self.height = (@as(u16, data[3]) << 8) | data[4];
        self.width = (@as(u16, data[5]) << 8) | data[6];
        self.num_components = data[7];

        // Distinguish between invalid and unsupported component counts
        switch (self.num_components) {
            1, 3 => {}, // Supported: grayscale and YCbCr
            4 => return error.UnsupportedComponentCount, // CMYK - valid but unsupported
            0 => return error.InvalidComponentCount, // Invalid: no components
            else => return error.InvalidComponentCount, // Invalid: too many components
        }

        // Parse component information
        var pos: usize = 8;
        var max_h_sampling: u4 = 0;
        var max_v_sampling: u4 = 0;

        for (0..self.num_components) |i| {
            if (pos + 3 > data.len) return error.InvalidSOF;

            self.components[i] = .{
                .id = data[pos],
                .h_sampling = @intCast(data[pos + 1] >> 4),
                .v_sampling = @intCast(data[pos + 1] & 0x0F),
                .quant_table_id = data[pos + 2],
            };

            max_h_sampling = @max(max_h_sampling, self.components[i].h_sampling);
            max_v_sampling = @max(max_v_sampling, self.components[i].v_sampling);

            pos += 3;
        }

        // Validate sampling factors
        if (max_h_sampling > 4 or max_v_sampling > 4) {
            return error.UnsupportedSamplingFactor;
        }

        // Validate specific chroma subsampling combinations
        if (self.num_components == 3) {
            // For color images, check if we support the chroma subsampling
            const y_h = self.components[0].h_sampling;
            const y_v = self.components[0].v_sampling;
            const cb_h = self.components[1].h_sampling;
            const cb_v = self.components[1].v_sampling;
            const cr_h = self.components[2].h_sampling;
            const cr_v = self.components[2].v_sampling;

            // Cb and Cr must have same sampling factors
            if (cb_h != cr_h or cb_v != cr_v) {
                return error.InvalidComponentCount; // Inconsistent chroma sampling
            }

            // Check for supported subsampling ratios
            const is_444 = (y_h == 1 and y_v == 1 and cb_h == 1 and cb_v == 1);
            const is_420 = (y_h == 2 and y_v == 2 and cb_h == 1 and cb_v == 1);
            const is_422 = (y_h == 2 and y_v == 1 and cb_h == 1 and cb_v == 1);
            const is_411 = (y_h == 4 and y_v == 1 and cb_h == 1 and cb_v == 1);

            if (!is_444 and !is_420 and !is_422 and !is_411) {
                // Note: While 4:2:2 and 4:1:1 pass validation, they use fallback processing
                // Only 4:4:4 and 4:2:0 have optimized implementations
                return error.UnsupportedSamplingFactor;
            }
        }

        // Calculate block dimensions
        const mcu_width = 8 * @as(u32, max_h_sampling);
        const mcu_height = 8 * @as(u32, max_v_sampling);
        const width_actual = ((@as(u32, self.width) + mcu_width - 1) / mcu_width) * mcu_width;
        const height_actual = ((@as(u32, self.height) + mcu_height - 1) / mcu_height) * mcu_height;

        self.block_width = (self.width + 7) / 8;
        self.block_height = (self.height + 7) / 8;
        self.block_width_actual = @intCast((width_actual + 7) / 8);
        self.block_height_actual = @intCast((height_actual + 7) / 8);

        // Allocate block storage
        const total_blocks = @as(usize, width_actual) * height_actual / 64;
        self.block_storage = try self.allocator.alloc([4][64]i32, total_blocks);

        // Allocate separate RGB storage
        self.rgb_storage = try self.allocator.alloc([3][64]u8, total_blocks);

        // Initialize block storage to zero
        for (self.block_storage.?) |*block_set| {
            for (block_set) |*block| {
                @memset(block, 0);
            }
        }

        // Initialize RGB storage to zero
        for (self.rgb_storage.?) |*rgb_block| {
            for (rgb_block) |*channel| {
                @memset(channel, 0);
            }
        }
    }

    // Parse Define Huffman Table (DHT) marker
    pub fn parseDHT(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 2) return error.InvalidDHT;

        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidDHT;

        var pos: usize = 2;

        while (pos < length) {
            if (pos + 17 > length) return error.InvalidDHT;

            const table_info = data[pos];
            const table_class = (table_info >> 4) & 1; // 0 = DC, 1 = AC
            const table_id = table_info & 0b11; // 0-3 as per JPEG standard

            if (table_id > 3) return error.InvalidHuffmanTable;

            pos += 1;

            // Read 16 bytes of bit lengths
            var bits: [16]u8 = undefined;
            @memcpy(&bits, data[pos .. pos + 16]);
            pos += 16;

            // Count total number of codes
            var total_codes: u16 = 0;
            for (bits) |count| {
                total_codes += count;
            }

            if (pos + total_codes > length) return error.InvalidDHT;

            // Allocate and read huffman values
            const huffval = try self.allocator.alloc(u8, total_codes);
            @memcpy(huffval, data[pos .. pos + total_codes]);
            pos += total_codes;

            // Build Huffman table
            var code_map = std.AutoArrayHashMap(HuffmanCode, u8).init(self.allocator);
            errdefer {
                code_map.deinit();
                self.allocator.free(huffval);
            }

            var fast_table: [512]u8 = @splat(255);
            var fast_size: [512]u5 = @splat(0);

            // Build codes according to JPEG standard
            var code: u16 = 0;
            var huffval_index: usize = 0;
            for (bits, 0..) |count, i| {
                var j: usize = 0;
                while (j < count) : (j += 1) {
                    // Check for invalid code (all 1s)
                    if (code == (@as(u17, @intCast(1)) << (@as(u5, @intCast(i)) + 1)) - 1) {
                        return error.InvalidHuffmanTable;
                    }

                    const byte = huffval[huffval_index];
                    huffval_index += 1;
                    try code_map.put(.{ .length_minus_one = @as(u4, @intCast(i)), .code = code }, byte);

                    // Build fast lookup table for codes <= 9 bits
                    if (i + 1 <= 9) {
                        const first_index = code << 9 - @as(u4, @intCast(i + 1));
                        const num_indexes = @as(usize, 1) << @as(u4, @intCast(9 - (i + 1)));
                        for (0..num_indexes) |index| {
                            std.debug.assert(fast_table[first_index + index] == 255);
                            fast_table[first_index + index] = byte;
                            fast_size[first_index + index] = @as(u5, @intCast(i + 1));
                        }
                    }

                    code += 1;
                }
                code <<= 1;
            }

            const table = HuffmanTable{
                .allocator = self.allocator,
                .code_counts = bits,
                .code_map = code_map,
                .fast_table = fast_table,
                .fast_size = fast_size,
            };

            // Free huffval array - no longer needed after building code_map
            self.allocator.free(huffval);

            // Store table
            if (table_class == 0) {
                if (self.dc_tables[table_id]) |*old_table| {
                    old_table.deinit();
                }
                self.dc_tables[table_id] = table;
            } else {
                if (self.ac_tables[table_id]) |*old_table| {
                    old_table.deinit();
                }
                self.ac_tables[table_id] = table;
            }
        }
    }

    // Parse Define Quantization Table (DQT) marker
    pub fn parseDQT(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 2) return error.InvalidDQT;

        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidDQT;

        var pos: usize = 2;

        while (pos < length) {
            if (pos + 1 > length) return error.InvalidDQT;

            const table_info = data[pos];
            const precision = (table_info >> 4) & 0x0F; // 0 = 8-bit, 1 = 16-bit
            const table_id = table_info & 0b11; // Only use bottom 2 bits

            if (table_id > 3) return error.InvalidQuantTable;

            pos += 1;

            const element_size: usize = if (precision == 0) 1 else 2;
            if (pos + 64 * element_size > length) return error.InvalidDQT;

            // Read quantization table and convert from zigzag to natural order
            var table: [64]u16 = undefined;

            if (precision == 0) {
                // 8-bit values - stored in zigzag order in file, convert to natural order
                for (0..64) |i| {
                    table[zigzag[i]] = data[pos + i];
                }
                pos += 64;
            } else {
                // 16-bit values - stored in zigzag order in file, convert to natural order
                for (0..64) |i| {
                    table[zigzag[i]] = (@as(u16, data[pos + i * 2]) << 8) | data[pos + i * 2 + 1];
                }
                pos += 128;
            }

            self.quant_tables[table_id] = table;
        }
    }

    // Parse Start of Scan (SOS) marker
    pub fn parseSOS(self: *JpegDecoder, data: []const u8) !ScanInfo {
        if (data.len < 6) return error.InvalidSOS;

        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidSOS;

        const num_components = data[2];
        // For progressive JPEG, individual scans can have fewer components
        if (self.frame_type == .baseline and num_components != self.num_components) return error.InvalidSOS;
        if (self.frame_type == .progressive and (num_components == 0 or num_components > self.num_components)) return error.InvalidSOS;

        const scan_components = try self.allocator.alloc(ScanComponent, num_components);

        var pos: usize = 3;
        for (0..num_components) |i| {
            if (pos + 2 > length) return error.InvalidSOS;

            scan_components[i] = .{
                .component_id = data[pos],
                .dc_table_id = @intCast(data[pos + 1] >> 4),
                .ac_table_id = @intCast(data[pos + 1] & 0x0F),
            };

            pos += 2;
        }

        // Read spectral selection and successive approximation
        if (pos + 3 > length) return error.InvalidSOS;

        const start_of_spectral = data[pos];
        const end_of_spectral = data[pos + 1];
        const approximation = data[pos + 2];

        // Validate spectral selection parameters
        if (self.frame_type == .baseline) {
            // For baseline JPEG, these should be 0, 63, 0
            if (start_of_spectral != 0 or end_of_spectral != 63 or approximation != 0) {
                return error.InvalidSOS;
            }
        } else if (self.frame_type == .progressive) {
            // For progressive JPEG, validate spectral selection
            if (start_of_spectral > 63 or end_of_spectral > 63) return error.InvalidSOS;
            if (end_of_spectral < start_of_spectral) return error.InvalidSOS;

            // DC-only scans have start=0, end=0; AC-only scans have start>0
            const any_zero = start_of_spectral == 0 or end_of_spectral == 0;
            const both_zero = start_of_spectral == 0 and end_of_spectral == 0;
            if (any_zero and !both_zero) return error.InvalidSOS;
        }

        return ScanInfo{
            .components = scan_components,
            .start_of_spectral_selection = start_of_spectral,
            .end_of_spectral_selection = end_of_spectral,
            .approximation_high = @intCast((approximation >> 4) & 0x0F),
            .approximation_low = @intCast(approximation & 0x0F),
        };
    }

    // Parse Define Restart Interval (DRI) marker
    pub fn parseDRI(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 4) return error.InvalidDRI;

        const length = (@as(u16, data[0]) << 8) | data[1];
        if (length != 4) return error.InvalidDRI;

        self.restart_interval = (@as(u16, data[2]) << 8) | data[3];
    }
};

// Huffman table for decoding
const HuffmanTable = struct {
    allocator: Allocator,
    // Number of codes for each bit length (1-16)
    code_counts: [16]u8,
    // Hash map for full lookup
    code_map: std.AutoArrayHashMap(HuffmanCode, u8),
    // Fast lookup table for short codes
    fast_table: [512]u8, // 2^9 entries
    fast_size: [512]u5,

    pub fn deinit(self: *HuffmanTable) void {
        self.code_map.deinit();
    }
};

const HuffmanCode = struct { length_minus_one: u4, code: u16 };

// Bit reader for entropy-coded segments
pub const BitReader = struct {
    data: []const u8,
    byte_pos: usize = 0,
    bit_buffer: u32 = 0,
    bit_count: u5 = 0,

    pub fn init(data: []const u8) BitReader {
        return .{ .data = data };
    }

    pub fn peekBits(self: *BitReader, num_bits: u5) !u32 {
        if (num_bits > 24) return error.InvalidData;
        try self.fillBits(num_bits);
        return (self.bit_buffer >> 1) >> @intCast(31 - num_bits);
    }

    pub fn fillBits(self: *BitReader, num_bits: u5) !void {
        while (self.bit_count < num_bits) {
            if (self.byte_pos >= self.data.len) {
                return error.UnexpectedEndOfData;
            }

            var byte_curr: u32 = self.data[self.byte_pos];
            self.byte_pos += 1;

            while (byte_curr == 0xFF) {
                if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
                const byte_next: u8 = self.data[self.byte_pos];
                self.byte_pos += 1;

                if (byte_next == 0x00) {
                    break;
                } else if (byte_next == 0xFF) {
                    continue;
                } else if (byte_next >= 0xD0 and byte_next <= 0xD7) {
                    // Restart marker found - just skip it, validation happens at scan level
                    if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
                    byte_curr = self.data[self.byte_pos];
                    self.byte_pos += 1;
                } else {
                    std.debug.print("DEBUG: Found marker 0xFF{X:0>2} - ending scan\n", .{byte_next});
                    self.byte_pos -= 2;
                    return error.UnexpectedEndOfData;
                }
            }

            self.bit_buffer |= byte_curr << @intCast(24 - self.bit_count);
            self.bit_count += 8;
        }
    }

    pub fn consumeBits(self: *BitReader, num_bits: u5) void {
        std.debug.assert(num_bits <= self.bit_count and num_bits <= 16);
        self.bit_buffer <<= num_bits;
        self.bit_count -= num_bits;
    }

    pub fn getBits(self: *BitReader, n: u5) !u32 {
        const bits = try self.peekBits(n);
        self.consumeBits(n);
        return @intCast(bits);
    }

    pub fn flushBits(self: *BitReader) void {
        if (self.bit_count > 8 and self.bit_count % 8 != 0) {
            const bits_to_flush: u5 = self.bit_count % 8;
            self.bit_buffer <<= bits_to_flush;
            self.bit_count = self.bit_count - bits_to_flush;
        } else if (self.bit_count % 8 == 0) {
            return;
        } else if (self.bit_count < 8) {
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
    }
};

// Perform a scan (baseline or progressive)
pub fn performScan(decoder: *JpegDecoder, scan_info: ScanInfo) !void {
    if (decoder.block_storage == null) return error.BlockStorageNotAllocated;

    if (decoder.frame_type == .baseline) {
        // Baseline JPEG: single scan with all data
        try performBaselineScan(decoder, scan_info);
    } else {
        // Progressive JPEG: accumulate data across multiple scans
        try performProgressiveScan(decoder, scan_info);
    }
}

// Upsample and convert a single YCbCr block to RGB
pub fn yCbCrToRgbBlock(_: *JpegDecoder, y_block: *[64]i32, cb_block: *const [64]i32, cr_block: *const [64]i32, rgb_block: *[3][64]u8) void {
    // YCbCr to RGB conversion coefficients (ITU-R BT.601 standard)
    const co_1: @Vector(8, f32) = @splat(1.402); // Cr to R
    const co_2: @Vector(8, f32) = @splat(1.772); // Cb to B
    const co_3: @Vector(8, f32) = @splat(-0.344136); // Cb to G
    const co_4: @Vector(8, f32) = @splat(-0.714136); // Cr to G
    const vec_0: @Vector(8, f32) = @splat(0.0);
    const vec_255: @Vector(8, f32) = @splat(255.0);

    for (0..8) |y| {
        const y_vec_i32: @Vector(8, i32) = y_block[y * 8 ..][0..8].*;
        const y_vec: @Vector(8, f32) = @floatFromInt(y_vec_i32);

        const cb_vec_i32: @Vector(8, i32) = cb_block[y * 8 ..][0..8].*;
        const cb_vec: @Vector(8, f32) = @floatFromInt(cb_vec_i32);

        const cr_vec_i32: @Vector(8, i32) = cr_block[y * 8 ..][0..8].*;
        const cr_vec: @Vector(8, f32) = @floatFromInt(cr_vec_i32);

        var r_vec = y_vec + cr_vec * co_1;
        var g_vec = y_vec + cb_vec * co_3 + cr_vec * co_4;
        var b_vec = y_vec + cb_vec * co_2;

        r_vec = std.math.clamp(r_vec, vec_0, vec_255);
        g_vec = std.math.clamp(g_vec, vec_0, vec_255);
        b_vec = std.math.clamp(b_vec, vec_0, vec_255);

        const r_u8: @Vector(8, u8) = @intFromFloat(r_vec);
        const g_u8: @Vector(8, u8) = @intFromFloat(g_vec);
        const b_u8: @Vector(8, u8) = @intFromFloat(b_vec);

        var r_array: [8]u8 = undefined;
        var g_array: [8]u8 = undefined;
        var b_array: [8]u8 = undefined;

        for (0..8) |i| {
            r_array[i] = r_u8[i];
            g_array[i] = g_u8[i];
            b_array[i] = b_u8[i];
        }

        @memcpy(rgb_block[0][y * 8 ..][0..8], &r_array);
        @memcpy(rgb_block[1][y * 8 ..][0..8], &g_array);
        @memcpy(rgb_block[2][y * 8 ..][0..8], &b_array);
    }
}

// Perform baseline scan
fn performBaselineScan(decoder: *JpegDecoder, scan_info: ScanInfo) !void {
    // Calculate maximum sampling factors
    var max_h_factor: u4 = 1;
    var max_v_factor: u4 = 1;
    for (decoder.components[0..decoder.num_components]) |comp| {
        max_h_factor = @max(max_h_factor, comp.h_sampling);
        max_v_factor = @max(max_v_factor, comp.v_sampling);
    }

    // DC prediction values for each component
    var prediction_values: [4]i32 = @splat(0);

    // Scan structure
    const noninterleaved = scan_info.components.len == 1 and scan_info.components[0].component_id == 1;
    // For non-interleaved scans (Y only), step by 1, otherwise use sampling factors
    const y_step = if (noninterleaved) 1 else max_v_factor;
    const x_step = if (noninterleaved) 1 else max_h_factor;

    var mcu_count: u32 = 0;
    var mcus_since_restart: u32 = 0;

    var y: usize = 0;
    while (y < decoder.block_height) : (y += y_step) {
        var x: usize = 0;
        while (x < decoder.block_width) : (x += x_step) {
            // Handle restart intervals for baseline scans
            if (decoder.restart_interval != 0 and mcus_since_restart == decoder.restart_interval) {
                // Reset DC predictions
                prediction_values = @splat(0);
                mcus_since_restart = 0;
                // Reset expected RST marker
                decoder.expected_rst_marker = 0;
                // Flush bits to byte boundary
                decoder.bit_reader.flushBits();
            }
            // Decode each component at this position
            for (scan_info.components) |scan_comp| {
                // Find the component index for this scan component
                var component_index: usize = 0;
                var v_max: usize = undefined;
                var h_max: usize = undefined;

                for (decoder.components[0..decoder.num_components], 0..) |frame_component, i| {
                    if (frame_component.id == scan_comp.component_id) {
                        component_index = i;
                        v_max = if (noninterleaved) 1 else frame_component.v_sampling;
                        h_max = if (noninterleaved) 1 else frame_component.h_sampling;
                        break;
                    }
                }

                // Decode all blocks for this component in this MCU
                for (0..v_max) |v| {
                    for (0..h_max) |h| {
                        // Standard coordinate calculation
                        const actual_x = x + h;
                        const actual_y = y + v;

                        if (actual_y >= decoder.block_height or actual_x >= decoder.block_width) continue;

                        const block_id = actual_y * decoder.block_width_actual + actual_x;
                        const block = &decoder.block_storage.?[block_id][component_index];

                        // Fill bit buffer before decoding
                        try decoder.bit_reader.fillBits(24);

                        // Decode block directly into storage
                        decodeBlockBaseline(decoder, scan_comp, block, &prediction_values[component_index]) catch |err| {
                            if (err == error.UnexpectedEndOfData) return;
                            return err;
                        };
                    }
                }
            }

            mcu_count += 1;
            mcus_since_restart += 1;
        }
    }
}

// Perform progressive scan
fn performProgressiveScan(decoder: *JpegDecoder, scan_info: ScanInfo) !void {
    var skips: u32 = 0;

    // Definition of noninterleaved
    const noninterleaved = scan_info.components.len == 1 and scan_info.components[0].component_id == 1;

    // Calculate sampling factors
    var max_h_factor: u4 = 1;
    var max_v_factor: u4 = 1;
    for (decoder.components[0..decoder.num_components]) |comp| {
        max_h_factor = @max(max_h_factor, comp.h_sampling);
        max_v_factor = @max(max_v_factor, comp.v_sampling);
    }

    const y_step = if (noninterleaved) 1 else max_v_factor;
    const x_step = if (noninterleaved) 1 else max_h_factor;

    // Scan loop structure
    var y: usize = 0;
    while (y < decoder.block_height) : (y += y_step) {
        var x: usize = 0;
        while (x < decoder.block_width) : (x += x_step) {
            const mcu_id = y * decoder.block_width_actual + x;

            // Handle restart intervals
            if (decoder.restart_interval != 0 and mcu_id % (decoder.restart_interval * y_step * x_step) == 0) {
                decoder.bit_reader.flushBits();
                decoder.dc_prediction_values = @splat(0);
                skips = 0;
            }

            for (0..scan_info.components.len) |index| {
                const scan_comp = scan_info.components[index];

                var component_index: usize = undefined;
                var v_max: usize = undefined;
                var h_max: usize = undefined;

                // Find the component
                for (decoder.components[0..decoder.num_components], 0..) |frame_component, i| {
                    if (frame_component.id == scan_comp.component_id) {
                        component_index = i;
                        v_max = if (noninterleaved) 1 else frame_component.v_sampling;
                        h_max = if (noninterleaved) 1 else frame_component.h_sampling;
                        break;
                    }
                }

                for (0..v_max) |v| {
                    for (0..h_max) |h| {
                        const block_id = (y + v) * decoder.block_width_actual + (x + h);
                        const block = &decoder.block_storage.?[block_id][component_index];

                        // Fill bits
                        decoder.bit_reader.fillBits(24) catch {};

                        try decodeBlockProgressive(decoder, scan_info, scan_comp, block, &decoder.dc_prediction_values[component_index], &skips);
                    }
                }
            }
        }
    }

    // Save skip count for next progressive AC scan
    if (scan_info.start_of_spectral_selection != 0) {
        decoder.skip_count = skips;
    }
}

// Decode a single block in progressive mode
fn decodeBlockProgressive(decoder: *JpegDecoder, scan_info: ScanInfo, scan_comp: ScanComponent, block: *[64]i32, dc_prediction: *i32, skips: *u32) !void {
    if (scan_info.start_of_spectral_selection == 0) {
        const dc_table = decoder.dc_tables[scan_comp.dc_table_id] orelse return error.MissingHuffmanTable;
        if (scan_info.approximation_high == 0) {
            const maybe_magnitude = try decoder.readCode(&dc_table);
            if (maybe_magnitude > 11) return error.InvalidDCCoefficient;
            const diff = try decoder.readMagnitudeCoded(@intCast(maybe_magnitude));
            const dc_coefficient = diff + dc_prediction.*;
            dc_prediction.* = dc_coefficient;
            block[0] = dc_coefficient << @intCast(scan_info.approximation_low);
        } else if (scan_info.approximation_high != 0) {
            const bit: u32 = try decoder.bit_reader.getBits(1);
            block[0] += @as(i32, @intCast(bit)) << @intCast(scan_info.approximation_low);
        }
    } else if (scan_info.start_of_spectral_selection != 0) {
        const ac_table = decoder.ac_tables[scan_comp.ac_table_id] orelse return error.MissingHuffmanTable;
        if (scan_info.approximation_high == 0) {
            var ac: usize = scan_info.start_of_spectral_selection;
            // Check skips == 0 first
            if (skips.* == 0) {
                while (ac <= scan_info.end_of_spectral_selection) {
                    var coeff: i32 = 0;
                    const zero_run_length_and_magnitude = try decoder.readCode(&ac_table);
                    const zero_run_length = zero_run_length_and_magnitude >> 4;
                    const maybe_magnitude = zero_run_length_and_magnitude & 0x0F;

                    if (maybe_magnitude == 0) {
                        if (zero_run_length < 15) {
                            const extra_skips: u32 = try decoder.bit_reader.getBits(@intCast(zero_run_length));
                            skips.* = (@as(u32, 1) << @intCast(zero_run_length));
                            skips.* += extra_skips;
                            break; // process skips
                        } // no special case for zrl == 15
                    } else if (maybe_magnitude != 0) {
                        if (maybe_magnitude > 10) return error.InvalidACCoefficient;
                        coeff = try decoder.readMagnitudeCoded(@intCast(maybe_magnitude));
                    }

                    for (0..zero_run_length) |_| {
                        block[zigzag[ac]] = 0;
                        ac += 1;
                    }
                    block[zigzag[ac]] = coeff << @intCast(scan_info.approximation_low);
                    ac += 1;
                }
            }

            if (skips.* > 0) {
                skips.* -= 1;
                while (ac <= scan_info.end_of_spectral_selection) {
                    block[zigzag[ac]] = 0;
                    ac += 1;
                }
            }
        } else if (scan_info.approximation_high != 0) {
            const bit: i32 = @as(i32, 1) << @intCast(scan_info.approximation_low);
            var ac: usize = scan_info.start_of_spectral_selection;
            if (skips.* == 0) {
                while (ac <= scan_info.end_of_spectral_selection) {
                    var coeff: i32 = 0;
                    const zero_run_length_and_magnitude = try decoder.readCode(&ac_table);
                    var zero_run_length = zero_run_length_and_magnitude >> 4;
                    const maybe_magnitude = zero_run_length_and_magnitude & 0x0F;

                    if (maybe_magnitude == 0) {
                        if (zero_run_length < 15) {
                            skips.* = (@as(u32, 1) << @intCast(zero_run_length));
                            const extra_skips: u32 = try decoder.bit_reader.getBits(@intCast(zero_run_length));
                            skips.* += extra_skips;
                            break; // start processing skips
                        } // no special treatment for zero_run_length == 15
                    } else if (maybe_magnitude != 0) {
                        const sign_bit: u32 = try decoder.bit_reader.getBits(1);
                        coeff = if (sign_bit == 1) bit else -bit;
                    }

                    // Process zero run and place coefficient
                    while (ac <= scan_info.end_of_spectral_selection) {
                        if (block[zigzag[ac]] == 0) {
                            if (zero_run_length > 0) {
                                zero_run_length -= 1;
                                ac += 1;
                            } else {
                                block[zigzag[ac]] = coeff;
                                ac += 1;
                                break;
                            }
                        } else {
                            const sign_bit: u32 = try decoder.bit_reader.getBits(1);
                            if (sign_bit != 0) {
                                block[zigzag[ac]] += if (block[zigzag[ac]] > 0) bit else -bit;
                            }
                            ac += 1;
                        }
                    }
                }
            }

            // Process skips
            if (skips.* > 0) {
                while (ac <= scan_info.end_of_spectral_selection) : (ac += 1) {
                    if (block[zigzag[ac]] != 0) {
                        const sign_bit: u32 = try decoder.bit_reader.getBits(1);
                        if (sign_bit != 0) {
                            block[zigzag[ac]] += if (block[zigzag[ac]] > 0) bit else -bit;
                        }
                    }
                }
                skips.* -= 1;
            }
        }
    }
}

// Decode a single block in baseline mode
fn decodeBlockBaseline(decoder: *JpegDecoder, scan_comp: ScanComponent, block: *[64]i32, dc_prediction: *i32) !void {
    // For baseline, clear the block
    @memset(block, 0);

    // Decode DC coefficient
    const dc_table = decoder.dc_tables[scan_comp.dc_table_id] orelse return error.MissingHuffmanTable;
    const dc_symbol = try decoder.readCode(&dc_table);

    if (dc_symbol > 11) return error.InvalidDCCoefficient;

    const dc_diff = try decoder.readMagnitudeCoded(@intCast(dc_symbol));

    dc_prediction.* += dc_diff;
    block[0] = dc_prediction.*;

    // Decode AC coefficients using the existing function
    const ac_table = decoder.ac_tables[scan_comp.ac_table_id] orelse return error.MissingHuffmanTable;
    try decoder.decodeAC(&ac_table, block);
}

// Parse JPEG file and decode image
// Helper function to find the end of entropy-coded scan data
fn findScanEnd(data: []const u8, start_pos: usize) usize {
    var scan_end = start_pos;
    while (scan_end < data.len - 1) {
        if (data[scan_end] == 0xFF) {
            if (scan_end + 1 < data.len) {
                const next_byte = data[scan_end + 1];
                // 0xFF00 is byte stuffing, not a marker
                if (next_byte == 0x00) {
                    scan_end += 2;
                    continue;
                }
                // Check if it's a restart marker (can appear in entropy data)
                if (next_byte >= 0xD0 and next_byte <= 0xD7) {
                    scan_end += 2;
                    continue;
                }
                // Any other marker ends the entropy-coded segment
                break;
            }
        }
        scan_end += 1;
    }
    return scan_end;
}

// Helper function to read marker length from data
fn readMarkerLength(data: []const u8, pos: usize) !u16 {
    if (pos + 2 > data.len) return error.UnexpectedEndOfData;
    return (@as(u16, data[pos]) << 8) | data[pos + 1];
}

// Helper function to process a Start of Scan marker
fn processScanMarker(decoder: *JpegDecoder, data: []const u8, pos: usize) !usize {
    const scan_info = try decoder.parseSOS(data[pos + 2 ..]);
    const header_len = try readMarkerLength(data, pos + 2);
    const scan_start = pos + 2 + header_len;

    const scan_end = findScanEnd(data, scan_start);
    decoder.bit_reader = BitReader.init(data[scan_start..scan_end]);

    // For baseline JPEG, don't perform scan here - loadJpeg will call performBlockScan
    if (decoder.frame_type == .baseline) {
        // Track allocated components for baseline
        decoder.scan_components = scan_info.components;
        return scan_end; // Signal that baseline processing is complete
    }

    // For progressive JPEG, perform the scan
    performScan(decoder, scan_info) catch |err| {
        // Free scan components before propagating error
        decoder.allocator.free(scan_info.components);
        return err;
    };

    // Free scan components for progressive (don't store in decoder)
    decoder.allocator.free(scan_info.components);
    return scan_end;
}

pub fn decode(allocator: Allocator, data: []const u8) !JpegDecoder {
    var decoder = JpegDecoder.init(allocator);
    errdefer decoder.deinit();

    // Check for JPEG SOI marker
    if (data.len < 2 or !std.mem.eql(u8, data[0..2], &signature)) {
        return error.InvalidJpegFile;
    }

    var pos: usize = 2;

    // Parse JPEG markers
    while (pos < data.len - 1) {
        if (data[pos] != 0xFF) {
            return error.InvalidMarker;
        }

        const marker_bytes = [2]u8{ data[pos], data[pos + 1] };
        const marker = Marker.fromBytes(marker_bytes) orelse {
            // Skip unknown markers
            pos += 2;
            if (pos + 2 > data.len) break;
            const length = try readMarkerLength(data, pos);
            if (length < 2) return error.InvalidMarker;
            pos += length;
            continue;
        };

        switch (marker) {
            .SOI => {
                pos += 2;
                continue;
            },
            .EOI => break,

            .SOF0, .SOF2 => {
                const frame_type: FrameType = if (marker == .SOF0) .baseline else .progressive;
                try decoder.parseSOF(data[pos + 2 ..], frame_type);
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },

            // Specific unsupported SOF markers
            .SOF1 => return error.UnsupportedExtendedSequential,
            .SOF3 => return error.UnsupportedLosslessJpeg,

            .DHT => {
                try decoder.parseDHT(data[pos + 2 ..]);
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },

            .DQT => {
                try decoder.parseDQT(data[pos + 2 ..]);
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },

            .SOS => {
                const scan_end = try processScanMarker(&decoder, data, pos);
                // For baseline JPEG, return immediately after first scan
                if (decoder.frame_type == .baseline) {
                    return decoder;
                }
                // For progressive JPEG, continue parsing more scans
                pos = scan_end;
            },

            .DRI => {
                try decoder.parseDRI(data[pos + 2 ..]);
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },

            // Detect arithmetic coding
            .DAC => return error.UnsupportedArithmeticCoding,

            // Detect hierarchical JPEG
            .DHP => return error.UnsupportedHierarchicalJpeg,

            // Detect differential JPEG
            .DNL => return error.UnsupportedJpegVariant,

            .APP0, .APP1, .APP2, .APP3, .APP4, .APP5, .APP6, .APP7, .APP8, .APP9, .APP10, .APP11, .APP12, .APP13, .APP14, .APP15, .COM => {
                // Skip application and comment markers
                if (pos + 4 > data.len) break;
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },

            else => {
                // Check for other unsupported SOF markers
                const marker_value = @intFromEnum(marker);
                if (marker_value >= 0xFFC5 and marker_value <= 0xFFCF) {
                    // SOF5-SOF15 are unsupported variants
                    return error.UnsupportedJpegVariant;
                }
                // Skip other unknown markers with length
                if (pos + 4 > data.len) break;
                const length = try readMarkerLength(data, pos + 2);
                pos += 2 + length;
            },
        }
    }

    // For progressive JPEG that finished all scans
    if (decoder.frame_type == .progressive) {
        return decoder;
    }

    return error.NoScanData;
}

// Error types
pub const JpegError = error{
    InvalidJpegFile,
    InvalidMarker,
    InvalidSOF,
    InvalidDHT,
    InvalidDQT,
    InvalidSOS,
    InvalidDRI,
    UnsupportedJpegFormat,
    // Specific unsupported format errors
    UnsupportedExtendedSequential, // SOF1
    UnsupportedLosslessJpeg, // SOF3
    UnsupportedJpegVariant, // SOF5-SOF15
    UnsupportedArithmeticCoding, // DAC marker
    Unsupported12BitPrecision, // 12-bit samples
    Unsupported16BitPrecision, // 16-bit samples
    UnsupportedPrecision, // Other precision values
    UnsupportedComponentCount, // Valid but unsupported component counts
    UnsupportedSamplingFactor, // Sampling factors > 4
    UnsupportedHierarchicalJpeg, // DHP marker
    // Invalid format errors
    InvalidComponentCount,
    InvalidHuffmanTable,
    InvalidQuantTable,
    NoScanData,
    UnexpectedEndOfData,
    InvalidByteStuffing,
    OutOfMemory,
    InvalidHuffmanCode,
    MissingHuffmanTable,
    MissingQuantTable,
    InvalidDCCoefficient,
    InvalidACCoefficient,
    InvalidACValue,
    BlockStorageNotAllocated,
    RgbStorageNotAllocated,
};

// IDCT implementation based on stb_image
fn f2f(comptime x: f32) i32 {
    // 4096 = 1 << 12
    return @intFromFloat(x * 4096 + 0.5);
}

fn idct1D(s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32) struct { i32, i32, i32, i32, i32, i32, i32, i32 } {
    var p2 = s2;
    var p3 = s6;

    var p1 = (p2 + p3) * f2f(0.5411961);
    var t2 = p1 + p3 * f2f(-1.847759065);
    var t3 = p1 + p2 * f2f(0.765366865);
    p2 = s0;
    p3 = s4;
    var t0 = (p2 + p3) * 4096;
    var t1 = (p2 - p3) * 4096;
    const x0 = t0 + t3;
    const x3 = t0 - t3;
    const x1 = t1 + t2;
    const x2 = t1 - t2;
    t0 = s7;
    t1 = s5;
    t2 = s3;
    t3 = s1;
    p3 = t0 + t2;
    var p4 = t1 + t3;
    p1 = t0 + t3;
    p2 = t1 + t2;
    const p5 = (p3 + p4) * f2f(1.175875602);
    t0 = t0 * f2f(0.298631336);
    t1 = t1 * f2f(2.053119869);
    t2 = t2 * f2f(3.072711026);
    t3 = t3 * f2f(1.501321110);
    p1 = p5 + p1 * f2f(-0.899976223);
    p2 = p5 + p2 * f2f(-2.562915447);
    p3 = p3 * f2f(-1.961570560);
    p4 = p4 * f2f(-0.390180644);
    t3 += p1 + p4;
    t2 += p2 + p3;
    t1 += p2 + p4;
    t0 += p1 + p3;

    return .{ x0, x1, x2, x3, t0, t1, t2, t3 };
}

fn idct8x8(block: *[64]i32) void {
    // Pass 1: process columns
    for (0..8) |x| {
        const s0 = block[0 * 8 + x];
        const s1 = block[1 * 8 + x];
        const s2 = block[2 * 8 + x];
        const s3 = block[3 * 8 + x];
        const s4 = block[4 * 8 + x];
        const s5 = block[5 * 8 + x];
        const s6 = block[6 * 8 + x];
        const s7 = block[7 * 8 + x];

        var x0: i32 = 0;
        var x1: i32 = 0;
        var x2: i32 = 0;
        var x3: i32 = 0;
        var t0: i32 = 0;
        var t1: i32 = 0;
        var t2: i32 = 0;
        var t3: i32 = 0;

        x0, x1, x2, x3, t0, t1, t2, t3 = idct1D(s0, s1, s2, s3, s4, s5, s6, s7);

        x0 += 512;
        x1 += 512;
        x2 += 512;
        x3 += 512;

        block[0 * 8 + x] = (x0 + t3) >> 10;
        block[1 * 8 + x] = (x1 + t2) >> 10;
        block[2 * 8 + x] = (x2 + t1) >> 10;
        block[3 * 8 + x] = (x3 + t0) >> 10;
        block[4 * 8 + x] = (x3 - t0) >> 10;
        block[5 * 8 + x] = (x2 - t1) >> 10;
        block[6 * 8 + x] = (x1 - t2) >> 10;
        block[7 * 8 + x] = (x0 - t3) >> 10;
    }

    // Pass 2: process rows
    for (0..8) |y| {
        const s0 = block[y * 8 + 0];
        const s1 = block[y * 8 + 1];
        const s2 = block[y * 8 + 2];
        const s3 = block[y * 8 + 3];
        const s4 = block[y * 8 + 4];
        const s5 = block[y * 8 + 5];
        const s6 = block[y * 8 + 6];
        const s7 = block[y * 8 + 7];

        var x0: i32 = 0;
        var x1: i32 = 0;
        var x2: i32 = 0;
        var x3: i32 = 0;
        var t0: i32 = 0;
        var t1: i32 = 0;
        var t2: i32 = 0;
        var t3: i32 = 0;

        x0, x1, x2, x3, t0, t1, t2, t3 = idct1D(s0, s1, s2, s3, s4, s5, s6, s7);

        // add 0.5 scaled up by factor
        x0 += (1 << 17) / 2;
        x1 += (1 << 17) / 2;
        x2 += (1 << 17) / 2;
        x3 += (1 << 17) / 2;

        block[y * 8 + 0] = (x0 + t3) >> 17;
        block[y * 8 + 1] = (x1 + t2) >> 17;
        block[y * 8 + 2] = (x2 + t1) >> 17;
        block[y * 8 + 3] = (x3 + t0) >> 17;
        block[y * 8 + 4] = (x3 - t0) >> 17;
        block[y * 8 + 5] = (x2 - t1) >> 17;
        block[y * 8 + 6] = (x1 - t2) >> 17;
        block[y * 8 + 7] = (x0 - t3) >> 17;
    }
}

// Upsample chroma component for 4:2:0 subsampling using bilinear interpolation
fn upsampleChroma420(input: []const [64]i32, output: *[256]i32, h_blocks: u4, v_blocks: u4, max_h: u4, max_v: u4) void {
    // For 4:2:0, input is typically 1 block (8x8), output should be max_h*8 x max_v*8
    assert(h_blocks == 1 and v_blocks == 1);
    assert(input.len == 1);

    const src_block = &input[0];
    const dst_width = @as(usize, max_h) * 8;
    const dst_height = @as(usize, max_v) * 8;
    const scale_x = 8.0 / @as(f32, @floatFromInt(dst_width));
    const scale_y = 8.0 / @as(f32, @floatFromInt(dst_height));

    // Bilinear interpolation upsampling for better quality
    for (0..dst_height) |dst_y| {
        for (0..dst_width) |dst_x| {
            // Calculate source coordinates with sub-pixel precision
            const src_x_f = (@as(f32, @floatFromInt(dst_x)) + 0.5) * scale_x - 0.5;
            const src_y_f = (@as(f32, @floatFromInt(dst_y)) + 0.5) * scale_y - 0.5;

            // Get integer and fractional parts
            const x0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(src_x_f)))));
            const y0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(src_y_f)))));
            const x1 = @min(7, x0 + 1);
            const y1 = @min(7, y0 + 1);

            const fx = src_x_f - @as(f32, @floatFromInt(x0));
            const fy = src_y_f - @as(f32, @floatFromInt(y0));

            // Get the four surrounding pixels
            const p00 = @as(f32, @floatFromInt(src_block[@as(usize, @intCast(y0)) * 8 + @as(usize, @intCast(x0))]));
            const p10 = @as(f32, @floatFromInt(src_block[@as(usize, @intCast(y0)) * 8 + @as(usize, @intCast(x1))]));
            const p01 = @as(f32, @floatFromInt(src_block[@as(usize, @intCast(y1)) * 8 + @as(usize, @intCast(x0))]));
            const p11 = @as(f32, @floatFromInt(src_block[@as(usize, @intCast(y1)) * 8 + @as(usize, @intCast(x1))]));

            // Bilinear interpolation
            const interp_x0 = std.math.lerp(p00, p10, fx);
            const interp_x1 = std.math.lerp(p01, p11, fx);
            const result = std.math.lerp(interp_x0, interp_x1, fy);

            const dst_idx = dst_y * dst_width + dst_x;
            output[dst_idx] = @intFromFloat(@round(result));
        }
    }
}

// Block scan function that fills block storage (from master)
fn performBlockScan(decoder: *JpegDecoder) !void {
    if (decoder.block_storage == null) return error.BlockStorageNotAllocated;

    // Calculate maximum sampling factors
    var max_h_factor: u4 = 1;
    var max_v_factor: u4 = 1;
    for (decoder.components[0..decoder.num_components]) |comp| {
        max_h_factor = @max(max_h_factor, comp.h_sampling);
        max_v_factor = @max(max_v_factor, comp.v_sampling);
    }

    // Scan structure
    const noninterleaved = decoder.scan_components.len == 1 and decoder.scan_components[0].component_id == 1;
    const y_step = if (noninterleaved) 1 else max_v_factor;
    const x_step = if (noninterleaved) 1 else max_h_factor;

    // DC prediction values for each component
    var prediction_values: [4]i32 = @splat(0);

    var y: usize = 0;
    while (y < decoder.block_height) : (y += y_step) {
        var x: usize = 0;
        while (x < decoder.block_width) : (x += x_step) {
            // Decode each component at this position
            for (decoder.scan_components) |scan_comp| {
                // Find the component index for this scan component
                var component_index: usize = 0;
                var v_max: usize = undefined;
                var h_max: usize = undefined;

                for (decoder.components[0..decoder.num_components], 0..) |frame_component, i| {
                    if (frame_component.id == scan_comp.component_id) {
                        component_index = i;
                        v_max = if (noninterleaved) 1 else frame_component.v_sampling;
                        h_max = if (noninterleaved) 1 else frame_component.h_sampling;
                        break;
                    }
                }

                // Decode all blocks for this component in this MCU
                for (0..v_max) |v| {
                    for (0..h_max) |h| {
                        if (y + v >= decoder.block_height or x + h >= decoder.block_width) continue;

                        const block_id = (y + v) * decoder.block_width_actual + (x + h);
                        const block = &decoder.block_storage.?[block_id][component_index];

                        // Decode block directly into storage
                        try decodeBlockToStorage(decoder, scan_comp, block, &prediction_values[component_index]);
                    }
                }
            }
        }
    }
}

// Decode a single block directly into block storage (from master)
fn decodeBlockToStorage(decoder: *JpegDecoder, scan_comp: ScanComponent, block: *[64]i32, dc_prediction: *i32) !void {
    // Clear the block
    @memset(block, 0);

    // Decode DC coefficient
    const dc_table = decoder.dc_tables[scan_comp.dc_table_id] orelse return error.MissingHuffmanTable;
    const dc_symbol = try decoder.decodeHuffmanSymbol(&dc_table);

    if (dc_symbol > 11) return error.InvalidDCCoefficient;

    var dc_diff: i32 = 0;
    if (dc_symbol > 0) {
        const dc_bits = try decoder.bit_reader.getBits(@intCast(dc_symbol));
        dc_diff = @intCast(dc_bits);

        // Convert from unsigned to signed
        if (dc_bits < (@as(u32, 1) << @intCast(dc_symbol - 1))) {
            dc_diff = @as(i32, @intCast(dc_bits)) - @as(i32, @intCast((@as(u32, 1) << @intCast(dc_symbol)) - 1));
        }
    }

    dc_prediction.* += dc_diff;
    block[0] = dc_prediction.*;

    // Decode AC coefficients
    const ac_table = decoder.ac_tables[scan_comp.ac_table_id] orelse return error.MissingHuffmanTable;
    var k: usize = 1;

    while (k < 64) {
        const ac_symbol = try decoder.decodeHuffmanSymbol(&ac_table);

        if (ac_symbol == 0x00) {
            // End of block
            break;
        }

        if (ac_symbol == 0xF0) {
            // ZRL - 16 zeros
            k += 16;
            continue;
        }

        const zero_run = ac_symbol >> 4;
        const coeff_bits = ac_symbol & 0x0F;

        if (coeff_bits == 0) return error.InvalidACCoefficient;

        k += zero_run;
        if (k >= 64) break;

        const ac_bits = try decoder.bit_reader.getBits(@intCast(coeff_bits));
        var ac_value: i32 = @intCast(ac_bits);

        // Convert from unsigned to signed
        if (ac_bits < (@as(u32, 1) << @intCast(coeff_bits - 1))) {
            ac_value = @as(i32, @intCast(ac_bits)) - @as(i32, @intCast((@as(u32, 1) << @intCast(coeff_bits)) - 1));
        }

        block[zigzag[k]] = ac_value;
        k += 1;
    }
}

// Dequantize all blocks in storage
pub fn dequantizeAllBlocks(decoder: *JpegDecoder) !void {
    if (decoder.block_storage == null) return error.BlockStorageNotAllocated;

    // Apply dequantization to all blocks
    for (decoder.block_storage.?) |*block_set| {
        for (decoder.components[0..decoder.num_components], 0..) |comp, comp_idx| {
            const quant_table = decoder.quant_tables[comp.quant_table_id] orelse return error.MissingQuantTable;

            for (0..64) |i| {
                block_set[comp_idx][i] *= @as(i32, @intCast(quant_table[i]));
            }
        }
    }
}

// Apply IDCT to all blocks in storage
pub fn idctAllBlocks(decoder: *JpegDecoder) void {
    if (decoder.block_storage == null) return;

    // Apply IDCT to all blocks
    for (decoder.block_storage.?) |*block_set| {
        for (0..decoder.num_components) |comp_idx| {
            idct8x8(&block_set[comp_idx]);

            // Apply level shift (+128) only to Y component (component 0) - master's approach
            // Cb and Cr components stay centered around 0
            if (comp_idx == 0) {
                for (0..64) |i| {
                    block_set[comp_idx][i] += 128;
                }
            }
        }
    }
}

// Upsample chroma for a specific Y block within an MCU
fn upsampleChromaForBlock(decoder: *JpegDecoder, mcu_col: usize, mcu_row: usize, h_offset: usize, v_offset: usize, max_h: u4, max_v: u4, cb_out: *[64]i32, cr_out: *[64]i32) void {

    // For 4:2:0, we need to interpolate from the 2x2 pixel grid at the MCU level to 8x8 for each Y block
    // The h_offset and v_offset tell us which quadrant of the MCU we're in

    // Get the chroma block for this MCU
    const chroma_y = mcu_row * max_v;
    const chroma_x = mcu_col * max_h;
    if (chroma_y >= decoder.block_height or chroma_x >= decoder.block_width) {
        @memset(cb_out, 0);
        @memset(cr_out, 0);
        return;
    }

    const chroma_block_index = chroma_y * decoder.block_width_actual + chroma_x;
    const cb_block = &decoder.block_storage.?[chroma_block_index][1];
    const cr_block = &decoder.block_storage.?[chroma_block_index][2];

    // For 4:2:0 with 2x2 Y blocks per MCU, we need to map the 8x8 chroma to each 8x8 Y block
    // Each Y block gets a quarter of the chroma samples, interpolated
    if (max_h == 2 and max_v == 2) {
        // Calculate which 4x4 region of the chroma block maps to this Y block
        const chroma_offset_x = h_offset * 4;
        const chroma_offset_y = v_offset * 4;

        // Simple approach: replicate the 4x4 chroma region to fill the 8x8 Y block
        for (0..8) |y| {
            for (0..8) |x| {
                // Map to the 4x4 region in the original 8x8 chroma block
                const src_y = chroma_offset_y + (y / 2);
                const src_x = chroma_offset_x + (x / 2);
                const src_idx = src_y * 8 + src_x;
                const dst_idx = y * 8 + x;

                cb_out[dst_idx] = cb_block[src_idx];
                cr_out[dst_idx] = cr_block[src_idx];
            }
        }
    } else {
        // For other subsampling ratios, just copy the chroma block
        @memcpy(cb_out, cb_block);
        @memcpy(cr_out, cr_block);
    }
}

// Convert YCbCr blocks to RGB with proper 4:2:0 chroma upsampling
pub fn ycbcrToRgbAllBlocks(decoder: *JpegDecoder) !void {
    if (decoder.block_storage == null) return error.BlockStorageNotAllocated;

    if (decoder.num_components == 1) {
        // Grayscale - blocks already level-shifted in IDCT
        for (decoder.block_storage.?, 0..) |*block_set, idx| {
            for (0..64) |i| {
                const y_val = block_set[0][i];
                const rgb_val: u8 = @intCast(std.math.clamp(y_val, 0, 255));
                decoder.rgb_storage.?[idx][0][i] = rgb_val; // R
                decoder.rgb_storage.?[idx][1][i] = rgb_val; // G
                decoder.rgb_storage.?[idx][2][i] = rgb_val; // B
            }
        }
        return;
    }

    // Check chroma subsampling mode
    const max_h = decoder.components[0].h_sampling;
    const max_v = decoder.components[0].v_sampling;

    // 4:4:4 - no chroma subsampling, each component has same number of blocks
    if (max_h == 1 and max_v == 1) {
        for (decoder.block_storage.?, 0..) |*block_set, idx| {
            // Direct YCbCr to RGB conversion without upsampling
            for (0..64) |i| {
                const Y = block_set[0][i];
                const Cb = block_set[1][i];
                const Cr = block_set[2][i];

                const ycbcr: Ycbcr = .{
                    .y = @as(f32, @floatFromInt(Y)),
                    .cb = @as(f32, @floatFromInt(Cb + 128)),
                    .cr = @as(f32, @floatFromInt(Cr + 128)),
                };
                const rgb = ycbcr.toRgb();

                decoder.rgb_storage.?[idx][0][i] = rgb.r;
                decoder.rgb_storage.?[idx][1][i] = rgb.g;
                decoder.rgb_storage.?[idx][2][i] = rgb.b;
            }
        }
        return;
    }

    // 4:2:2 - horizontal chroma subsampling only
    if (max_h == 2 and max_v == 1) {
        var mcu_y: usize = 0;
        while (mcu_y < decoder.block_height) : (mcu_y += 1) {
            var mcu_x: usize = 0;
            while (mcu_x < decoder.block_width) : (mcu_x += 2) {
                const chroma_block_index = mcu_y * decoder.block_width_actual + mcu_x;

                // Process the 2 Y blocks in this MCU
                for (0..2) |h| {
                    const y_block_x = mcu_x + h;
                    if (y_block_x >= decoder.block_width) continue;

                    const y_block_index = mcu_y * decoder.block_width_actual + y_block_x;

                    for (0..64) |pixel_idx| {
                        const py = pixel_idx / 8;
                        const px = pixel_idx % 8;

                        const Y = decoder.block_storage.?[y_block_index][0][pixel_idx];

                        // Horizontal interpolation for chroma
                        const chroma_x_f = (@as(f32, @floatFromInt(h * 8 + px)) + 0.5) * 0.5 - 0.5;
                        const cx0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(chroma_x_f)))));
                        const cx1 = @min(7, cx0 + 1);
                        const fx = chroma_x_f - @as(f32, @floatFromInt(cx0));

                        const chroma_idx = py * 8 + @as(usize, @intCast(cx0));
                        const chroma_idx_next = py * 8 + @as(usize, @intCast(cx1));

                        const cb0 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][chroma_idx]));
                        const cb1 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][chroma_idx_next]));
                        const Cb = @as(i32, @intFromFloat(@round(std.math.lerp(cb0, cb1, fx))));

                        const cr0 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][chroma_idx]));
                        const cr1 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][chroma_idx_next]));
                        const Cr = @as(i32, @intFromFloat(@round(std.math.lerp(cr0, cr1, fx))));

                        const ycbcr: Ycbcr = .{ .y = @as(f32, @floatFromInt(Y)), .cb = @as(f32, @floatFromInt(Cb + 128)), .cr = @as(f32, @floatFromInt(Cr + 128)) };
                        const rgb = ycbcr.toRgb();

                        decoder.rgb_storage.?[y_block_index][0][pixel_idx] = rgb.r;
                        decoder.rgb_storage.?[y_block_index][1][pixel_idx] = rgb.g;
                        decoder.rgb_storage.?[y_block_index][2][pixel_idx] = rgb.b;
                    }
                }
            }
        }
        return;
    }

    // 4:1:1 - 4:1 horizontal chroma subsampling
    if (max_h == 4 and max_v == 1) {
        var mcu_y: usize = 0;
        while (mcu_y < decoder.block_height) : (mcu_y += 1) {
            var mcu_x: usize = 0;
            while (mcu_x < decoder.block_width) : (mcu_x += 4) {
                const chroma_block_index = mcu_y * decoder.block_width_actual + mcu_x;

                // Process the 4 Y blocks in this MCU
                for (0..4) |h| {
                    const y_block_x = mcu_x + h;
                    if (y_block_x >= decoder.block_width) continue;

                    const y_block_index = mcu_y * decoder.block_width_actual + y_block_x;

                    for (0..64) |pixel_idx| {
                        const py = pixel_idx / 8;
                        const px = pixel_idx % 8;

                        const Y = decoder.block_storage.?[y_block_index][0][pixel_idx];

                        // Horizontal interpolation for 4:1 chroma
                        const chroma_x_f = (@as(f32, @floatFromInt(h * 8 + px)) + 0.5) * 0.25 - 0.5;
                        const cx0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(chroma_x_f)))));
                        const cx1 = @min(7, cx0 + 1);
                        const fx = chroma_x_f - @as(f32, @floatFromInt(cx0));

                        const chroma_idx = py * 8 + @as(usize, @intCast(cx0));
                        const chroma_idx_next = py * 8 + @as(usize, @intCast(cx1));

                        const cb0 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][chroma_idx]));
                        const cb1 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][chroma_idx_next]));
                        const Cb = @as(i32, @intFromFloat(@round(std.math.lerp(cb0, cb1, fx))));

                        const cr0 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][chroma_idx]));
                        const cr1 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][chroma_idx_next]));
                        const Cr = @as(i32, @intFromFloat(@round(std.math.lerp(cr0, cr1, fx))));

                        const ycbcr: Ycbcr = .{ .y = @as(f32, @floatFromInt(Y)), .cb = @as(f32, @floatFromInt(Cb + 128)), .cr = @as(f32, @floatFromInt(Cr + 128)) };
                        const rgb = ycbcr.toRgb();

                        decoder.rgb_storage.?[y_block_index][0][pixel_idx] = rgb.r;
                        decoder.rgb_storage.?[y_block_index][1][pixel_idx] = rgb.g;
                        decoder.rgb_storage.?[y_block_index][2][pixel_idx] = rgb.b;
                    }
                }
            }
        }
        return;
    }

    // 4:2:0 chroma subsampling (both horizontal and vertical)
    // Process in MCU units
    var mcu_y: usize = 0;
    while (mcu_y < decoder.block_height) : (mcu_y += max_v) {
        var mcu_x: usize = 0;
        while (mcu_x < decoder.block_width) : (mcu_x += max_h) {
            // Get the chroma block (stored at MCU origin)
            const chroma_block_index = mcu_y * decoder.block_width_actual + mcu_x;

            // Process each Y block in this MCU
            for (0..max_v) |v| {
                for (0..max_h) |h| {
                    const y_block_y = mcu_y + v;
                    const y_block_x = mcu_x + h;

                    if (y_block_y >= decoder.block_height or y_block_x >= decoder.block_width) continue;

                    const y_block_index = y_block_y * decoder.block_width_actual + y_block_x;

                    // Convert this Y block using upsampled chroma
                    for (0..64) |pixel_idx| {
                        const py = pixel_idx / 8;
                        const px = pixel_idx % 8;

                        const Y = decoder.block_storage.?[y_block_index][0][pixel_idx];

                        // Bilinear interpolation for chroma upsampling
                        const chroma_y_f = (@as(f32, @floatFromInt(v * 8 + py)) + 0.5) * 0.5 - 0.5;
                        const chroma_x_f = (@as(f32, @floatFromInt(h * 8 + px)) + 0.5) * 0.5 - 0.5;

                        const cy0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(chroma_y_f)))));
                        const cx0 = @max(0, @min(7, @as(i32, @intFromFloat(@floor(chroma_x_f)))));
                        const cy1 = @min(7, cy0 + 1);
                        const cx1 = @min(7, cx0 + 1);

                        const fy = chroma_y_f - @as(f32, @floatFromInt(cy0));
                        const fx = chroma_x_f - @as(f32, @floatFromInt(cx0));

                        // Get the four surrounding chroma values for Cb
                        const cb00 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][@as(usize, @intCast(cy0)) * 8 + @as(usize, @intCast(cx0))]));
                        const cb10 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][@as(usize, @intCast(cy0)) * 8 + @as(usize, @intCast(cx1))]));
                        const cb01 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][@as(usize, @intCast(cy1)) * 8 + @as(usize, @intCast(cx0))]));
                        const cb11 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][1][@as(usize, @intCast(cy1)) * 8 + @as(usize, @intCast(cx1))]));

                        // Get the four surrounding chroma values for Cr
                        const cr00 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][@as(usize, @intCast(cy0)) * 8 + @as(usize, @intCast(cx0))]));
                        const cr10 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][@as(usize, @intCast(cy0)) * 8 + @as(usize, @intCast(cx1))]));
                        const cr01 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][@as(usize, @intCast(cy1)) * 8 + @as(usize, @intCast(cx0))]));
                        const cr11 = @as(f32, @floatFromInt(decoder.block_storage.?[chroma_block_index][2][@as(usize, @intCast(cy1)) * 8 + @as(usize, @intCast(cx1))]));

                        // Bilinear interpolation
                        const cb_interp_x0 = std.math.lerp(cb00, cb10, fx);
                        const cb_interp_x1 = std.math.lerp(cb01, cb11, fx);
                        const Cb = @as(i32, @intFromFloat(@round(std.math.lerp(cb_interp_x0, cb_interp_x1, fy))));

                        const cr_interp_x0 = std.math.lerp(cr00, cr10, fx);
                        const cr_interp_x1 = std.math.lerp(cr01, cr11, fx);
                        const Cr = @as(i32, @intFromFloat(@round(std.math.lerp(cr_interp_x0, cr_interp_x1, fy))));

                        // Convert using library's high-quality YCbCr conversion (KEY: this is what master did!)
                        const ycbcr: Ycbcr = .{ .y = @as(f32, @floatFromInt(Y)), .cb = @as(f32, @floatFromInt(Cb + 128)), .cr = @as(f32, @floatFromInt(Cr + 128)) };
                        const rgb = ycbcr.toRgb();

                        // Store RGB in separate storage to avoid overwriting chroma data
                        decoder.rgb_storage.?[y_block_index][0][pixel_idx] = rgb.r;
                        decoder.rgb_storage.?[y_block_index][1][pixel_idx] = rgb.g;
                        decoder.rgb_storage.?[y_block_index][2][pixel_idx] = rgb.b;
                    }
                }
            }
        }
    }
}

// Render RGB blocks to pixels (simple after YCbCr conversion)
pub fn renderRgbBlocksToPixels(comptime T: type, decoder: *JpegDecoder, img: *Image(T)) !void {
    if (decoder.rgb_storage == null) return error.RgbStorageNotAllocated;

    // Simple rendering - read from RGB storage
    var block_y: usize = 0;
    while (block_y < decoder.block_height) : (block_y += 1) {
        const pixel_y = block_y * 8;

        var block_x: usize = 0;
        while (block_x < decoder.block_width) : (block_x += 1) {
            const block_index = block_y * decoder.block_width_actual + block_x;
            const pixel_x = block_x * 8;

            for (0..8) |y| {
                for (0..8) |x| {
                    if (pixel_y + y >= decoder.height or pixel_x + x >= decoder.width) {
                        continue;
                    }

                    const pixel_idx = y * 8 + x;
                    const r = decoder.rgb_storage.?[block_index][0][pixel_idx];
                    const g = decoder.rgb_storage.?[block_index][1][pixel_idx];
                    const b = decoder.rgb_storage.?[block_index][2][pixel_idx];

                    const rgb = Rgb{ .r = r, .g = g, .b = b };
                    img.at(pixel_y + y, pixel_x + x).* = convertColor(T, rgb);
                }
            }
        }
    }
}

/// Load JPEG file from disk and decode to specified pixel type.
/// Supports baseline DCT (SOF0) and progressive DCT (SOF2) JPEG formats.
/// Handles grayscale (1 component) and YCbCr color (3 components) with 4:4:4, 4:2:2, 4:1:1, and 4:2:0 chroma subsampling.
///
/// Parameters:
/// - T: Desired output pixel type (u8, Rgb, etc.) - color conversion applied if needed
/// - allocator: Memory allocator for image data
/// - file_path: Path to JPEG file
///
/// Returns: Decoded Image(T) with automatic color space conversion from source format
///
/// Errors: InvalidJpegFile, UnsupportedJpegFormat, OutOfMemory, and various JPEG parsing errors
pub fn load(comptime T: type, allocator: Allocator, file_path: []const u8) !Image(T) {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // Early signature validation - check for JPEG SOI marker
    var signature_buffer: [2]u8 = undefined;
    const bytes_read = try file.read(&signature_buffer);
    if (bytes_read < 2 or !std.mem.eql(u8, signature_buffer[0..2], &signature)) {
        return error.InvalidJpegFile;
    }

    // Reset file position and read entire file
    try file.seekTo(0);
    const file_size = try file.getEndPos();
    const data = try allocator.alloc(u8, file_size);
    defer allocator.free(data);

    _ = try file.read(data);

    var decoder = try decode(allocator, data);
    defer decoder.deinit();

    // Create output image
    var img = try Image(T).init(allocator, decoder.height, decoder.width);
    errdefer img.deinit(allocator);

    // Complete block-based pipeline:
    // Step 1: Decode all blocks into storage (storage allocated during parseSOF)
    // For baseline JPEG, decode blocks here. For progressive, decode() already did it.
    if (decoder.frame_type == .baseline) {
        try performBlockScan(&decoder);
    }

    // Step 2: Apply dequantization to all blocks
    try dequantizeAllBlocks(&decoder);

    // Step 3: Apply IDCT to all blocks
    idctAllBlocks(&decoder);

    // Step 4: Convert YCbCr to RGB with proper chroma upsampling (RGB storage allocated during parseSOF)
    try ycbcrToRgbAllBlocks(&decoder);

    // Step 4: Render RGB blocks to pixels
    try renderRgbBlocksToPixels(T, &decoder, &img);

    return img;
}

test "JPEG encode -> decode RGB roundtrip" {
    const gpa = std.testing.allocator;

    var img = try Image(Rgb).init(gpa, 16, 16);
    defer img.deinit(gpa);
    for (0..img.rows) |y| {
        for (0..img.cols) |x| {
            const r: u8 = @intCast((x * 255) / (img.cols - 1));
            const g: u8 = @intCast((y * 255) / (img.rows - 1));
            const b: u8 = @intCast(((x + y) * 255) / (img.cols + img.rows - 2));
            img.at(y, x).* = .{ .r = r, .g = g, .b = b };
        }
    }

    const bytes = try encodeImage(Rgb, gpa, img, .{ .quality = 85 });
    defer gpa.free(bytes);

    var decoder = try decode(gpa, bytes);
    defer decoder.deinit();
    if (decoder.frame_type == .baseline) try performBlockScan(&decoder);
    try dequantizeAllBlocks(&decoder);
    idctAllBlocks(&decoder);
    try ycbcrToRgbAllBlocks(&decoder);
    var out = try Image(Rgb).init(gpa, decoder.height, decoder.width);
    defer out.deinit(gpa);
    try renderRgbBlocksToPixels(Rgb, &decoder, &out);

    const ps = try img.psnr(out);
    try std.testing.expect(ps > 28.0);
}

test "JPEG encode -> decode grayscale roundtrip" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).init(gpa, 16, 16);
    defer img.deinit(gpa);
    for (0..img.rows) |y| {
        for (0..img.cols) |x| {
            img.at(y, x).* = @intCast(((x + y) * 255) / (img.cols + img.rows - 2));
        }
    }
    const bytes = try encodeImage(u8, gpa, img, .{ .quality = 85 });
    defer gpa.free(bytes);

    var decoder = try decode(gpa, bytes);
    defer decoder.deinit();
    if (decoder.frame_type == .baseline) try performBlockScan(&decoder);
    try dequantizeAllBlocks(&decoder);
    idctAllBlocks(&decoder);
    try ycbcrToRgbAllBlocks(&decoder);
    var out = try Image(Rgb).init(gpa, decoder.height, decoder.width);
    defer out.deinit(gpa);
    try renderRgbBlocksToPixels(Rgb, &decoder, &out);

    // Convert original gray to RGB for PSNR
    var gray_rgb = try img.convert(Rgb, gpa);
    defer gray_rgb.deinit(gpa);
    const ps = try gray_rgb.psnr(out);
    try std.testing.expect(ps > 24.0);
}

// Basic tests
test "JPEG marker parsing" {
    const testing = std.testing;

    // Test marker conversion
    const soi_bytes = [2]u8{ 0xFF, 0xD8 };
    const soi = Marker.fromBytes(soi_bytes);
    try testing.expect(soi == .SOI);

    const sof0_bytes = [2]u8{ 0xFF, 0xC0 };
    const sof0 = Marker.fromBytes(sof0_bytes);
    try testing.expect(sof0 == .SOF0);
}

test "BitReader basic operations" {
    const testing = std.testing;

    const data = [_]u8{ 0b10110011, 0b01010101 };
    var reader = BitReader.init(&data);

    // Read first 4 bits
    const bits1 = try reader.getBits(4);
    try testing.expectEqual(@as(u16, 0b1011), bits1);

    // Read next 4 bits
    const bits2 = try reader.getBits(4);
    try testing.expectEqual(@as(u16, 0b0011), bits2);

    // Read next 8 bits
    const bits3 = try reader.getBits(8);
    try testing.expectEqual(@as(u16, 0b01010101), bits3);
}

test "Ycbcr to RGB conversion" {
    const testing = std.testing;

    // Test grayscale - standard Y=128
    const gray_ycbcr: Ycbcr = .{ .y = 128, .cb = 128, .cr = 128 };
    const gray = gray_ycbcr.toRgb();
    try testing.expectEqual(@as(u8, 128), gray.r);
    try testing.expectEqual(@as(u8, 128), gray.g);
    try testing.expectEqual(@as(u8, 128), gray.b);

    // Test white - standard Y=255
    const white_ycbcr: Ycbcr = .{ .y = 255, .cb = 128, .cr = 128 };
    const white = white_ycbcr.toRgb();
    try testing.expectEqual(@as(u8, 255), white.r);
    try testing.expectEqual(@as(u8, 255), white.g);
    try testing.expectEqual(@as(u8, 255), white.b);

    // Test black - standard Y=0
    const black_ycbcr: Ycbcr = .{ .y = 0, .cb = 128, .cr = 128 };
    const black = black_ycbcr.toRgb();
    try testing.expectEqual(@as(u8, 0), black.r);
    try testing.expectEqual(@as(u8, 0), black.g);
    try testing.expectEqual(@as(u8, 0), black.b);
}
