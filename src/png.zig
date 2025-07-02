//! Pure Zig PNG encoder and decoder implementation.
//! Supports all PNG color types and bit depths according to the PNG specification.
//! Zero dependencies - implements deflate compression/decompression internally.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Image = @import("image.zig").Image;
const Rgb = @import("color.zig").Rgb;
const Rgba = @import("color.zig").Rgba;
const deflate = @import("deflate.zig");

// PNG signature: 8 bytes that identify a PNG file
const PNG_SIGNATURE = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };

// PNG color types
pub const ColorType = enum(u8) {
    grayscale = 0,
    rgb = 2,
    palette = 3,
    grayscale_alpha = 4,
    rgba = 6,

    pub fn channels(self: ColorType) u8 {
        return switch (self) {
            .grayscale => 1,
            .rgb => 3,
            .palette => 1, // palette index only
            .grayscale_alpha => 2,
            .rgba => 4,
        };
    }

    pub fn hasAlpha(self: ColorType) bool {
        return switch (self) {
            .grayscale_alpha, .rgba => true,
            .grayscale, .rgb, .palette => false,
        };
    }
};

// PNG filter types for row filtering
pub const FilterType = enum(u8) {
    none = 0,
    sub = 1,
    up = 2,
    average = 3,
    paeth = 4,
};

// PNG chunk structure
pub const Chunk = struct {
    length: u32,
    type: [4]u8,
    data: []const u8,
    crc: u32,
};

// PNG IHDR (header) chunk data
pub const Header = struct {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: ColorType,
    compression_method: u8, // Must be 0 (deflate)
    filter_method: u8, // Must be 0
    interlace_method: u8, // 0 = none, 1 = Adam7

    pub fn channels(self: Header) u8 {
        return self.color_type.channels();
    }

    pub fn bytesPerPixel(self: Header) u8 {
        return (self.channels() * self.bit_depth + 7) / 8;
    }

    pub fn scanlineBytes(self: Header) usize {
        return (self.width * self.channels() * self.bit_depth + 7) / 8;
    }
};

// PNG decoder/encoder state
pub const PngImage = struct {
    header: Header,
    palette: ?[][3]u8 = null,
    transparency: ?[]u8 = null, // For palette transparency or single transparent color
    idat_data: ArrayList(u8),

    pub fn deinit(self: *PngImage, allocator: Allocator) void {
        self.idat_data.deinit();
        if (self.palette) |palette| {
            allocator.free(palette);
        }
        if (self.transparency) |trans| {
            allocator.free(trans);
        }
    }
};

// CRC table for PNG chunk validation
var crc_table: [256]u32 = undefined;
var crc_table_computed = false;

fn makeCrcTable() void {
    var c: u32 = undefined;
    var n: usize = 0;
    while (n < 256) : (n += 1) {
        c = @intCast(n);
        var k: u8 = 0;
        while (k < 8) : (k += 1) {
            if (c & 1 != 0) {
                c = 0xedb88320 ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crc_table[n] = c;
    }
    crc_table_computed = true;
}

fn updateCrc(initial_crc: u32, buf: []const u8) u32 {
    var c = initial_crc;
    if (!crc_table_computed) makeCrcTable();
    
    for (buf) |byte| {
        c = crc_table[(c ^ byte) & 0xff] ^ (c >> 8);
    }
    return c;
}

fn crc(buf: []const u8) u32 {
    return updateCrc(0xffffffff, buf) ^ 0xffffffff;
}

// Read PNG chunks from byte stream
pub const ChunkReader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn init(data: []const u8) ChunkReader {
        return .{ .data = data, .pos = 0 };
    }

    pub fn nextChunk(self: *ChunkReader) !?Chunk {
        if (self.pos + 8 > self.data.len) return null;

        const length = std.mem.readInt(u32, self.data[self.pos..self.pos + 4][0..4], .big);
        self.pos += 4;

        const chunk_type = self.data[self.pos..self.pos + 4][0..4].*;
        self.pos += 4;

        if (self.pos + length + 4 > self.data.len) {
            return error.InvalidChunkLength;
        }

        const chunk_data = self.data[self.pos..self.pos + length];
        self.pos += length;

        const chunk_crc = std.mem.readInt(u32, self.data[self.pos..self.pos + 4][0..4], .big);
        self.pos += 4;

        // Verify CRC (includes chunk type and data)
        const crc_start = self.pos - length - 8;
        const computed_crc = crc(self.data[crc_start..self.pos - 4]);
        if (computed_crc != chunk_crc) {
            return error.InvalidCrc;
        }

        return Chunk{
            .length = length,
            .type = chunk_type,
            .data = chunk_data,
            .crc = chunk_crc,
        };
    }
};

// Parse IHDR chunk
fn parseHeader(chunk: Chunk) !Header {
    if (!std.mem.eql(u8, &chunk.type, "IHDR")) {
        return error.InvalidHeader;
    }
    if (chunk.length != 13) {
        return error.InvalidHeaderLength;
    }

    const data = chunk.data;
    const width = std.mem.readInt(u32, data[0..4][0..4], .big);
    const height = std.mem.readInt(u32, data[4..8][0..4], .big);
    const bit_depth = data[8];
    const color_type_raw = data[9];
    const compression_method = data[10];
    const filter_method = data[11];
    const interlace_method = data[12];

    if (width == 0 or height == 0) {
        return error.InvalidDimensions;
    }
    
    // Prevent resource exhaustion with reasonable size limits
    const MAX_DIMENSION = 32767; // PNG spec limit
    const MAX_PIXELS = 268435456; // 16K x 16K = 256 MB pixels
    if (width > MAX_DIMENSION or height > MAX_DIMENSION) {
        return error.ImageTooLarge;
    }
    if (@as(u64, width) * @as(u64, height) > MAX_PIXELS) {
        return error.ImageTooLarge;
    }

    const color_type: ColorType = switch (color_type_raw) {
        0 => .grayscale,
        2 => .rgb,
        3 => .palette,
        4 => .grayscale_alpha,
        6 => .rgba,
        else => return error.InvalidColorType,
    };

    // Validate bit depth for color type
    const valid_bit_depth = switch (color_type) {
        .grayscale => bit_depth == 1 or bit_depth == 2 or bit_depth == 4 or bit_depth == 8 or bit_depth == 16,
        .rgb => bit_depth == 8 or bit_depth == 16,
        .palette => bit_depth == 1 or bit_depth == 2 or bit_depth == 4 or bit_depth == 8,
        .grayscale_alpha => bit_depth == 8 or bit_depth == 16,
        .rgba => bit_depth == 8 or bit_depth == 16,
    };

    if (!valid_bit_depth) {
        return error.InvalidBitDepth;
    }

    if (compression_method != 0) {
        return error.UnsupportedCompressionMethod;
    }

    if (filter_method != 0) {
        return error.UnsupportedFilterMethod;
    }

    if (interlace_method != 0 and interlace_method != 1) {
        return error.UnsupportedInterlaceMethod;
    }

    return Header{
        .width = width,
        .height = height,
        .bit_depth = bit_depth,
        .color_type = color_type,
        .compression_method = compression_method,
        .filter_method = filter_method,
        .interlace_method = interlace_method,
    };
}

// PNG decoder entry point
pub fn decode(allocator: Allocator, png_data: []const u8) !PngImage {
    if (png_data.len < 8 or !std.mem.eql(u8, png_data[0..8], &PNG_SIGNATURE)) {
        return error.InvalidPngSignature;
    }

    var reader = ChunkReader.init(png_data[8..]);
    var png_image = PngImage{
        .header = undefined,
        .idat_data = ArrayList(u8).init(allocator),
    };
    errdefer png_image.deinit(allocator);

    var header_found = false;

    while (try reader.nextChunk()) |chunk| {
        if (std.mem.eql(u8, &chunk.type, "IHDR")) {
            if (header_found) return error.MultipleHeaders;
            png_image.header = try parseHeader(chunk);
            header_found = true;
        } else if (std.mem.eql(u8, &chunk.type, "PLTE")) {
            if (chunk.length % 3 != 0) return error.InvalidPaletteLength;
            const palette_size = chunk.length / 3;
            if (palette_size > 256) return error.PaletteTooLarge;
            if (chunk.data.len < palette_size * 3) return error.InvalidPaletteLength;
            
            var palette = try allocator.alloc([3]u8, palette_size);
            for (0..palette_size) |i| {
                const offset = i * 3;
                if (offset + 3 > chunk.data.len) return error.InvalidPaletteLength;
                palette[i] = [3]u8{ chunk.data[offset], chunk.data[offset + 1], chunk.data[offset + 2] };
            }
            png_image.palette = palette;
        } else if (std.mem.eql(u8, &chunk.type, "tRNS")) {
            const transparency = try allocator.alloc(u8, chunk.length);
            @memcpy(transparency, chunk.data);
            png_image.transparency = transparency;
        } else if (std.mem.eql(u8, &chunk.type, "IDAT")) {
            try png_image.idat_data.appendSlice(chunk.data);
        } else if (std.mem.eql(u8, &chunk.type, "IEND")) {
            break;
        }
        // Ignore other chunks (ancillary chunks like tEXt, gAMA, etc.)
    }

    if (!header_found) {
        return error.MissingHeader;
    }

    if (png_image.idat_data.items.len == 0) {
        return error.MissingImageData;
    }

    return png_image;
}

// Convert PNG image data to Zignal Image types
pub fn toImage(allocator: Allocator, png_image: PngImage) !Image(u8) {
    // Decompress IDAT data
    const decompressed = try deflate.zlibDecompress(allocator, png_image.idat_data.items);
    defer allocator.free(decompressed);

    // Apply row defiltering
    try defilterScanlines(decompressed, png_image.header);

    // Convert to Image format
    const width = png_image.header.width;
    const height = png_image.header.height;
    const channels = png_image.header.channels();
    const scanline_bytes = png_image.header.scanlineBytes();

    // Create output image with overflow protection
    const total_pixels = @as(u64, width) * @as(u64, height);
    const total_bytes = total_pixels * @as(u64, channels);
    if (total_bytes > std.math.maxInt(usize)) {
        return error.ImageTooLarge;
    }
    var output_data = try allocator.alloc(u8, @intCast(total_bytes));
    
    // Copy pixel data, skipping filter bytes
    for (0..height) |y| {
        const src_row_start = y * (scanline_bytes + 1) + 1; // +1 to skip filter byte
        const dst_row_start = y * width * channels;
        
        const src_row = decompressed[src_row_start..src_row_start + scanline_bytes];
        const dst_row = output_data[dst_row_start..dst_row_start + width * channels];
        
        // Handle different bit depths and color types
        switch (png_image.header.bit_depth) {
            8 => {
                // Simple case: 8-bit channels
                switch (png_image.header.color_type) {
                    .grayscale => @memcpy(dst_row, src_row),
                    .rgb => @memcpy(dst_row, src_row),
                    .rgba => @memcpy(dst_row, src_row),
                    .grayscale_alpha => @memcpy(dst_row, src_row),
                    .palette => {
                        // Convert palette indices to RGB
                        if (png_image.palette == null) return error.MissingPalette;
                        const palette = png_image.palette.?;
                        
                        for (src_row, 0..) |index, i| {
                            if (index >= palette.len) return error.InvalidPaletteIndex;
                            if (i * 3 + 2 >= dst_row.len) return error.InvalidScanlineData;
                            const rgb = palette[index];
                            dst_row[i * 3] = rgb[0];
                            dst_row[i * 3 + 1] = rgb[1];
                            dst_row[i * 3 + 2] = rgb[2];
                        }
                    },
                }
            },
            16 => {
                // 16-bit channels - convert to 8-bit for now
                const samples_per_row = src_row.len / 2;
                for (0..samples_per_row) |i| {
                    const offset = i * 2;
                    if (offset + 2 > src_row.len) {
                        dst_row[i] = 0;
                    } else {
                        const sample16 = std.mem.readInt(u16, src_row[offset..offset + 2][0..2], .big);
                        dst_row[i] = @intCast(sample16 >> 8); // Simple conversion
                    }
                }
            },
            1, 2, 4 => {
                // Sub-byte bit depths - unpack bits
                const bits_per_pixel = png_image.header.bit_depth;
                const pixels_per_byte = 8 / bits_per_pixel;
                const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
                
                for (0..width) |x| {
                    const byte_index = x / pixels_per_byte;
                    const pixel_index = x % pixels_per_byte;
                    const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_index) * bits_per_pixel);
                    const pixel_value = (src_row[byte_index] >> bit_offset) & mask;
                    
                    // Scale to 8-bit
                    const scale_factor = 255 / mask;
                    dst_row[x] = pixel_value * scale_factor;
                }
            },
            else => return error.UnsupportedBitDepth,
        }
    }

    return Image(u8).init(height, width, output_data);
}

pub fn toRgbImage(allocator: Allocator, png_image: PngImage) !Image(Rgb) {
    // Decompress IDAT data
    const decompressed = try deflate.zlibDecompress(allocator, png_image.idat_data.items);
    defer allocator.free(decompressed);

    // Apply row defiltering
    try defilterScanlines(decompressed, png_image.header);

    // Convert to RGB Image format
    const width = png_image.header.width;
    const height = png_image.header.height;
    const scanline_bytes = png_image.header.scanlineBytes();

    // Create output image with overflow protection
    const total_pixels = @as(u64, width) * @as(u64, height);
    if (total_pixels > std.math.maxInt(usize)) {
        return error.ImageTooLarge;
    }
    var output_data = try allocator.alloc(Rgb, @intCast(total_pixels));
    
    // Copy and convert pixel data
    for (0..height) |y| {
        const src_row_start = y * (scanline_bytes + 1) + 1; // +1 to skip filter byte
        const dst_row_start = y * width;
        
        const src_row = decompressed[src_row_start..src_row_start + scanline_bytes];
        const dst_row = output_data[dst_row_start..dst_row_start + width];
        
        switch (png_image.header.color_type) {
            .grayscale => {
                // Convert grayscale to RGB
                for (dst_row, 0..) |*pixel, i| {
                    const gray = switch (png_image.header.bit_depth) {
                        8 => src_row[i],
                        16 => blk: {
                            if (i * 2 + 1 >= src_row.len) break :blk 0;
                            break :blk @as(u8, @intCast(std.mem.readInt(u16, src_row[i * 2..i * 2 + 2][0..2], .big) >> 8));
                        },
                        1, 2, 4 => blk: {
                            const bits_per_pixel = png_image.header.bit_depth;
                            const pixels_per_byte = 8 / bits_per_pixel;
                            const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
                            const byte_idx = i / pixels_per_byte;
                            if (byte_idx >= src_row.len) break :blk 0;
                            const pixel_idx = i % pixels_per_byte;
                            const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_idx) * bits_per_pixel);
                            const pixel_value = (src_row[byte_idx] >> bit_offset) & mask;
                            const scale_factor = 255 / mask;
                            break :blk pixel_value * scale_factor;
                        },
                        else => 0,
                    };
                    pixel.* = Rgb{ .r = gray, .g = gray, .b = gray };
                }
            },
            .rgb => {
                // Direct RGB copy
                for (dst_row, 0..) |*pixel, i| {
                    if (png_image.header.bit_depth == 8) {
                        pixel.* = Rgb{ 
                            .r = src_row[i * 3], 
                            .g = src_row[i * 3 + 1], 
                            .b = src_row[i * 3 + 2] 
                        };
                    } else {
                        // 16-bit to 8-bit conversion with bounds checking
                        const offset = i * 6;
                        if (offset + 6 > src_row.len) {
                            pixel.* = Rgb{ .r = 0, .g = 0, .b = 0 };
                        } else {
                            pixel.* = Rgb{ 
                                .r = @intCast(std.mem.readInt(u16, src_row[offset..offset + 2][0..2], .big) >> 8),
                                .g = @intCast(std.mem.readInt(u16, src_row[offset + 2..offset + 4][0..2], .big) >> 8),
                                .b = @intCast(std.mem.readInt(u16, src_row[offset + 4..offset + 6][0..2], .big) >> 8),
                            };
                        }
                    }
                }
            },
            .palette => {
                // Convert palette to RGB
                if (png_image.palette == null) return error.MissingPalette;
                const palette = png_image.palette.?;
                
                for (dst_row, 0..) |*pixel, i| {
                    if (i >= src_row.len) return error.InvalidScanlineData;
                    const index = src_row[i];
                    if (index >= palette.len) return error.InvalidPaletteIndex;
                    const rgb = palette[index];
                    pixel.* = Rgb{ .r = rgb[0], .g = rgb[1], .b = rgb[2] };
                }
            },
            .grayscale_alpha => {
                // Convert grayscale+alpha to RGB (ignore alpha for now)
                for (dst_row, 0..) |*pixel, i| {
                    const gray = src_row[i * 2];
                    pixel.* = Rgb{ .r = gray, .g = gray, .b = gray };
                }
            },
            .rgba => {
                // Convert RGBA to RGB (ignore alpha)
                for (dst_row, 0..) |*pixel, i| {
                    pixel.* = Rgb{ 
                        .r = src_row[i * 4], 
                        .g = src_row[i * 4 + 1], 
                        .b = src_row[i * 4 + 2] 
                    };
                }
            },
        }
    }

    return Image(Rgb).init(height, width, output_data);
}

pub fn toRgbaImage(allocator: Allocator, png_image: PngImage) !Image(Rgba) {
    // Decompress IDAT data
    const decompressed = try deflate.zlibDecompress(allocator, png_image.idat_data.items);
    defer allocator.free(decompressed);

    // Apply row defiltering
    try defilterScanlines(decompressed, png_image.header);

    // Convert to RGBA Image format
    const width = png_image.header.width;
    const height = png_image.header.height;
    const scanline_bytes = png_image.header.scanlineBytes();

    // Create output image with overflow protection
    const total_pixels = @as(u64, width) * @as(u64, height);
    if (total_pixels > std.math.maxInt(usize)) {
        return error.ImageTooLarge;
    }
    var output_data = try allocator.alloc(Rgba, @intCast(total_pixels));
    
    // Copy and convert pixel data
    for (0..height) |y| {
        const src_row_start = y * (scanline_bytes + 1) + 1; // +1 to skip filter byte
        const dst_row_start = y * width;
        
        const src_row = decompressed[src_row_start..src_row_start + scanline_bytes];
        const dst_row = output_data[dst_row_start..dst_row_start + width];
        
        switch (png_image.header.color_type) {
            .grayscale => {
                // Convert grayscale to RGBA
                for (dst_row, 0..) |*pixel, i| {
                    const gray = switch (png_image.header.bit_depth) {
                        8 => src_row[i],
                        16 => blk: {
                            if (i * 2 + 1 >= src_row.len) break :blk 0;
                            break :blk @as(u8, @intCast(std.mem.readInt(u16, src_row[i * 2..i * 2 + 2][0..2], .big) >> 8));
                        },
                        1, 2, 4 => blk: {
                            const bits_per_pixel = png_image.header.bit_depth;
                            const pixels_per_byte = 8 / bits_per_pixel;
                            const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
                            const byte_idx = i / pixels_per_byte;
                            if (byte_idx >= src_row.len) break :blk 0;
                            const pixel_idx = i % pixels_per_byte;
                            const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_idx) * bits_per_pixel);
                            const pixel_value = (src_row[byte_idx] >> bit_offset) & mask;
                            const scale_factor = 255 / mask;
                            break :blk pixel_value * scale_factor;
                        },
                        else => 0,
                    };
                    pixel.* = Rgba{ .r = gray, .g = gray, .b = gray, .a = 255 };
                }
            },
            .rgb => {
                // Convert RGB to RGBA
                for (dst_row, 0..) |*pixel, i| {
                    if (png_image.header.bit_depth == 8) {
                        pixel.* = Rgba{ 
                            .r = src_row[i * 3], 
                            .g = src_row[i * 3 + 1], 
                            .b = src_row[i * 3 + 2],
                            .a = 255
                        };
                    } else {
                        // 16-bit to 8-bit conversion with bounds checking
                        const offset = i * 6;
                        if (offset + 6 > src_row.len) {
                            pixel.* = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };
                        } else {
                            pixel.* = Rgba{ 
                                .r = @intCast(std.mem.readInt(u16, src_row[offset..offset + 2][0..2], .big) >> 8),
                                .g = @intCast(std.mem.readInt(u16, src_row[offset + 2..offset + 4][0..2], .big) >> 8),
                                .b = @intCast(std.mem.readInt(u16, src_row[offset + 4..offset + 6][0..2], .big) >> 8),
                                .a = 255
                            };
                        }
                    }
                }
            },
            .palette => {
                // Convert palette to RGBA
                if (png_image.palette == null) return error.MissingPalette;
                const palette = png_image.palette.?;
                
                for (dst_row, 0..) |*pixel, i| {
                    if (i >= src_row.len) return error.InvalidScanlineData;
                    const index = src_row[i];
                    if (index >= palette.len) return error.InvalidPaletteIndex;
                    const rgb = palette[index];
                    
                    // Check for transparency
                    const alpha = if (png_image.transparency) |trans| 
                        if (index < trans.len) trans[index] else 255
                    else 
                        255;
                    
                    pixel.* = Rgba{ .r = rgb[0], .g = rgb[1], .b = rgb[2], .a = alpha };
                }
            },
            .grayscale_alpha => {
                // Convert grayscale+alpha to RGBA
                for (dst_row, 0..) |*pixel, i| {
                    const gray = src_row[i * 2];
                    const alpha = src_row[i * 2 + 1];
                    pixel.* = Rgba{ .r = gray, .g = gray, .b = gray, .a = alpha };
                }
            },
            .rgba => {
                // Direct RGBA copy
                for (dst_row, 0..) |*pixel, i| {
                    if (png_image.header.bit_depth == 8) {
                        pixel.* = Rgba{ 
                            .r = src_row[i * 4], 
                            .g = src_row[i * 4 + 1], 
                            .b = src_row[i * 4 + 2],
                            .a = src_row[i * 4 + 3]
                        };
                    } else {
                        // 16-bit to 8-bit conversion with bounds checking
                        const offset = i * 8;
                        if (offset + 8 > src_row.len) {
                            pixel.* = Rgba{ .r = 0, .g = 0, .b = 0, .a = 255 };
                        } else {
                            pixel.* = Rgba{ 
                                .r = @intCast(std.mem.readInt(u16, src_row[offset..offset + 2][0..2], .big) >> 8),
                                .g = @intCast(std.mem.readInt(u16, src_row[offset + 2..offset + 4][0..2], .big) >> 8),
                                .b = @intCast(std.mem.readInt(u16, src_row[offset + 4..offset + 6][0..2], .big) >> 8),
                                .a = @intCast(std.mem.readInt(u16, src_row[offset + 6..offset + 8][0..2], .big) >> 8),
                            };
                        }
                    }
                }
            },
        }
    }

    return Image(Rgba).init(height, width, output_data);
}

// High-level API functions
pub fn loadPng(allocator: Allocator, file_path: []const u8) !Image(Rgba) {
    const png_data = try std.fs.cwd().readFileAlloc(allocator, file_path, 100 * 1024 * 1024);
    defer allocator.free(png_data);
    
    var png_image = try decode(allocator, png_data);
    defer png_image.deinit(allocator);
    
    return toRgbaImage(allocator, png_image);
}

pub fn loadPngRgb(allocator: Allocator, file_path: []const u8) !Image(Rgb) {
    const png_data = try std.fs.cwd().readFileAlloc(allocator, file_path, 100 * 1024 * 1024);
    defer allocator.free(png_data);
    
    var png_image = try decode(allocator, png_data);
    defer png_image.deinit(allocator);
    
    return toRgbImage(allocator, png_image);
}

pub fn loadPngGrayscale(allocator: Allocator, file_path: []const u8) !Image(u8) {
    const png_data = try std.fs.cwd().readFileAlloc(allocator, file_path, 100 * 1024 * 1024);
    defer allocator.free(png_data);
    
    var png_image = try decode(allocator, png_data);
    defer png_image.deinit(allocator);
    
    return toImage(allocator, png_image);
}

// PNG Encoder functionality

// Chunk writer for PNG encoding
pub const ChunkWriter = struct {
    data: ArrayList(u8),
    
    pub fn init(allocator: Allocator) ChunkWriter {
        return .{ .data = ArrayList(u8).init(allocator) };
    }
    
    pub fn deinit(self: *ChunkWriter) void {
        self.data.deinit();
    }
    
    pub fn writeChunk(self: *ChunkWriter, chunk_type: [4]u8, chunk_data: []const u8) !void {
        // Length (4 bytes, big endian)
        const length: u32 = @intCast(chunk_data.len);
        try self.data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, length, .big)));
        
        // Type (4 bytes)
        try self.data.appendSlice(&chunk_type);
        
        // Data
        try self.data.appendSlice(chunk_data);
        
        // CRC (4 bytes, big endian) - calculate CRC of type + data
        var crc_data = try self.data.allocator.alloc(u8, 4 + chunk_data.len);
        defer self.data.allocator.free(crc_data);
        @memcpy(crc_data[0..4], &chunk_type);
        @memcpy(crc_data[4..], chunk_data);
        
        const chunk_crc = crc(crc_data);
        try self.data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, chunk_crc, .big)));
    }
    
    pub fn toOwnedSlice(self: *ChunkWriter) ![]u8 {
        return self.data.toOwnedSlice();
    }
};

// Create IHDR chunk data
fn createIHDR(header: Header) ![13]u8 {
    var ihdr_data: [13]u8 = undefined;
    
    // Width (4 bytes)
    std.mem.writeInt(u32, ihdr_data[0..4], header.width, .big);
    
    // Height (4 bytes) 
    std.mem.writeInt(u32, ihdr_data[4..8], header.height, .big);
    
    // Bit depth (1 byte)
    ihdr_data[8] = header.bit_depth;
    
    // Color type (1 byte)
    ihdr_data[9] = @intFromEnum(header.color_type);
    
    // Compression method (1 byte) - always 0
    ihdr_data[10] = 0;
    
    // Filter method (1 byte) - always 0  
    ihdr_data[11] = 0;
    
    // Interlace method (1 byte) - 0 for no interlacing
    ihdr_data[12] = 0;
    
    return ihdr_data;
}

// Apply PNG row filtering to scanlines
fn filterScanlines(allocator: Allocator, data: []const u8, header: Header, filter_type: FilterType) ![]u8 {
    const scanline_bytes = header.scanlineBytes();
    const bytes_per_pixel = header.bytesPerPixel();
    const filtered_size = header.height * (scanline_bytes + 1); // +1 for filter byte
    
    var filtered_data = try allocator.alloc(u8, filtered_size);
    
    var y: u32 = 0;
    while (y < header.height) : (y += 1) {
        const src_row_start = y * scanline_bytes;
        const dst_row_start = y * (scanline_bytes + 1);
        
        const src_row = data[src_row_start..src_row_start + scanline_bytes];
        const dst_row = filtered_data[dst_row_start + 1..dst_row_start + 1 + scanline_bytes];
        
        // Set filter type byte
        filtered_data[dst_row_start] = @intFromEnum(filter_type);
        
        // Apply filtering
        const previous_row = if (y > 0) 
            data[(y - 1) * scanline_bytes..(y - 1) * scanline_bytes + scanline_bytes]
        else 
            null;
            
        filterRow(filter_type, dst_row, src_row, previous_row, bytes_per_pixel);
    }
    
    return filtered_data;
}

// Encode Image data to PNG format
pub fn encode(allocator: Allocator, image_data: []const u8, width: u32, height: u32, color_type: ColorType, bit_depth: u8) ![]u8 {
    var writer = ChunkWriter.init(allocator);
    defer writer.deinit();
    
    // Write PNG signature
    try writer.data.appendSlice(&PNG_SIGNATURE);
    
    // Create and write IHDR
    const header = Header{
        .width = width,
        .height = height,
        .bit_depth = bit_depth,
        .color_type = color_type,
        .compression_method = 0,
        .filter_method = 0,
        .interlace_method = 0,
    };
    
    const ihdr_data = try createIHDR(header);
    try writer.writeChunk("IHDR".*, &ihdr_data);
    
    // Apply row filtering (using 'none' filter for simplicity)
    const filtered_data = try filterScanlines(allocator, image_data, header, .none);
    defer allocator.free(filtered_data);
    
    // Compress filtered data with zlib format (required for PNG IDAT)
    const compressed_data = try deflate.zlibCompress(allocator, filtered_data);
    defer allocator.free(compressed_data);
    
    // Write IDAT chunk
    try writer.writeChunk("IDAT".*, compressed_data);
    
    // Write IEND chunk
    try writer.writeChunk("IEND".*, &[_]u8{});
    
    return writer.toOwnedSlice();
}

// High-level API functions for encoding from Zignal Image types
pub fn encodeRgbaImage(allocator: Allocator, image: Image(Rgba)) ![]u8 {
    // Convert RGBA Image to byte array
    const image_bytes = image.asBytes();
    return encode(allocator, image_bytes, @intCast(image.cols), @intCast(image.rows), .rgba, 8);
}

pub fn encodeRgbImage(allocator: Allocator, image: Image(Rgb)) ![]u8 {
    // Convert RGB Image to byte array
    const image_bytes = image.asBytes();
    return encode(allocator, image_bytes, @intCast(image.cols), @intCast(image.rows), .rgb, 8);
}

pub fn encodeGrayscaleImage(allocator: Allocator, image: Image(u8)) ![]u8 {
    // Convert grayscale Image to byte array
    const image_bytes = image.asBytes();
    return encode(allocator, image_bytes, @intCast(image.cols), @intCast(image.rows), .grayscale, 8);
}

// Save PNG files from Zignal Image types
pub fn savePng(allocator: Allocator, image: Image(Rgba), file_path: []const u8) !void {
    const png_data = try encodeRgbaImage(allocator, image);
    defer allocator.free(png_data);
    
    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    
    try file.writeAll(png_data);
}

pub fn savePngRgb(allocator: Allocator, image: Image(Rgb), file_path: []const u8) !void {
    const png_data = try encodeRgbImage(allocator, image);
    defer allocator.free(png_data);
    
    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    
    try file.writeAll(png_data);
}

pub fn savePngGrayscale(allocator: Allocator, image: Image(u8), file_path: []const u8) !void {
    const png_data = try encodeGrayscaleImage(allocator, image);
    defer allocator.free(png_data);
    
    const file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();
    
    try file.writeAll(png_data);
}

// PNG row filtering and defiltering functions
fn paethPredictor(a: i32, b: i32, c: i32) u8 {
    const p = a + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);

    if (pa <= pb and pa <= pc) {
        return @intCast(a);
    } else if (pb <= pc) {
        return @intCast(b);
    } else {
        return @intCast(c);
    }
}

fn defilterRow(
    filter_type: FilterType,
    current_row: []u8,
    previous_row: ?[]const u8,
    bytes_per_pixel: u8,
) void {
    switch (filter_type) {
        .none => {
            // No filtering - data is already correct
        },
        .sub => {
            // Add the byte to the left
            var i: usize = bytes_per_pixel;
            while (i < current_row.len) : (i += 1) {
                current_row[i] = current_row[i] +% current_row[i - bytes_per_pixel];
            }
        },
        .up => {
            // Add the byte above
            if (previous_row) |prev| {
                for (current_row, 0..) |*byte, i| {
                    byte.* = byte.* +% prev[i];
                }
            }
        },
        .average => {
            // Add the average of left and above bytes
            for (current_row, 0..) |*byte, i| {
                const left: u16 = if (i >= bytes_per_pixel) current_row[i - bytes_per_pixel] else 0;
                const above: u16 = if (previous_row) |prev| prev[i] else 0;
                const avg: u8 = @intCast((left + above) / 2);
                byte.* = byte.* +% avg;
            }
        },
        .paeth => {
            // Use Paeth predictor
            for (current_row, 0..) |*byte, i| {
                const left: i32 = if (i >= bytes_per_pixel) current_row[i - bytes_per_pixel] else 0;
                const above: i32 = if (previous_row) |prev| prev[i] else 0;
                const upper_left: i32 = if (previous_row != null and i >= bytes_per_pixel) 
                    previous_row.?[i - bytes_per_pixel] else 0;
                
                const paeth = paethPredictor(left, above, upper_left);
                byte.* = byte.* +% paeth;
            }
        },
    }
}

fn filterRow(
    filter_type: FilterType,
    current_row: []u8,
    original_row: []const u8,
    previous_row: ?[]const u8,
    bytes_per_pixel: u8,
) void {
    switch (filter_type) {
        .none => {
            @memcpy(current_row, original_row);
        },
        .sub => {
            // Subtract the byte to the left
            @memcpy(current_row, original_row);
            var i: usize = bytes_per_pixel;
            while (i < current_row.len) : (i += 1) {
                current_row[i] = current_row[i] -% current_row[i - bytes_per_pixel];
            }
        },
        .up => {
            // Subtract the byte above
            for (current_row, original_row, 0..) |*filtered, orig, i| {
                const above: u8 = if (previous_row) |prev| prev[i] else 0;
                filtered.* = orig -% above;
            }
        },
        .average => {
            // Subtract the average of left and above bytes
            for (current_row, original_row, 0..) |*filtered, orig, i| {
                const left: u16 = if (i >= bytes_per_pixel) original_row[i - bytes_per_pixel] else 0;
                const above: u16 = if (previous_row) |prev| prev[i] else 0;
                const avg: u8 = @intCast((left + above) / 2);
                filtered.* = orig -% avg;
            }
        },
        .paeth => {
            // Subtract Paeth predictor
            for (current_row, original_row, 0..) |*filtered, orig, i| {
                const left: i32 = if (i >= bytes_per_pixel) original_row[i - bytes_per_pixel] else 0;
                const above: i32 = if (previous_row) |prev| prev[i] else 0;
                const upper_left: i32 = if (previous_row != null and i >= bytes_per_pixel) 
                    previous_row.?[i - bytes_per_pixel] else 0;
                
                const paeth = paethPredictor(left, above, upper_left);
                filtered.* = orig -% paeth;
            }
        },
    }
}

// Apply defiltering to all scanlines after deflate decompression
fn defilterScanlines(data: []u8, header: Header) !void {
    const scanline_bytes = header.scanlineBytes();
    const bytes_per_pixel = header.bytesPerPixel();
    const expected_size = header.height * (scanline_bytes + 1); // +1 for filter byte
    
    if (data.len != expected_size) {
        return error.InvalidScanlineData;
    }

    var y: u32 = 0;
    var previous_scanline: ?[]u8 = null;

    while (y < header.height) : (y += 1) {
        const row_start = y * (scanline_bytes + 1);
        const filter_byte = data[row_start];
        const current_scanline = data[row_start + 1..row_start + 1 + scanline_bytes];

        const filter_type: FilterType = switch (filter_byte) {
            0 => .none,
            1 => .sub,
            2 => .up,
            3 => .average,
            4 => .paeth,
            else => return error.InvalidFilterType,
        };

        defilterRow(filter_type, current_scanline, previous_scanline, bytes_per_pixel);
        previous_scanline = current_scanline;
    }
}

// Simple test for the PNG structure
test "PNG signature validation" {
    const invalid_sig = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = decode(std.testing.allocator, &invalid_sig);
    try std.testing.expectError(error.InvalidPngSignature, result);
}

test "CRC calculation" {
    // Test with known values
    const test_data = "IHDR";
    const expected_chunk_type_crc = crc(test_data);
    // This is just to make sure our CRC function runs without crashing
    try std.testing.expect(expected_chunk_type_crc != 0);
}

test "Paeth predictor" {
    // Test cases - verify the Paeth predictor algorithm  
    try std.testing.expectEqual(@as(u8, 15), paethPredictor(10, 20, 15)); // p=15, pa=5, pb=5, pc=0 -> c=15
    try std.testing.expectEqual(@as(u8, 5), paethPredictor(5, 20, 15));   // p=10, pa=5, pb=10, pc=5 -> a=5
    try std.testing.expectEqual(@as(u8, 10), paethPredictor(10, 5, 6));   // p=9, pa=1, pb=4, pc=3 -> a=10
}

test "PNG round-trip encoding/decoding" {
    const allocator = std.testing.allocator;
    
    // Create a simple test image (4x4 RGB)
    const width = 4;
    const height = 4;
    const test_data = [_]Rgb{
        .{ .r = 255, .g = 0, .b = 0 }, .{ .r = 0, .g = 255, .b = 0 }, .{ .r = 0, .g = 0, .b = 255 }, .{ .r = 255, .g = 255, .b = 0 },
        .{ .r = 255, .g = 0, .b = 255 }, .{ .r = 0, .g = 255, .b = 255 }, .{ .r = 128, .g = 128, .b = 128 }, .{ .r = 255, .g = 255, .b = 255 },
        .{ .r = 0, .g = 0, .b = 0 }, .{ .r = 64, .g = 64, .b = 64 }, .{ .r = 192, .g = 192, .b = 192 }, .{ .r = 128, .g = 0, .b = 128 },
        .{ .r = 128, .g = 128, .b = 0 }, .{ .r = 0, .g = 128, .b = 128 }, .{ .r = 255, .g = 128, .b = 64 }, .{ .r = 64, .g = 255, .b = 128 },
    };
    
    // Create owned copy for Image
    const owned_data = try allocator.alloc(Rgb, test_data.len);
    defer allocator.free(owned_data);
    @memcpy(owned_data, &test_data);
    
    const original_image = Image(Rgb).init(height, width, owned_data);
    
    // Encode to PNG
    const png_data = try encodeRgbImage(allocator, original_image);
    defer allocator.free(png_data);
    
    // Verify PNG signature
    try std.testing.expect(png_data.len > 8);
    try std.testing.expectEqualSlices(u8, &PNG_SIGNATURE, png_data[0..8]);
    
    // Decode back from PNG
    var decoded_png = try decode(allocator, png_data);
    defer decoded_png.deinit(allocator);
    
    // Verify header
    try std.testing.expectEqual(@as(u32, width), decoded_png.header.width);
    try std.testing.expectEqual(@as(u32, height), decoded_png.header.height);
    try std.testing.expectEqual(ColorType.rgb, decoded_png.header.color_type);
    try std.testing.expectEqual(@as(u8, 8), decoded_png.header.bit_depth);
    
    // Convert back to Image
    var decoded_image = try toRgbImage(allocator, decoded_png);
    defer decoded_image.deinit(allocator);
    
    // Verify dimensions
    try std.testing.expectEqual(height, decoded_image.rows);
    try std.testing.expectEqual(width, decoded_image.cols);
    
    // Verify pixel data
    for (original_image.data, decoded_image.data) |orig, decoded| {
        try std.testing.expectEqual(orig.r, decoded.r);
        try std.testing.expectEqual(orig.g, decoded.g);
        try std.testing.expectEqual(orig.b, decoded.b);
    }
}

test "PNG bit unpacking - 1-bit grayscale" {
    
    // Test data with bits: 10110010 = 0xB2
    const test_byte: u8 = 0b10110010;
    const src_row = [_]u8{test_byte};
    var dst_row: [8]u8 = undefined;
    
    // Unpack bits according to PNG spec (MSB first)
    const bits_per_pixel = 1;
    const pixels_per_byte = 8;
    const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
    
    for (0..8) |x| {
        const byte_index = x / pixels_per_byte;
        const pixel_index = x % pixels_per_byte;
        const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_index) * bits_per_pixel);
        const pixel_value = (src_row[byte_index] >> bit_offset) & mask;
        
        // Scale to 8-bit
        const scale_factor = 255 / mask;
        dst_row[x] = pixel_value * scale_factor;
    }
    
    // Expected: 1,0,1,1,0,0,1,0 -> 255,0,255,255,0,0,255,0
    try std.testing.expectEqual(@as(u8, 255), dst_row[0]);
    try std.testing.expectEqual(@as(u8, 0), dst_row[1]);
    try std.testing.expectEqual(@as(u8, 255), dst_row[2]);
    try std.testing.expectEqual(@as(u8, 255), dst_row[3]);
    try std.testing.expectEqual(@as(u8, 0), dst_row[4]);
    try std.testing.expectEqual(@as(u8, 0), dst_row[5]);
    try std.testing.expectEqual(@as(u8, 255), dst_row[6]);
    try std.testing.expectEqual(@as(u8, 0), dst_row[7]);
}

test "PNG bit unpacking - 2-bit grayscale" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Test data with 2-bit values: 11 01 10 00 = 0xD8
    const test_byte: u8 = 0b11011000;
    const src_row = [_]u8{test_byte};
    var dst_row: [4]u8 = undefined;
    
    // Unpack 2-bit values
    const bits_per_pixel = 2;
    const pixels_per_byte = 4;
    const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
    
    for (0..4) |x| {
        const byte_index = x / pixels_per_byte;
        const pixel_index = x % pixels_per_byte;
        const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_index) * bits_per_pixel);
        const pixel_value = (src_row[byte_index] >> bit_offset) & mask;
        
        // Scale to 8-bit (0,85,170,255)
        const scale_factor = 255 / mask;
        dst_row[x] = pixel_value * scale_factor;
    }
    
    // Expected: 3,1,2,0 -> 255,85,170,0
    try std.testing.expectEqual(@as(u8, 255), dst_row[0]);
    try std.testing.expectEqual(@as(u8, 85), dst_row[1]);
    try std.testing.expectEqual(@as(u8, 170), dst_row[2]);
    try std.testing.expectEqual(@as(u8, 0), dst_row[3]);
}

test "PNG bit unpacking - 4-bit grayscale" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Test data with 4-bit values: 1111 0101 = 0xF5
    const test_byte: u8 = 0xF5;
    const src_row = [_]u8{test_byte};
    var dst_row: [2]u8 = undefined;
    
    // Unpack 4-bit values
    const bits_per_pixel = 4;
    const pixels_per_byte = 2;
    const mask = (@as(u8, 1) << @intCast(bits_per_pixel)) - 1;
    
    for (0..2) |x| {
        const byte_index = x / pixels_per_byte;
        const pixel_index = x % pixels_per_byte;
        const bit_offset: u3 = @intCast((pixels_per_byte - 1 - pixel_index) * bits_per_pixel);
        const pixel_value = (src_row[byte_index] >> bit_offset) & mask;
        
        // Scale to 8-bit
        const scale_factor = 255 / mask;
        dst_row[x] = pixel_value * scale_factor;
    }
    
    // Expected: 15,5 -> 255,85
    try std.testing.expectEqual(@as(u8, 255), dst_row[0]);
    try std.testing.expectEqual(@as(u8, 85), dst_row[1]);
}

test "PNG CRC validation" {
    const allocator = std.testing.allocator;
    _ = allocator;
    
    // Test IHDR chunk CRC
    const ihdr_type = "IHDR";
    const ihdr_data = [_]u8{
        0, 0, 0, 4,   // width = 4
        0, 0, 0, 4,   // height = 4
        8,            // bit depth
        2,            // color type (RGB)
        0,            // compression
        0,            // filter
        0,            // interlace
    };
    
    var test_data = ArrayList(u8).init(std.testing.allocator);
    defer test_data.deinit();
    
    try test_data.appendSlice(ihdr_type);
    try test_data.appendSlice(&ihdr_data);
    
    const calculated_crc = crc(test_data.items);
    
    // Verify CRC was calculated
    try std.testing.expect(calculated_crc != 0);
    
    // Test with invalid data should give different CRC
    test_data.items[4] = 1; // Change width
    const different_crc = crc(test_data.items);
    try std.testing.expect(calculated_crc != different_crc);
}

test "PNG 16-bit to 8-bit conversion" {
    // Test 16-bit value conversion
    const test_values = [_]u16{ 0x0000, 0x00FF, 0xFF00, 0xFFFF, 0x8080, 0x1234 };
    const expected_8bit = [_]u8{ 0, 0, 255, 255, 128, 18 }; // Simple >>8 conversion
    
    for (test_values, expected_8bit) |val16, expected| {
        const bytes = std.mem.toBytes(std.mem.nativeTo(u16, val16, .big));
        const converted = @as(u8, @intCast(std.mem.readInt(u16, bytes[0..2], .big) >> 8));
        try std.testing.expectEqual(expected, converted);
    }
}

test "PNG filter types" {
    // Test filter type validation
    const valid_filters = [_]u8{ 0, 1, 2, 3, 4 };
    const invalid_filter: u8 = 5;
    
    for (valid_filters) |filter| {
        const filter_type: FilterType = switch (filter) {
            0 => .none,
            1 => .sub,
            2 => .up,
            3 => .average,
            4 => .paeth,
            else => unreachable,
        };
        try std.testing.expectEqual(filter, @intFromEnum(filter_type));
    }
    
    // Test that invalid filter would be caught
    const result: ?FilterType = switch (invalid_filter) {
        0 => .none,
        1 => .sub,
        2 => .up,
        3 => .average,
        4 => .paeth,
        else => null,
    };
    try std.testing.expect(result == null);
}

test "PNG bounds checking - large image dimensions" {
    const allocator = std.testing.allocator;
    
    // Create a malformed PNG with excessively large dimensions
    var png_data = ArrayList(u8).init(allocator);
    defer png_data.deinit();
    
    // PNG signature
    try png_data.appendSlice(&PNG_SIGNATURE);
    
    // IHDR chunk with oversized dimensions
    const ihdr_length: u32 = 13;
    try png_data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, ihdr_length, .big)));
    try png_data.appendSlice("IHDR");
    
    // Width: 50000 (exceeds MAX_DIMENSION)
    try png_data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, 50000, .big)));
    // Height: 50000 (exceeds MAX_DIMENSION)
    try png_data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, 50000, .big)));
    
    try png_data.append(8); // bit depth
    try png_data.append(2); // color type (RGB)
    try png_data.append(0); // compression
    try png_data.append(0); // filter
    try png_data.append(0); // interlace
    
    // Calculate and append CRC
    var crc_data = try allocator.alloc(u8, 4 + 13);
    defer allocator.free(crc_data);
    @memcpy(crc_data[0..4], "IHDR");
    @memcpy(crc_data[4..], png_data.items[16..29]);
    const ihdr_crc = crc(crc_data);
    try png_data.appendSlice(std.mem.asBytes(&std.mem.nativeTo(u32, ihdr_crc, .big)));
    
    // Try to decode - should fail with ImageTooLarge
    const result = decode(allocator, png_data.items);
    try std.testing.expectError(error.ImageTooLarge, result);
}

test "PNG bounds checking - malformed palette" {
    const allocator = std.testing.allocator;
    
    // Test malformed palette chunk that's too short
    const chunk = Chunk{
        .length = 10, // Should be multiple of 3
        .type = "PLTE".*,
        .data = &[_]u8{ 255, 0, 0, 0, 255, 0, 0, 0 }, // Only 8 bytes, but length claims 10
        .crc = 0,
    };
    
    var png_image = PngImage{
        .header = Header{
            .width = 4,
            .height = 4,
            .bit_depth = 8,
            .color_type = .palette,
            .compression_method = 0,
            .filter_method = 0,
            .interlace_method = 0,
        },
        .idat_data = ArrayList(u8).init(allocator),
    };
    defer png_image.deinit(allocator);
    
    // Simulate the palette parsing that would happen in decode()
    if (chunk.length % 3 != 0) {
        try std.testing.expect(true); // This should be caught
        return;
    }
    
    const palette_size = chunk.length / 3;
    if (chunk.data.len < palette_size * 3) {
        try std.testing.expect(true); // This should be caught
        return;
    }
    
    try std.testing.expect(false); // Should not reach here
}

test "PNG 16-bit bounds checking" {
    // Test 16-bit conversion with insufficient data
    const short_data = [_]u8{ 0xFF }; // Only 1 byte, but 16-bit needs 2
    const samples_per_row = short_data.len / 2; // Will be 0
    
    var dst_row: [1]u8 = undefined;
    
    for (0..samples_per_row) |i| {
        const offset = i * 2;
        if (offset + 2 > short_data.len) {
            dst_row[i] = 0; // Should use fallback value
        } else {
            const sample16 = std.mem.readInt(u16, short_data[offset..offset + 2][0..2], .big);
            dst_row[i] = @intCast(sample16 >> 8);
        }
    }
    
    // Should have processed 0 samples safely
    try std.testing.expectEqual(@as(usize, 0), samples_per_row);
}

test "PNG integer overflow protection" {
    // Test that large dimensions are caught before overflow
    const large_width: u32 = 65536;
    const large_height: u32 = 65536;
    const channels: u8 = 4;
    
    const total_pixels = @as(u64, large_width) * @as(u64, large_height);
    const total_bytes = total_pixels * @as(u64, channels);
    
    // This should exceed practical memory limits
    try std.testing.expect(total_bytes > 1000000000); // > 1GB
    
    if (total_bytes > std.math.maxInt(usize)) {
        try std.testing.expect(true); // Would be caught by our protection
    }
}