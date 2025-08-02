//! PCF (Portable Compiled Format) font parser for zignal
//!
//! This module provides zero-dependency parsing of PCF font files,
//! a binary format used by X11 for efficient bitmap font storage.

const std = @import("std");
const testing = std.testing;
const BitmapFont = @import("BitmapFont.zig");
const GlyphData = @import("GlyphData.zig");
const unicode = @import("unicode.zig");
const LoadFilter = @import("../font.zig").LoadFilter;

/// Errors that can occur during PCF parsing
pub const PcfError = error{
    InvalidFormat,
    InvalidVersion,
    MissingRequired,
    InvalidTableEntry,
    InvalidBitmapData,
    AllocationFailed,
    UnsupportedFormat,
    InvalidCompression,
};

/// PCF format constants
const PCF_FILE_VERSION = 0x70636601; // "\x01fcp" in little-endian
const PCF_FORMAT_MASK = 0xffffff00;

/// PCF table types
const PCF_PROPERTIES = (1 << 0);
const PCF_ACCELERATORS = (1 << 1);
const PCF_METRICS = (1 << 2);
const PCF_BITMAPS = (1 << 3);
const PCF_INK_METRICS = (1 << 4);
const PCF_BDF_ENCODINGS = (1 << 5);
const PCF_SWIDTHS = (1 << 6);
const PCF_GLYPH_NAMES = (1 << 7);
const PCF_BDF_ACCELERATORS = (1 << 8);

/// PCF format flags
const PCF_DEFAULT_FORMAT = 0x00000000;
const PCF_INKBOUNDS = 0x00000200;
const PCF_ACCEL_W_INKBOUNDS = 0x00000100;
const PCF_COMPRESSED_METRICS = 0x00000100;

/// PCF byte order flags
const PCF_BIT_MASK = (1 << 2); // bit 2 for bit order within bytes
const PCF_BYTE_MASK = (1 << 3); // bit 3 for byte order (MSB/LSB)
const PCF_GLYPH_PAD_MASK = (3 << 0);
const PCF_SCAN_UNIT_MASK = (3 << 0);

/// Get byte order from format field
fn getByteOrder(format: u32) std.builtin.Endian {
    return if ((format & PCF_BYTE_MASK) != 0) .big else .little;
}

/// Table of contents entry
const TableEntry = struct {
    type: u32,
    format: u32,
    size: u32,
    offset: u32,
};

/// PCF metrics (compressed format)
const CompressedMetric = struct {
    left_sided_bearing: i8,
    right_sided_bearing: i8,
    character_width: i8,
    ascent: i8,
    descent: i8,
};

/// PCF metrics (uncompressed format)
const UncompressedMetric = struct {
    left_sided_bearing: i16,
    right_sided_bearing: i16,
    character_width: i16,
    ascent: i16,
    descent: i16,
    attributes: u16,
};

/// PCF accelerator table
const Accelerator = struct {
    no_overlap: bool,
    constant_metrics: bool,
    terminal_font: bool,
    constant_width: bool,
    ink_inside: bool,
    ink_metrics: bool,
    draw_direction: bool,
    font_ascent: i32,
    font_descent: i32,
    max_overlap: i32,
    min_bounds: UncompressedMetric,
    max_bounds: UncompressedMetric,
    ink_min_bounds: ?UncompressedMetric,
    ink_max_bounds: ?UncompressedMetric,
};

/// PCF encoding entry
const EncodingEntry = struct {
    min_char_or_byte2: u16,
    max_char_or_byte2: u16,
    min_byte1: u16,
    max_byte1: u16,
    default_char: u16,
    glyph_indices: []u16,
};

/// Load a PCF font from a file path
/// Parameters:
/// - allocator: Memory allocator
/// - path: Path to PCF file
/// - filter: Filter for which characters to load
pub fn load(allocator: std.mem.Allocator, path: []const u8, filter: LoadFilter) !BitmapFont {
    // Check if file is gzip compressed
    const is_compressed = std.mem.endsWith(u8, path, ".gz");

    if (is_compressed) {
        // TODO: Handle gzip decompression using deflate module
        return PcfError.InvalidCompression;
    }

    // Read entire file into memory
    const file_contents = try std.fs.cwd().readFileAlloc(allocator, path, 50 * 1024 * 1024); // 50MB max
    defer allocator.free(file_contents);

    // Use arena for temporary allocations
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // Parse PCF file
    var stream = std.io.fixedBufferStream(file_contents);
    const reader = stream.reader();

    // Read and verify header
    const header = try reader.readInt(u32, .little);
    if (header != PCF_FILE_VERSION) {
        return PcfError.InvalidFormat;
    }

    // Read table count
    const table_count = try reader.readInt(u32, .little);
    if (table_count == 0 or table_count > 1024) { // Sanity check
        return PcfError.InvalidFormat;
    }

    // Read table of contents
    const tables = try arena_allocator.alloc(TableEntry, table_count);
    for (tables) |*table| {
        table.type = try reader.readInt(u32, .little);
        table.format = try reader.readInt(u32, .little);
        table.size = try reader.readInt(u32, .little);
        table.offset = try reader.readInt(u32, .little);
    }

    // Find required tables
    const metrics_table = findTable(tables, PCF_METRICS) orelse return PcfError.MissingRequired;
    const bitmaps_table = findTable(tables, PCF_BITMAPS) orelse return PcfError.MissingRequired;
    const encodings_table = findTable(tables, PCF_BDF_ENCODINGS) orelse return PcfError.MissingRequired;
    const accel_table = findTable(tables, PCF_ACCELERATORS) orelse findTable(tables, PCF_BDF_ACCELERATORS);

    // Parse accelerator table for font metrics
    var font_ascent: i16 = 0;
    var font_descent: i16 = 0;
    var max_width: u16 = 0;
    var max_height: u16 = 0;

    if (accel_table) |accel| {
        const accel_data = try parseAccelerator(file_contents, accel);
        font_ascent = @as(i16, @intCast(@min(accel_data.font_ascent, std.math.maxInt(i16))));
        font_descent = @as(i16, @intCast(@min(accel_data.font_descent, std.math.maxInt(i16))));
        max_width = @as(u16, @intCast(@min(@max(accel_data.max_bounds.character_width, 0), std.math.maxInt(u16))));
        const total_height = @max(0, accel_data.font_ascent) + @max(0, accel_data.font_descent);
        max_height = @as(u16, @intCast(@min(total_height, std.math.maxInt(u16))));
    } else {
        // Default values if no accelerator table
        font_ascent = 14;
        font_descent = 2;
        max_width = 16;
        max_height = 16;
    }

    // Parse encodings to get character mappings
    const encoding = try parseEncodings(arena_allocator, file_contents, encodings_table);

    // Parse metrics
    const metrics = try parseMetrics(arena_allocator, file_contents, metrics_table, encoding.glyph_indices.len);

    // Parse bitmap data
    const bitmap_info = try parseBitmaps(arena_allocator, file_contents, bitmaps_table);

    // Convert to BitmapFont format
    return convertToBitmapFont(allocator, metrics, bitmap_info, encoding, filter, font_ascent, max_width, max_height);
}

/// Find a table in the table of contents
fn findTable(tables: []const TableEntry, table_type: u32) ?TableEntry {
    for (tables) |table| {
        if (table.type == table_type) {
            return table;
        }
    }
    return null;
}

/// Parse accelerator table
fn parseAccelerator(data: []const u8, table: TableEntry) !Accelerator {
    if (table.offset + table.size > data.len) {
        return PcfError.InvalidTableEntry;
    }

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    var accel: Accelerator = undefined;

    // In PCF, we need to read fields in the correct order
    // The accelerator table has these fields after format:
    // 1. noOverlap (1 byte)
    // 2. constantMetrics (1 byte)
    // 3. terminalFont (1 byte)
    // 4. constantWidth (1 byte)
    // 5. inkInside (1 byte)
    // 6. inkMetrics (1 byte)
    // 7. drawDirection (1 byte)
    // 8. padding (1 byte)
    // 9. fontAscent (4 bytes)
    // 10. fontDescent (4 bytes)
    // etc.

    // Read the boolean flags as individual bytes
    accel.no_overlap = (try reader.readByte()) != 0;
    accel.constant_metrics = (try reader.readByte()) != 0;
    accel.terminal_font = (try reader.readByte()) != 0;
    accel.constant_width = (try reader.readByte()) != 0;
    accel.ink_inside = (try reader.readByte()) != 0;
    accel.ink_metrics = (try reader.readByte()) != 0;
    accel.draw_direction = (try reader.readByte()) != 0;
    _ = try reader.readByte(); // padding

    // Read font metrics
    accel.font_ascent = try reader.readInt(i32, byte_order);
    accel.font_descent = try reader.readInt(i32, byte_order);
    accel.max_overlap = try reader.readInt(i32, byte_order);

    // Read min bounds
    accel.min_bounds = try readUncompressedMetric(reader, byte_order);
    accel.max_bounds = try readUncompressedMetric(reader, byte_order);

    // Read ink bounds if present
    if ((table.format & PCF_ACCEL_W_INKBOUNDS) != 0) {
        accel.ink_min_bounds = try readUncompressedMetric(reader, byte_order);
        accel.ink_max_bounds = try readUncompressedMetric(reader, byte_order);
    } else {
        accel.ink_min_bounds = null;
        accel.ink_max_bounds = null;
    }

    return accel;
}

/// Read uncompressed metric
fn readUncompressedMetric(reader: anytype, byte_order: std.builtin.Endian) !UncompressedMetric {
    return UncompressedMetric{
        .left_sided_bearing = try reader.readInt(i16, byte_order),
        .right_sided_bearing = try reader.readInt(i16, byte_order),
        .character_width = try reader.readInt(i16, byte_order),
        .ascent = try reader.readInt(i16, byte_order),
        .descent = try reader.readInt(i16, byte_order),
        .attributes = try reader.readInt(u16, byte_order),
    };
}

/// Parse encodings table
fn parseEncodings(allocator: std.mem.Allocator, data: []const u8, table: TableEntry) !EncodingEntry {
    if (table.offset + table.size > data.len) {
        return PcfError.InvalidTableEntry;
    }

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    var encoding: EncodingEntry = undefined;

    // Read encoding info
    encoding.min_char_or_byte2 = try reader.readInt(u16, byte_order);
    encoding.max_char_or_byte2 = try reader.readInt(u16, byte_order);
    encoding.min_byte1 = try reader.readInt(u16, byte_order);
    encoding.max_byte1 = try reader.readInt(u16, byte_order);
    encoding.default_char = try reader.readInt(u16, byte_order);

    // Calculate total encodings with overflow protection
    const cols = @as(usize, encoding.max_char_or_byte2 - encoding.min_char_or_byte2 + 1);
    const rows = @as(usize, encoding.max_byte1 - encoding.min_byte1 + 1);
    const encodings_count = cols * rows;

    // Read glyph indices
    encoding.glyph_indices = try allocator.alloc(u16, encodings_count);
    for (encoding.glyph_indices) |*index| {
        index.* = try reader.readInt(u16, byte_order);
    }

    return encoding;
}

/// Metrics parsing result
const MetricsInfo = struct {
    metrics: []UncompressedMetric,
    glyph_count: usize,
};

/// Parse metrics table
fn parseMetrics(allocator: std.mem.Allocator, data: []const u8, table: TableEntry, max_glyphs: usize) !MetricsInfo {
    if (table.offset + table.size > data.len) {
        return PcfError.InvalidTableEntry;
    }

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    const compressed = (format & PCF_COMPRESSED_METRICS) != 0;

    var result: MetricsInfo = undefined;

    if (compressed) {
        // Read compressed metrics count
        const metrics_count = try reader.readInt(u16, byte_order);
        result.glyph_count = metrics_count;

        // Allocate and read compressed metrics
        result.metrics = try allocator.alloc(UncompressedMetric, metrics_count);

        for (result.metrics) |*metric| {
            // Read bytes and convert to signed values
            const lsb_byte = try reader.readInt(u8, .little);
            const rsb_byte = try reader.readInt(u8, .little);
            const cw_byte = try reader.readInt(u8, .little);
            const asc_byte = try reader.readInt(u8, .little);
            const desc_byte = try reader.readInt(u8, .little);

            const comp = CompressedMetric{
                .left_sided_bearing = @as(i8, @intCast(@as(i16, lsb_byte) - 0x80)),
                .right_sided_bearing = @as(i8, @intCast(@as(i16, rsb_byte) - 0x80)),
                .character_width = @as(i8, @intCast(@as(i16, cw_byte) - 0x80)),
                .ascent = @as(i8, @intCast(@as(i16, asc_byte) - 0x80)),
                .descent = @as(i8, @intCast(@as(i16, desc_byte) - 0x80)),
            };

            // Convert to uncompressed format
            metric.* = UncompressedMetric{
                .left_sided_bearing = comp.left_sided_bearing,
                .right_sided_bearing = comp.right_sided_bearing,
                .character_width = comp.character_width,
                .ascent = comp.ascent,
                .descent = comp.descent,
                .attributes = 0,
            };
        }
    } else {
        // Read uncompressed metrics count
        const metrics_count = try reader.readInt(u32, byte_order);
        result.glyph_count = @min(metrics_count, max_glyphs);

        // Allocate and read uncompressed metrics
        result.metrics = try allocator.alloc(UncompressedMetric, result.glyph_count);

        for (result.metrics) |*metric| {
            metric.* = try readUncompressedMetric(reader, byte_order);
        }
    }

    return result;
}

/// Bitmap parsing result
const BitmapInfo = struct {
    bitmap_data: []u8,
    offsets: []u32,
    bitmap_sizes: []u32,
    format: u32,
};

/// Parse bitmaps table
fn parseBitmaps(allocator: std.mem.Allocator, data: []const u8, table: TableEntry) !BitmapInfo {
    if (table.offset + table.size > data.len) {
        return PcfError.InvalidTableEntry;
    }

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    // Read glyph count
    const glyph_count = try reader.readInt(u32, byte_order);

    // Allocate offset array
    var result: BitmapInfo = undefined;
    result.format = format;
    result.offsets = try allocator.alloc(u32, glyph_count);

    // Read offsets
    for (result.offsets) |*offset| {
        offset.* = try reader.readInt(u32, byte_order);
    }

    // Skip bitmap sizes array (4 u32 values)
    _ = try reader.readInt(u32, byte_order);
    _ = try reader.readInt(u32, byte_order);
    _ = try reader.readInt(u32, byte_order);
    _ = try reader.readInt(u32, byte_order);

    // For now, just read remaining data in the table
    const remaining = try reader.context.getEndPos() - try reader.context.getPos();
    const total_size = remaining;

    result.bitmap_sizes = try allocator.alloc(u32, 1);
    result.bitmap_sizes[0] = @intCast(total_size);

    // Read bitmap data
    result.bitmap_data = try allocator.alloc(u8, total_size);
    const bytes_read = try reader.read(result.bitmap_data);
    if (bytes_read != total_size) {
        return PcfError.InvalidBitmapData;
    }

    return result;
}

/// Check if a glyph should be included based on filter
fn shouldIncludeGlyph(encoding: u32, filter: LoadFilter) bool {
    switch (filter) {
        .all => return true,
        .ranges => |ranges| {
            for (ranges) |range| {
                if (encoding >= range.start and encoding <= range.end) {
                    return true;
                }
            }
            return false;
        },
    }
}

/// Convert parsed PCF data to BitmapFont format
fn convertToBitmapFont(
    allocator: std.mem.Allocator,
    metrics_info: MetricsInfo,
    bitmap_info: BitmapInfo,
    encoding: EncodingEntry,
    filter: LoadFilter,
    font_ascent: i16,
    max_width: u16,
    max_height: u16,
) !BitmapFont {
    // Determine which glyphs to include
    var glyph_list = std.ArrayList(struct {
        codepoint: u32,
        glyph_index: usize,
        metric: UncompressedMetric,
    }).init(allocator);
    defer glyph_list.deinit();

    var all_ascii = true;
    var min_char: u8 = 255;
    var max_char: u8 = 0;

    // Build glyph list based on encodings and filter
    for (encoding.glyph_indices, 0..) |glyph_idx, enc_idx| {
        if (glyph_idx == 0xFFFF) continue; // Skip non-existent glyphs

        // Calculate codepoint from encoding index
        const row = enc_idx / (encoding.max_char_or_byte2 - encoding.min_char_or_byte2 + 1);
        const col = enc_idx % (encoding.max_char_or_byte2 - encoding.min_char_or_byte2 + 1);
        const codepoint = @as(u32, @intCast(((encoding.min_byte1 + row) << 8) | (encoding.min_char_or_byte2 + col)));

        if (!shouldIncludeGlyph(codepoint, filter)) continue;

        if (glyph_idx < metrics_info.glyph_count) {
            try glyph_list.append(.{
                .codepoint = codepoint,
                .glyph_index = glyph_idx,
                .metric = metrics_info.metrics[glyph_idx],
            });

            if (codepoint > 127) {
                all_ascii = false;
            } else {
                min_char = @min(min_char, @as(u8, @intCast(codepoint)));
                max_char = @max(max_char, @as(u8, @intCast(codepoint)));
            }
        }
    }

    // Convert bitmap data from PCF format to our format
    var converted_bitmaps = std.ArrayList(u8).init(allocator);
    defer converted_bitmaps.deinit();

    var glyph_map = std.AutoHashMap(u32, usize).init(allocator);
    errdefer glyph_map.deinit();

    var glyph_data_list = try allocator.alloc(GlyphData, glyph_list.items.len);
    errdefer allocator.free(glyph_data_list);

    for (glyph_list.items, 0..) |glyph, idx| {
        const metric = glyph.metric;
        const glyph_width = @as(u16, @intCast(@abs(metric.right_sided_bearing - metric.left_sided_bearing)));
        const glyph_height = @as(u16, @intCast(@abs(metric.ascent + metric.descent)));

        // Get bitmap data for this glyph
        const bitmap_offset = bitmap_info.offsets[glyph.glyph_index];

        // Store converted bitmap offset
        const converted_offset = converted_bitmaps.items.len;

        // Convert PCF bitmap to our format
        const pcf_bytes_per_row = (glyph_width + 7) / 8;
        const pcf_pad: u32 = switch (bitmap_info.format & PCF_GLYPH_PAD_MASK) {
            0 => 1, // PCF_GLYPH_PAD_1
            1 => 2, // PCF_GLYPH_PAD_2
            2 => 4, // PCF_GLYPH_PAD_4
            3 => 8, // PCF_GLYPH_PAD_8
            else => unreachable,
        };
        const pcf_row_bytes = ((pcf_bytes_per_row + pcf_pad - 1) / pcf_pad) * pcf_pad;

        // Convert each row
        for (0..glyph_height) |row| {
            const src_offset = bitmap_offset + row * pcf_row_bytes;

            // Convert bitmap bytes (handle byte order if needed)
            for (0..pcf_bytes_per_row) |byte_idx| {
                if (src_offset + byte_idx < bitmap_info.bitmap_data.len) {
                    const byte = bitmap_info.bitmap_data[src_offset + byte_idx];
                    // PCF uses MSB first, convert if needed
                    const converted_byte = if ((bitmap_info.format & PCF_BIT_MASK) != 0)
                        @bitReverse(byte)
                    else
                        byte;
                    try converted_bitmaps.append(converted_byte);
                } else {
                    try converted_bitmaps.append(0);
                }
            }
        }

        // Create glyph data entry
        try glyph_map.put(glyph.codepoint, idx);

        const adjusted_y_offset = font_ascent - metric.ascent;

        glyph_data_list[idx] = GlyphData{
            .width = @intCast(glyph_width),
            .height = @intCast(glyph_height),
            .x_offset = metric.left_sided_bearing,
            .y_offset = adjusted_y_offset,
            .device_width = metric.character_width,
            .bitmap_offset = converted_offset,
        };
    }

    // Create final bitmap data
    const bitmap_data = try allocator.alloc(u8, converted_bitmaps.items.len);
    @memcpy(bitmap_data, converted_bitmaps.items);

    return BitmapFont{
        .char_width = @as(u8, @intCast(@min(max_width, 255))),
        .char_height = @as(u8, @intCast(@min(max_height, 255))),
        .first_char = if (all_ascii) min_char else 0,
        .last_char = if (all_ascii) max_char else 0,
        .data = bitmap_data,
        .glyph_map = glyph_map,
        .glyph_data = glyph_data_list,
    };
}

test "PCF format detection" {
    const pcf_header = "\x01fcp";
    const result = @import("format.zig").FontFormat.detectFromBytes(pcf_header);
    try testing.expect(result == .pcf);
}

test "PCF table parsing" {
    // Create a minimal PCF file structure for testing
    var buffer: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    const writer = stream.writer();

    // Write header
    try writer.writeInt(u32, PCF_FILE_VERSION, .little);
    try writer.writeInt(u32, 1, .little); // 1 table

    // Write table entry
    try writer.writeInt(u32, PCF_PROPERTIES, .little);
    try writer.writeInt(u32, PCF_DEFAULT_FORMAT, .little);
    try writer.writeInt(u32, 16, .little); // size
    try writer.writeInt(u32, 32, .little); // offset

    const written = stream.getWritten();

    // Test parsing
    var test_stream = std.io.fixedBufferStream(written);
    const reader = test_stream.reader();

    const header = try reader.readInt(u32, .little);
    try testing.expectEqual(PCF_FILE_VERSION, header);

    const table_count = try reader.readInt(u32, .little);
    try testing.expectEqual(@as(u32, 1), table_count);
}
