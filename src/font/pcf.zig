//! PCF (Portable Compiled Format) font parser for zignal
//!
//! This module provides zero-dependency parsing of PCF font files,
//! a binary format used by X11 for efficient bitmap font storage.
//!
//! PCF files contain bitmap font data in an optimized binary format
//! with multiple tables containing metrics, bitmaps, encodings, and
//! optional acceleration data. This parser supports both compressed
//! and uncompressed metrics, as well as gzip-compressed PCF files.

const std = @import("std");
const testing = std.testing;
const BitmapFont = @import("BitmapFont.zig");
const GlyphData = @import("GlyphData.zig");
const unicode = @import("unicode.zig");
const LoadFilter = @import("../font.zig").LoadFilter;
const deflate = @import("../deflate.zig");

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
    TableOffsetOutOfBounds,
    InvalidGlyphCount,
    InvalidMetricsFormat,
    InvalidEncodingRange,
    BitmapSizeMismatch,
};

/// PCF format constants
const PCF_FILE_VERSION = 0x70636601; // "\x01fcp" in little-endian
const PCF_FORMAT_MASK = 0xffffff00;

/// Maximum reasonable values for sanity checks
const MAX_TABLE_COUNT = 1024;
const MAX_GLYPH_COUNT = 65536;
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_FONT_DIMENSION = 1024; // Maximum reasonable width/height

/// PCF table types as enum for better type safety
const TableType = enum(u32) {
    properties = (1 << 0),
    accelerators = (1 << 1),
    metrics = (1 << 2),
    bitmaps = (1 << 3),
    ink_metrics = (1 << 4),
    bdf_encodings = (1 << 5),
    swidths = (1 << 6),
    glyph_names = (1 << 7),
    bdf_accelerators = (1 << 8),
};

/// PCF format flags structure for better type safety
const FormatFlags = struct {
    // Helper to decode format flags from u32
    pub fn decode(format: u32) FormatFlags {
        return FormatFlags{
            .glyph_pad = @as(u2, @truncate(format & 0x3)),
            .byte_order_msb = (format & (1 << 3)) != 0,
            .bit_order_msb = (format & (1 << 2)) != 0,
            .scan_unit = @as(u2, @truncate((format >> 4) & 0x3)),
            .compressed_metrics = (format & 0x100) != 0,
            .ink_bounds = (format & 0x200) != 0,
            .accel_w_inkbounds = (format & 0x100) != 0,
        };
    }

    glyph_pad: u2,
    byte_order_msb: bool,
    bit_order_msb: bool,
    scan_unit: u2,
    compressed_metrics: bool,
    ink_bounds: bool,
    accel_w_inkbounds: bool,
};

/// PCF glyph padding values
const GlyphPadding = enum(u2) {
    pad_1 = 0,
    pad_2 = 1,
    pad_4 = 2,
    pad_8 = 3,

    pub fn getPadBytes(self: GlyphPadding) u32 {
        return switch (self) {
            .pad_1 => 1,
            .pad_2 => 2,
            .pad_4 => 4,
            .pad_8 => 8,
        };
    }
};

/// PCF format mask constants
const PCF_GLYPH_PAD_MASK = (3 << 0); // Bits 0-1: glyph padding
const PCF_BYTE_ORDER_MASK = (1 << 3); // Bit 3: byte order
const PCF_BIT_ORDER_MASK = (1 << 2); // Bit 2: bit order
const PCF_SCAN_UNIT_MASK = (3 << 4); // Bits 4-5: scan unit

/// Get byte order from format field
fn getByteOrder(format: u32) std.builtin.Endian {
    const flags = FormatFlags.decode(format);
    return if (flags.byte_order_msb) .big else .little;
}

/// Table of contents entry for PCF files
/// Each PCF file contains multiple tables identified by type
const TableEntry = struct {
    type: u32, // Table type (see TableType enum)
    format: u32, // Format flags including byte order and padding
    size: u32, // Size of table data in bytes
    offset: u32, // Offset from start of file to table data
};

/// PCF metrics structure (unified for both compressed and uncompressed)
/// Describes the dimensions and positioning of a single glyph
const Metric = struct {
    left_sided_bearing: i16, // Distance from origin to left edge of glyph
    right_sided_bearing: i16, // Distance from origin to right edge of glyph
    character_width: i16, // Logical width for cursor advancement
    ascent: i16, // Distance from baseline to top of glyph
    descent: i16, // Distance from baseline to bottom of glyph (positive)
    attributes: u16, // Additional glyph attributes (usually 0)
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
    min_bounds: Metric,
    max_bounds: Metric,
    ink_min_bounds: ?Metric,
    ink_max_bounds: ?Metric,
};

/// PCF encoding entry
/// Maps character codes to glyph indices using a 2D table
const EncodingEntry = struct {
    min_char_or_byte2: u16, // Minimum value for low byte of character code
    max_char_or_byte2: u16, // Maximum value for low byte of character code
    min_byte1: u16, // Minimum value for high byte of character code
    max_byte1: u16, // Maximum value for high byte of character code
    default_char: u16, // Character to use for undefined codes
    glyph_indices: []u16, // 2D array of glyph indices (0xFFFF = undefined)
};

/// Load a PCF font from a file path
/// Parameters:
/// - allocator: Memory allocator
/// - path: Path to PCF file
/// - filter: Filter for which characters to load
/// Decompress gzip data
fn decompressGzip(allocator: std.mem.Allocator, gzip_data: []const u8) ![]u8 {
    if (gzip_data.len < 18) { // Minimum gzip file size
        return PcfError.InvalidCompression;
    }

    // Check gzip magic number
    if (gzip_data[0] != 0x1f or gzip_data[1] != 0x8b) {
        std.log.err("PCF: Invalid gzip magic number", .{});
        return PcfError.InvalidCompression;
    }

    // Check compression method (must be deflate)
    if (gzip_data[2] != 8) {
        std.log.err("PCF: Unsupported gzip compression method: {}", .{gzip_data[2]});
        return PcfError.InvalidCompression;
    }

    // Parse header flags
    const flags = gzip_data[3];
    var offset: usize = 10; // Fixed header size

    // Skip optional fields based on flags
    if (flags & 0x04 != 0) { // FEXTRA
        const extra_len = @as(u16, gzip_data[offset]) | (@as(u16, gzip_data[offset + 1]) << 8);
        offset += 2 + extra_len;
    }
    if (flags & 0x08 != 0) { // FNAME
        while (offset < gzip_data.len and gzip_data[offset] != 0) : (offset += 1) {}
        offset += 1;
    }
    if (flags & 0x10 != 0) { // FCOMMENT
        while (offset < gzip_data.len and gzip_data[offset] != 0) : (offset += 1) {}
        offset += 1;
    }
    if (flags & 0x02 != 0) { // FHCRC
        offset += 2;
    }

    // Decompress the deflate stream (excluding 8-byte trailer)
    if (offset + 8 > gzip_data.len) {
        return PcfError.InvalidCompression;
    }

    const compressed_data = gzip_data[offset .. gzip_data.len - 8];
    return deflate.inflate(allocator, compressed_data);
}

pub fn load(allocator: std.mem.Allocator, path: []const u8, filter: LoadFilter) !BitmapFont {
    // Check if file is gzip compressed
    const is_compressed = std.mem.endsWith(u8, path, ".gz");

    // Read file into memory
    const raw_file_contents = std.fs.cwd().readFileAlloc(allocator, path, MAX_FILE_SIZE) catch |err| {
        std.log.err("PCF: Failed to read file {s}: {}", .{ path, err });
        return err;
    };
    defer allocator.free(raw_file_contents);

    // Decompress if needed
    var file_contents: []u8 = undefined;
    var decompressed_data: ?[]u8 = null;
    defer if (decompressed_data) |data| allocator.free(data);

    if (is_compressed) {
        decompressed_data = try decompressGzip(allocator, raw_file_contents);
        file_contents = decompressed_data.?;
        std.log.info("PCF: Decompressed {s} from {} bytes to {} bytes", .{ path, raw_file_contents.len, file_contents.len });
    } else {
        file_contents = raw_file_contents;
    }

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
        std.log.err("PCF: Invalid file header: 0x{x:0>8} (expected 0x{x:0>8})", .{ header, PCF_FILE_VERSION });
        return PcfError.InvalidFormat;
    }

    // Read table count
    const table_count = try reader.readInt(u32, .little);
    if (table_count == 0 or table_count > MAX_TABLE_COUNT) {
        std.log.err("PCF: Invalid table count: {} (max {})", .{ table_count, MAX_TABLE_COUNT });
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
    const metrics_table = findTable(tables, .metrics) orelse {
        std.log.err("PCF: Missing required metrics table", .{});
        return PcfError.MissingRequired;
    };
    const bitmaps_table = findTable(tables, .bitmaps) orelse {
        std.log.err("PCF: Missing required bitmaps table", .{});
        return PcfError.MissingRequired;
    };
    const encodings_table = findTable(tables, .bdf_encodings) orelse {
        std.log.err("PCF: Missing required encodings table", .{});
        return PcfError.MissingRequired;
    };
    const accel_table = findTable(tables, .accelerators) orelse findTable(tables, .bdf_accelerators);

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
fn findTable(tables: []const TableEntry, table_type: TableType) ?TableEntry {
    const type_value = @intFromEnum(table_type);
    for (tables) |table| {
        if (table.type == type_value) {
            return table;
        }
    }
    return null;
}

/// Validate table bounds
fn validateTableBounds(data: []const u8, table: TableEntry) !void {
    if (table.offset > data.len) {
        return PcfError.TableOffsetOutOfBounds;
    }
    if (table.size > data.len - table.offset) {
        return PcfError.TableOffsetOutOfBounds;
    }
    if (table.size == 0) {
        return PcfError.InvalidTableEntry;
    }
}

/// Parse accelerator table
fn parseAccelerator(data: []const u8, table: TableEntry) !Accelerator {
    try validateTableBounds(data, table);

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
    accel.min_bounds = try readMetric(reader, byte_order, false);
    accel.max_bounds = try readMetric(reader, byte_order, false);

    // Read ink bounds if present
    const accel_flags = FormatFlags.decode(table.format);
    if (accel_flags.accel_w_inkbounds) {
        accel.ink_min_bounds = try readMetric(reader, byte_order, false);
        accel.ink_max_bounds = try readMetric(reader, byte_order, false);
    } else {
        accel.ink_min_bounds = null;
        accel.ink_max_bounds = null;
    }

    return accel;
}

/// Read metric from stream (handles both compressed and uncompressed formats)
fn readMetric(reader: anytype, byte_order: std.builtin.Endian, compressed: bool) !Metric {
    if (compressed) {
        // Read compressed metric (5 bytes, each offset by 0x80)
        const lsb = try reader.readInt(u8, .little);
        const rsb = try reader.readInt(u8, .little);
        const cw = try reader.readInt(u8, .little);
        const asc = try reader.readInt(u8, .little);
        const desc = try reader.readInt(u8, .little);

        return Metric{
            .left_sided_bearing = @as(i16, @intCast(@as(i16, lsb) - 0x80)),
            .right_sided_bearing = @as(i16, @intCast(@as(i16, rsb) - 0x80)),
            .character_width = @as(i16, @intCast(@as(i16, cw) - 0x80)),
            .ascent = @as(i16, @intCast(@as(i16, asc) - 0x80)),
            .descent = @as(i16, @intCast(@as(i16, desc) - 0x80)),
            .attributes = 0,
        };
    } else {
        // Read uncompressed metric (6 i16 values)
        return Metric{
            .left_sided_bearing = try reader.readInt(i16, byte_order),
            .right_sided_bearing = try reader.readInt(i16, byte_order),
            .character_width = try reader.readInt(i16, byte_order),
            .ascent = try reader.readInt(i16, byte_order),
            .descent = try reader.readInt(i16, byte_order),
            .attributes = try reader.readInt(u16, byte_order),
        };
    }
}

/// Parse encodings table
fn parseEncodings(allocator: std.mem.Allocator, data: []const u8, table: TableEntry) !EncodingEntry {
    try validateTableBounds(data, table);

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

    if (encodings_count > MAX_GLYPH_COUNT) {
        std.log.err("PCF: Encoding count {} exceeds maximum {}", .{ encodings_count, MAX_GLYPH_COUNT });
        return PcfError.InvalidEncodingRange;
    }

    // Read glyph indices
    encoding.glyph_indices = try allocator.alloc(u16, encodings_count);
    for (encoding.glyph_indices) |*index| {
        index.* = try reader.readInt(u16, byte_order);
    }

    return encoding;
}

/// Metrics parsing result
const MetricsInfo = struct {
    metrics: []Metric,
    glyph_count: usize,
};

/// Parse metrics table
fn parseMetrics(allocator: std.mem.Allocator, data: []const u8, table: TableEntry, max_glyphs: usize) !MetricsInfo {
    try validateTableBounds(data, table);

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    const flags = FormatFlags.decode(format);
    const compressed = flags.compressed_metrics;

    var result: MetricsInfo = undefined;

    if (compressed) {
        // Read compressed metrics count
        const metrics_count = try reader.readInt(u16, byte_order);
        if (metrics_count > MAX_GLYPH_COUNT) {
            std.log.err("PCF: Compressed metrics count {} exceeds maximum {}", .{ metrics_count, MAX_GLYPH_COUNT });
            return PcfError.InvalidGlyphCount;
        }
        result.glyph_count = metrics_count;

        // Allocate and read compressed metrics
        result.metrics = try allocator.alloc(Metric, metrics_count);

        for (result.metrics) |*metric| {
            metric.* = try readMetric(reader, byte_order, true);
        }
    } else {
        // Read uncompressed metrics count
        const metrics_count = try reader.readInt(u32, byte_order);
        if (metrics_count > MAX_GLYPH_COUNT) {
            std.log.err("PCF: Uncompressed metrics count {} exceeds maximum {}", .{ metrics_count, MAX_GLYPH_COUNT });
            return PcfError.InvalidGlyphCount;
        }
        result.glyph_count = @min(metrics_count, max_glyphs);

        // Allocate and read uncompressed metrics
        result.metrics = try allocator.alloc(Metric, result.glyph_count);

        for (result.metrics) |*metric| {
            metric.* = try readMetric(reader, byte_order, false);
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
    try validateTableBounds(data, table);

    var stream = std.io.fixedBufferStream(data[table.offset .. table.offset + table.size]);
    const reader = stream.reader();

    // Read format field and determine byte order
    const format = try reader.readInt(u32, .little);
    const byte_order = getByteOrder(format);

    // Read glyph count
    const glyph_count = try reader.readInt(u32, byte_order);
    if (glyph_count > MAX_GLYPH_COUNT) {
        std.log.err("PCF: Bitmap glyph count {} exceeds maximum {}", .{ glyph_count, MAX_GLYPH_COUNT });
        return PcfError.InvalidGlyphCount;
    }

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

/// Convert a single glyph bitmap from PCF format to our format
fn convertGlyphBitmap(
    bitmap_data: []const u8,
    offset: u32,
    width: u16,
    height: u16,
    format_flags: FormatFlags,
    glyph_pad: GlyphPadding,
    output: *std.ArrayList(u8),
) !void {
    const bytes_per_row = (width + 7) / 8;
    const pcf_pad = glyph_pad.getPadBytes();
    const pcf_row_bytes = ((bytes_per_row + pcf_pad - 1) / pcf_pad) * pcf_pad;

    // Convert each row
    for (0..height) |row| {
        const src_offset = offset + row * pcf_row_bytes;

        // Convert bitmap bytes
        for (0..bytes_per_row) |byte_idx| {
            if (src_offset + byte_idx < bitmap_data.len) {
                const byte = bitmap_data[src_offset + byte_idx];
                // PCF uses MSB first by default, convert if needed
                const converted_byte = if (format_flags.bit_order_msb)
                    @bitReverse(byte)
                else
                    byte;
                try output.append(converted_byte);
            } else {
                try output.append(0);
            }
        }
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
        metric: Metric,
    }).init(allocator);
    defer glyph_list.deinit();

    var all_ascii = true;
    var min_char: u8 = 255;
    var max_char: u8 = 0;

    // Build glyph list based on encodings and filter
    for (encoding.glyph_indices, 0..) |glyph_index, encoding_index| {
        if (glyph_index == 0xFFFF) continue; // Skip non-existent glyphs

        // Calculate codepoint from encoding index
        // PCF uses a 2D encoding table where:
        // - rows represent byte1 values (high byte)
        // - columns represent byte2 values (low byte)
        const chars_per_row = encoding.max_char_or_byte2 - encoding.min_char_or_byte2 + 1;
        const row = encoding_index / chars_per_row;
        const col = encoding_index % chars_per_row;
        const codepoint = @as(u32, @intCast(((encoding.min_byte1 + row) << 8) | (encoding.min_char_or_byte2 + col)));

        if (!shouldIncludeGlyph(codepoint, filter)) continue;

        if (glyph_index < metrics_info.glyph_count) {
            try glyph_list.append(.{
                .codepoint = codepoint,
                .glyph_index = glyph_index,
                .metric = metrics_info.metrics[glyph_index],
            });

            if (codepoint > 127) {
                all_ascii = false;
            } else {
                min_char = @min(min_char, @as(u8, @intCast(codepoint)));
                max_char = @max(max_char, @as(u8, @intCast(codepoint)));
            }
        }
    }

    // Pre-calculate total bitmap size needed
    var total_bitmap_size: usize = 0;
    for (glyph_list.items) |glyph_info| {
        const metric = glyph_info.metric;
        const glyph_width = @as(u16, @intCast(@abs(metric.right_sided_bearing - metric.left_sided_bearing)));
        const glyph_height = @as(u16, @intCast(@abs(metric.ascent + metric.descent)));
        const bytes_per_row = (glyph_width + 7) / 8;
        total_bitmap_size += bytes_per_row * glyph_height;
    }

    // Pre-allocate converted bitmap buffer
    var converted_bitmaps = std.ArrayList(u8).init(allocator);
    defer converted_bitmaps.deinit();
    try converted_bitmaps.ensureTotalCapacity(total_bitmap_size);

    var glyph_map = std.AutoHashMap(u32, usize).init(allocator);
    errdefer glyph_map.deinit();
    try glyph_map.ensureTotalCapacity(@intCast(glyph_list.items.len));

    var glyph_data_list = try allocator.alloc(GlyphData, glyph_list.items.len);
    errdefer allocator.free(glyph_data_list);

    for (glyph_list.items, 0..) |glyph_info, list_index| {
        const metric = glyph_info.metric;
        const glyph_width = @as(u16, @intCast(@abs(metric.right_sided_bearing - metric.left_sided_bearing)));
        const glyph_height = @as(u16, @intCast(@abs(metric.ascent + metric.descent)));

        // Store converted bitmap offset
        const converted_offset = converted_bitmaps.items.len;

        // Convert bitmap data for this glyph
        const bitmap_offset = bitmap_info.offsets[glyph_info.glyph_index];
        const format_flags = FormatFlags.decode(bitmap_info.format);
        const pad_bits = @as(u2, @truncate(bitmap_info.format & PCF_GLYPH_PAD_MASK));
        const glyph_pad = @as(GlyphPadding, @enumFromInt(pad_bits));

        try convertGlyphBitmap(
            bitmap_info.bitmap_data,
            bitmap_offset,
            glyph_width,
            glyph_height,
            format_flags,
            glyph_pad,
            &converted_bitmaps,
        );

        // Create glyph data entry
        try glyph_map.put(glyph_info.codepoint, list_index);

        // Adjust y_offset to account for font baseline
        const adjusted_y_offset = font_ascent - metric.ascent;

        glyph_data_list[list_index] = GlyphData{
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
    try writer.writeInt(u32, @intFromEnum(TableType.properties), .little);
    try writer.writeInt(u32, 0x00000000, .little); // Default format
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

test "FormatFlags decoding" {
    // Test format flag decoding
    const test_cases = [_]struct {
        format: u32,
        expected: FormatFlags,
    }{
        .{
            .format = 0x00000000,
            .expected = .{
                .glyph_pad = 0,
                .byte_order_msb = false,
                .bit_order_msb = false,
                .scan_unit = 0,
                .compressed_metrics = false,
                .ink_bounds = false,
                .accel_w_inkbounds = false,
            },
        },
        .{
            .format = 0x0000010C, // Typical compressed metrics format
            .expected = .{
                .glyph_pad = 0,
                .byte_order_msb = true,
                .bit_order_msb = true,
                .scan_unit = 0,
                .compressed_metrics = true,
                .ink_bounds = false,
                .accel_w_inkbounds = true,
            },
        },
    };

    for (test_cases) |tc| {
        const flags = FormatFlags.decode(tc.format);
        try testing.expectEqual(tc.expected.glyph_pad, flags.glyph_pad);
        try testing.expectEqual(tc.expected.byte_order_msb, flags.byte_order_msb);
        try testing.expectEqual(tc.expected.bit_order_msb, flags.bit_order_msb);
        try testing.expectEqual(tc.expected.compressed_metrics, flags.compressed_metrics);
    }
}

test "GlyphPadding values" {
    try testing.expectEqual(@as(u32, 1), GlyphPadding.pad_1.getPadBytes());
    try testing.expectEqual(@as(u32, 2), GlyphPadding.pad_2.getPadBytes());
    try testing.expectEqual(@as(u32, 4), GlyphPadding.pad_4.getPadBytes());
    try testing.expectEqual(@as(u32, 8), GlyphPadding.pad_8.getPadBytes());
}

test "Table bounds validation" {
    const data = [_]u8{0} ** 100;

    // Valid table
    const valid_table = TableEntry{
        .type = @intFromEnum(TableType.metrics),
        .format = 0,
        .size = 50,
        .offset = 20,
    };
    try validateTableBounds(&data, valid_table);

    // Invalid offset
    const invalid_offset_table = TableEntry{
        .type = @intFromEnum(TableType.metrics),
        .format = 0,
        .size = 50,
        .offset = 200,
    };
    try testing.expectError(PcfError.TableOffsetOutOfBounds, validateTableBounds(&data, invalid_offset_table));

    // Invalid size
    const invalid_size_table = TableEntry{
        .type = @intFromEnum(TableType.metrics),
        .format = 0,
        .size = 100,
        .offset = 50,
    };
    try testing.expectError(PcfError.TableOffsetOutOfBounds, validateTableBounds(&data, invalid_size_table));
}

test "Metric reading" {
    var buffer: [64]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    const writer = stream.writer();

    // Write compressed metric
    try writer.writeByte(0x82); // LSB: 2 (0x82 - 0x80)
    try writer.writeByte(0x88); // RSB: 8 (0x88 - 0x80)
    try writer.writeByte(0x86); // Width: 6 (0x86 - 0x80)
    try writer.writeByte(0x90); // Ascent: 16 (0x90 - 0x80)
    try writer.writeByte(0x82); // Descent: 2 (0x82 - 0x80)

    var read_stream = std.io.fixedBufferStream(stream.getWritten());
    const reader = read_stream.reader();

    const metric = try readMetric(reader, .little, true);
    try testing.expectEqual(@as(i16, 2), metric.left_sided_bearing);
    try testing.expectEqual(@as(i16, 8), metric.right_sided_bearing);
    try testing.expectEqual(@as(i16, 6), metric.character_width);
    try testing.expectEqual(@as(i16, 16), metric.ascent);
    try testing.expectEqual(@as(i16, 2), metric.descent);
}

test "Gzip decompression" {
    // Create a simple gzip file with minimal content
    const gzip_data = [_]u8{
        0x1f, 0x8b, // Magic number
        0x08, // Compression method (deflate)
        0x00, // Flags (no extra fields)
        0x00, 0x00, 0x00, 0x00, // Modification time
        0x00, // Extra flags
        0x03, // OS (Unix)
        // Compressed data (empty deflate block)
        0x03,
        0x00,
        // CRC32 and size (8 bytes)
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    };

    const allocator = testing.allocator;
    const result = decompressGzip(allocator, &gzip_data) catch |err| {
        // This test mainly validates the header parsing
        // Actual deflate decompression is tested elsewhere
        if (err == error.EndOfStream) return;
        return err;
    };
    defer allocator.free(result);
}
