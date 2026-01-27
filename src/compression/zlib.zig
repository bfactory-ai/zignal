//! zlib (RFC 1950) wrapper around DEFLATE

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const def = @import("deflate.zig");

/// Compresses data using zlib format.
/// Returns an owned slice that must be freed by caller.
pub fn compress(gpa: Allocator, data: []const u8, level: def.CompressionLevel, strategy: def.CompressionStrategy) ![]u8 {
    const deflate_data = try def.deflate(gpa, data, level, strategy);
    defer gpa.free(deflate_data);

    const checksum = std.hash.Adler32.hash(data);

    var result: ArrayList(u8) = .empty;
    // Pre-allocate exact size: 2 bytes header + data + 4 bytes checksum
    try result.ensureTotalCapacity(gpa, deflate_data.len + 6);
    defer result.deinit(gpa);

    const cmf: u8 = 0x78; // 32K window, deflate
    const flevel: u2 = switch (level) {
        .level_0, .level_1 => 0,
        .level_2, .level_3 => 1,
        .level_4, .level_5, .level_6 => 2,
        .level_7, .level_8, .level_9 => 3,
    };

    var flg: u8 = @as(u8, flevel) << 6;
    const header_base = (@as(u16, cmf) << 8) | flg;
    const fcheck = 31 - (header_base % 31);
    if (fcheck < 31) flg |= @intCast(fcheck);

    // Write header
    try result.append(gpa, cmf);
    try result.append(gpa, flg);

    // Write data
    try result.appendSlice(gpa, deflate_data);

    // Write checksum (Big Endian)
    try result.append(gpa, @intCast((checksum >> 24) & 0xFF));
    try result.append(gpa, @intCast((checksum >> 16) & 0xFF));
    try result.append(gpa, @intCast((checksum >> 8) & 0xFF));
    try result.append(gpa, @intCast(checksum & 0xFF));

    return result.toOwnedSlice(gpa);
}

/// Decompresses zlib data.
/// Returns an owned slice that must be freed by caller.
pub fn decompress(gpa: Allocator, zlib_data: []const u8, max_output_bytes: usize) ![]u8 {
    if (zlib_data.len < 6) return error.InvalidZlibData;

    const cmf = zlib_data[0];
    const flg = zlib_data[1];
    const header_check = (@as(u16, cmf) << 8) | flg;

    if ((cmf & 0x0F) != 8) return error.UnsupportedCompressionMethod;
    if ((header_check % 31) != 0) return error.InvalidZlibHeader;
    if ((flg & 0x20) != 0) return error.PresetDictionaryNotSupported;

    const deflate_data = zlib_data[2 .. zlib_data.len - 4];
    const decompressed = try def.inflate(gpa, deflate_data, max_output_bytes);
    errdefer gpa.free(decompressed);

    const expected_checksum = std.mem.readInt(u32, zlib_data[zlib_data.len - 4 ..][0..4], .big);
    const actual_checksum = std.hash.Adler32.hash(decompressed);

    if (actual_checksum != expected_checksum) {
        return error.ChecksumMismatch;
    }

    return decompressed;
}

test "zlib round trip" {
    const allocator = std.testing.allocator;
    const original_data = "Hello, zlib compression test for PNG!";
    const compressed = try compress(allocator, original_data, .level_0, .default);
    defer allocator.free(compressed);
    const decompressed = try decompress(allocator, compressed, original_data.len);
    defer allocator.free(decompressed);
    try std.testing.expectEqualSlices(u8, original_data, decompressed);
}

test "zlib header validation" {
    const allocator = std.testing.allocator;
    const test_data = "Test";
    const compressed = try compress(allocator, test_data, .level_0, .default);
    defer allocator.free(compressed);
    try std.testing.expect(compressed.len >= 6);
    const cmf = compressed[0];
    try std.testing.expectEqual(@as(u8, 8), cmf & 0x0F);
    const header_check = (@as(u16, compressed[0]) << 8) | compressed[1];
    try std.testing.expectEqual(@as(u16, 0), header_check % 31);
}

test "zlib compression levels" {
    const allocator = std.testing.allocator;
    const base = "The quick brown fox jumps over the lazy dog. ";
    const test_data = blk: {
        var data: ArrayList(u8) = .empty;
        defer data.deinit(allocator);
        for (0..10) |_| {
            try data.appendSlice(allocator, base);
        }
        break :blk try data.toOwnedSlice(allocator);
    };
    defer allocator.free(test_data);
    const levels = [_]def.CompressionLevel{ .level_0, .level_1, .level_3, .level_6, .level_9 };
    var sizes: [levels.len]usize = undefined;
    for (levels, 0..) |level, i| {
        const compressed = try compress(allocator, test_data, level, .default);
        defer allocator.free(compressed);
        sizes[i] = compressed.len;
        const decomp = try decompress(allocator, compressed, test_data.len);
        defer allocator.free(decomp);
        try std.testing.expectEqualSlices(u8, test_data, decomp);
    }
    // Level 0 (no compression) should be largest
    try std.testing.expect(sizes[0] > sizes[1]);
}
