//! gzip compression format (RFC 1952).
//!
//! This module provides compression and decompression using the gzip format,
//! which wraps DEFLATE compression with a header and CRC32 checksum.

const std = @import("std");
const deflate = @import("deflate.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// CRC32 implementation for gzip
const crc32_table = blk: {
    @setEvalBranchQuota(3000);
    var table: [256]u32 = undefined;
    const polynomial: u32 = 0xEDB88320; // Reversed polynomial

    for (0..256) |i| {
        var crc: u32 = @intCast(i);
        for (0..8) |_| {
            if (crc & 1 != 0) {
                crc = (crc >> 1) ^ polynomial;
            } else {
                crc = crc >> 1;
            }
        }
        table[i] = crc;
    }
    break :blk table;
};

/// Calculate CRC32 checksum
fn crc32(data: []const u8) u32 {
    var crc: u32 = 0xFFFFFFFF;
    for (data) |byte| {
        const table_idx = (crc ^ byte) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[table_idx];
    }
    return ~crc;
}

/// Compress data to gzip format
pub fn gzipCompress(gpa: Allocator, data: []const u8, level: deflate.CompressionLevel, strategy: deflate.CompressionStrategy) ![]u8 {
    // Compress data using deflate
    const compressed_data = try deflate.deflate(gpa, data, level, strategy);
    defer gpa.free(compressed_data);

    // Calculate CRC32 checksum of original data
    const checksum = crc32(data);

    // Create gzip output
    var result: ArrayList(u8) = .empty;
    defer result.deinit(gpa);

    // Write gzip header (10 bytes)
    try result.append(gpa, 0x1f); // Magic number byte 1
    try result.append(gpa, 0x8b); // Magic number byte 2
    try result.append(gpa, 0x08); // Compression method (deflate)
    try result.append(gpa, 0x00); // Flags (no extra fields)

    // Modification time (4 bytes) - set to 0
    try result.append(gpa, 0x00);
    try result.append(gpa, 0x00);
    try result.append(gpa, 0x00);
    try result.append(gpa, 0x00);

    try result.append(gpa, 0x00); // Extra flags
    try result.append(gpa, 0x03); // OS (Unix)

    // Write compressed data
    try result.appendSlice(gpa, compressed_data);

    // Write gzip trailer (8 bytes)
    // CRC32 (4 bytes, little-endian)
    try result.append(gpa, @intCast(checksum & 0xFF));
    try result.append(gpa, @intCast((checksum >> 8) & 0xFF));
    try result.append(gpa, @intCast((checksum >> 16) & 0xFF));
    try result.append(gpa, @intCast((checksum >> 24) & 0xFF));

    // Uncompressed size (4 bytes, little-endian)
    const size: u32 = @intCast(data.len & 0xFFFFFFFF);
    try result.append(gpa, @intCast(size & 0xFF));
    try result.append(gpa, @intCast((size >> 8) & 0xFF));
    try result.append(gpa, @intCast((size >> 16) & 0xFF));
    try result.append(gpa, @intCast((size >> 24) & 0xFF));

    return result.toOwnedSlice(gpa);
}

/// Decompress gzip data
pub fn gzipDecompress(allocator: Allocator, gzip_data: []const u8) ![]u8 {
    if (gzip_data.len < 18) { // Minimum gzip file size
        return error.InvalidGzipData;
    }

    // Check gzip magic number
    if (gzip_data[0] != 0x1f or gzip_data[1] != 0x8b) {
        return error.InvalidGzipHeader;
    }

    // Check compression method (must be deflate)
    if (gzip_data[2] != 8) {
        return error.UnsupportedCompressionMethod;
    }

    // Parse header flags
    const flags = gzip_data[3];
    var offset: usize = 10; // Fixed header size

    // Skip optional fields based on flags
    if (flags & 0x04 != 0) { // FEXTRA
        if (offset + 2 > gzip_data.len) return error.InvalidGzipData;
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
        return error.InvalidGzipData;
    }

    const compressed_data = gzip_data[offset .. gzip_data.len - 8];
    const decompressed = try deflate.inflate(allocator, compressed_data);
    errdefer allocator.free(decompressed);

    // Verify CRC32
    const expected_crc = std.mem.readInt(u32, gzip_data[gzip_data.len - 8 ..][0..4], .little);
    const actual_crc = crc32(decompressed);
    if (actual_crc != expected_crc) {
        allocator.free(decompressed);
        return error.ChecksumMismatch;
    }

    // Verify uncompressed size
    const expected_size = std.mem.readInt(u32, gzip_data[gzip_data.len - 4 ..][0..4], .little);
    const actual_size: u32 = @intCast(decompressed.len & 0xFFFFFFFF);
    if (actual_size != expected_size) {
        allocator.free(decompressed);
        return error.SizeMismatch;
    }

    return decompressed;
}

// Backwards compatibility wrapper with defaults
pub fn compressGzip(gpa: Allocator, data: []const u8) ![]u8 {
    return gzipCompress(gpa, data, .fastest, .default);
}
pub const decompressGzip = gzipDecompress;

test "gzip compression and decompression round-trip" {
    const allocator = std.testing.allocator;

    // Test data
    const original_data = "Hello, World! This is a test for gzip compression.";

    // Compress
    const compressed = try gzipCompress(allocator, original_data, .fastest, .default);
    defer allocator.free(compressed);

    // Verify gzip header
    try std.testing.expectEqual(@as(u8, 0x1f), compressed[0]);
    try std.testing.expectEqual(@as(u8, 0x8b), compressed[1]);
    try std.testing.expectEqual(@as(u8, 0x08), compressed[2]);

    // Decompress
    const decompressed = try gzipDecompress(allocator, compressed);
    defer allocator.free(decompressed);

    // Verify round-trip
    try std.testing.expectEqualSlices(u8, original_data, decompressed);
}

test "gzip with different compression levels" {
    const allocator = std.testing.allocator;
    const test_data = "The quick brown fox jumps over the lazy dog. " ** 10;

    const levels = [_]deflate.CompressionLevel{ .none, .fastest, .default, .best };

    for (levels) |level| {
        const compressed = try gzipCompress(allocator, test_data, level, .default);
        defer allocator.free(compressed);

        const decompressed = try gzipDecompress(allocator, compressed);
        defer allocator.free(decompressed);

        try std.testing.expectEqualSlices(u8, test_data, decompressed);
    }
}

test "gzip error handling" {
    const allocator = std.testing.allocator;

    // Test invalid magic number
    const bad_magic = [_]u8{ 0x00, 0x00 } ++ ([_]u8{0} ** 16);
    try std.testing.expectError(error.InvalidGzipHeader, gzipDecompress(allocator, &bad_magic));

    // Test too short data
    const too_short = [_]u8{ 0x1f, 0x8b };
    try std.testing.expectError(error.InvalidGzipData, gzipDecompress(allocator, &too_short));
}
