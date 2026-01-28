//! gzip compression format (RFC 1952).
//!
//! This module provides compression and decompression using the gzip format,
//! which wraps DEFLATE compression with a header and CRC32 checksum.

const std = @import("std");
const Allocator = std.mem.Allocator;
const flate = std.compress.flate;

pub const CompressionStrategy = enum { default, filtered, huffman_only, rle };

/// Compress data to gzip format
pub fn compress(gpa: Allocator, data: []const u8, options: flate.Compress.Options, strategy: CompressionStrategy) ![]u8 {
    var aw = std.Io.Writer.Allocating.init(gpa);
    defer aw.deinit();

    try aw.ensureTotalCapacity(data.len / 2 + 64);

    const buffer = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(buffer);

    if (strategy == .huffman_only) {
        var huff = try flate.Compress.Huffman.init(&aw.writer, buffer, .gzip);
        try huff.writer.writeAll(data);
        try huff.writer.flush();
    } else {
        const opts = mapOptions(options, strategy);
        var compressor = try flate.Compress.init(&aw.writer, buffer, .gzip, opts);
        try compressor.writer.writeAll(data);
        try compressor.writer.flush();
    }

    return aw.toOwnedSlice();
}

/// Decompress gzip data.
pub fn decompress(gpa: Allocator, gzip_data: []const u8, limit: std.Io.Limit) ![]u8 {
    var in_stream = std.Io.Reader.fixed(gzip_data);
    const buffer = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(buffer);

    var decompressor = flate.Decompress.init(&in_stream, .gzip, buffer);

    var aw = std.Io.Writer.Allocating.init(gpa);
    errdefer aw.deinit();

    if (limit.toInt()) |max| {
        try aw.ensureTotalCapacity(max);
    }

    var remaining = limit;
    while (remaining.nonzero()) {
        const n = decompressor.reader.stream(&aw.writer, remaining) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
        remaining = remaining.subtract(n).?;
    } else {
        // We reached the limit, check if there's more
        var one_byte_buf: [1]u8 = undefined;
        var dummy_writer = std.Io.Writer.fixed(&one_byte_buf);
        if (decompressor.reader.stream(&dummy_writer, .limited(1))) |n| {
            if (n > 0) return error.OutputLimitExceeded;
        } else |err| switch (err) {
            error.EndOfStream => {},
            else => |e| return e,
        }
    }

    return aw.toOwnedSlice();
}

fn mapOptions(options: flate.Compress.Options, strategy: CompressionStrategy) flate.Compress.Options {
    var opts = options;
    switch (strategy) {
        .default => {},
        .filtered => {
            opts.chain = @min(opts.chain, 16);
            opts.nice = @min(opts.nice, 32);
        },
        .rle => {
            opts.chain = @min(opts.chain, 8);
        },
        .huffman_only => {},
    }
    return opts;
}

test "gzip compression and decompression round-trip" {
    const allocator = std.testing.allocator;

    // Test data
    const original_data = "Hello, World! This is a test for gzip compression.";

    // Compress
    const compressed = try compress(allocator, original_data, flate.Compress.Options.level_1, .default);
    defer allocator.free(compressed);

    // Verify gzip header
    try std.testing.expectEqual(@as(u8, 0x1f), compressed[0]);
    try std.testing.expectEqual(@as(u8, 0x8b), compressed[1]);
    try std.testing.expectEqual(@as(u8, 0x08), compressed[2]);

    // Decompress
    const decompressed = try decompress(allocator, compressed, .limited(original_data.len));
    defer allocator.free(decompressed);

    // Verify round-trip
    try std.testing.expectEqualSlices(u8, original_data, decompressed);
}

test "gzip with different compression levels" {
    const allocator = std.testing.allocator;
    const base = "The quick brown fox jumps over the lazy dog. ";
    const test_data = blk: {
        var data: std.ArrayList(u8) = .empty;
        defer data.deinit(allocator);
        for (0..10) |_| {
            try data.appendSlice(allocator, base);
        }
        break :blk try data.toOwnedSlice(allocator);
    };
    defer allocator.free(test_data);

    const levels = [_]flate.Compress.Options{ flate.Compress.Options.level_1, flate.Compress.Options.level_6, flate.Compress.Options.level_9 };

    for (levels) |level| {
        const compressed = try compress(allocator, test_data, level, .default);
        defer allocator.free(compressed);

        const decompressed = try decompress(allocator, compressed, .limited(test_data.len));
        defer allocator.free(decompressed);

        try std.testing.expectEqualSlices(u8, test_data, decompressed);
    }
}

test "gzip decompress enforces caller limit" {
    const allocator = std.testing.allocator;
    const original = "limit me";
    const compressed = try compress(allocator, original, flate.Compress.Options.level_1, .default);
    defer allocator.free(compressed);

    const result = decompress(allocator, compressed, .limited(2));
    try std.testing.expectError(error.OutputLimitExceeded, result);
}

test "gzip error handling" {
    const allocator = std.testing.allocator;

    // Test invalid magic number
    const bad_magic = [_]u8{ 0x00, 0x00 } ++ @as([16]u8, @splat(0));
    // std flate might return BadGzipHeader via ReadFailed
    const res = decompress(allocator, &bad_magic, .limited(1));
    try std.testing.expectError(error.ReadFailed, res);
}
