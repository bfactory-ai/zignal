const std = @import("std");
const Allocator = std.mem.Allocator;
const flate = std.compress.flate;

pub const Strategy = union(enum) {
    default: flate.Compress.Options,
    filtered: flate.Compress.Options,
    huffman_only,
    rle: flate.Compress.Options,
};

/// Generic compression function
pub fn deflate(gpa: Allocator, data: []const u8, strategy: Strategy, container: flate.Container) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();

    // PNG and other users might benefit from some initial capacity
    try aw.ensureTotalCapacity(data.len / 2 + 64);

    const buffer = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(buffer);

    switch (strategy) {
        .huffman_only => {
            var huff: flate.Compress.Huffman = try .init(&aw.writer, buffer, container);
            try huff.writer.writeAll(data);
            try huff.writer.flush();
        },
        else => {
            const opts = applyStrategy(strategy);
            var compressor: flate.Compress = try .init(&aw.writer, buffer, container, opts);
            try compressor.writer.writeAll(data);
            try compressor.writer.flush();
        },
    }

    return aw.toOwnedSlice();
}

/// Generic decompression function
pub fn inflate(gpa: Allocator, data: []const u8, limit: std.Io.Limit, container: flate.Container) ![]u8 {
    var in_stream: std.Io.Reader = .fixed(data);
    const buffer = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(buffer);

    var decompressor: flate.Decompress = .init(&in_stream, container, buffer);

    var aw: std.Io.Writer.Allocating = .init(gpa);
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
        var dummy_writer: std.Io.Writer = .fixed(&one_byte_buf);
        if (decompressor.reader.stream(&dummy_writer, .limited(1))) |n| {
            if (n > 0) return error.OutputLimitExceeded;
        } else |err| switch (err) {
            error.EndOfStream => {},
            else => |e| return e,
        }
    }

    return aw.toOwnedSlice();
}

fn applyStrategy(strategy: Strategy) flate.Compress.Options {
    return switch (strategy) {
        .default => |opts| opts,
        .filtered => |opts| blk: {
            var new = opts;
            new.chain = @min(new.chain, 16);
            new.nice = @min(new.nice, 32);
            break :blk new;
        },
        .rle => |opts| blk: {
            var new = opts;
            new.chain = @min(new.chain, 8);
            break :blk new;
        },
        .huffman_only => unreachable,
    };
}
