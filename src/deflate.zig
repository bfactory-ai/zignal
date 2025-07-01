//! Pure Zig implementation of DEFLATE compression and decompression (RFC 1951)
//! Used by PNG for IDAT chunk compression/decompression

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// Fixed Huffman code lengths for literal/length alphabet (RFC 1951)
const FIXED_LITERAL_LENGTHS = [_]u8{
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 0-15
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 16-31
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 32-47
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 48-63
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 64-79
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 80-95
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 96-111
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 112-127
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 128-143
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 144-159
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 160-175
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 176-191
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 192-207
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 208-223
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 224-239
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 240-255
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 256-271
    7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, // 272-287
};

// Fixed distance code lengths (all 5 bits)
const FIXED_DISTANCE_LENGTHS = [_]u8{5} ** 32;

// Length codes 257-285 (extra bits and base lengths)
const LENGTH_CODES = [_]struct { base: u16, extra_bits: u8 }{
    .{ .base = 3, .extra_bits = 0 },   // 257
    .{ .base = 4, .extra_bits = 0 },   // 258
    .{ .base = 5, .extra_bits = 0 },   // 259
    .{ .base = 6, .extra_bits = 0 },   // 260
    .{ .base = 7, .extra_bits = 0 },   // 261
    .{ .base = 8, .extra_bits = 0 },   // 262
    .{ .base = 9, .extra_bits = 0 },   // 263
    .{ .base = 10, .extra_bits = 0 },  // 264
    .{ .base = 11, .extra_bits = 1 },  // 265
    .{ .base = 13, .extra_bits = 1 },  // 266
    .{ .base = 15, .extra_bits = 1 },  // 267
    .{ .base = 17, .extra_bits = 1 },  // 268
    .{ .base = 19, .extra_bits = 2 },  // 269
    .{ .base = 23, .extra_bits = 2 },  // 270
    .{ .base = 27, .extra_bits = 2 },  // 271
    .{ .base = 31, .extra_bits = 2 },  // 272
    .{ .base = 35, .extra_bits = 3 },  // 273
    .{ .base = 43, .extra_bits = 3 },  // 274
    .{ .base = 51, .extra_bits = 3 },  // 275
    .{ .base = 59, .extra_bits = 3 },  // 276
    .{ .base = 67, .extra_bits = 4 },  // 277
    .{ .base = 83, .extra_bits = 4 },  // 278
    .{ .base = 99, .extra_bits = 4 },  // 279
    .{ .base = 115, .extra_bits = 4 }, // 280
    .{ .base = 131, .extra_bits = 5 }, // 281
    .{ .base = 163, .extra_bits = 5 }, // 282
    .{ .base = 195, .extra_bits = 5 }, // 283
    .{ .base = 227, .extra_bits = 5 }, // 284
    .{ .base = 258, .extra_bits = 0 }, // 285
};

// Distance codes 0-29 (extra bits and base distances)
const DISTANCE_CODES = [_]struct { base: u16, extra_bits: u8 }{
    .{ .base = 1, .extra_bits = 0 },     // 0
    .{ .base = 2, .extra_bits = 0 },     // 1
    .{ .base = 3, .extra_bits = 0 },     // 2
    .{ .base = 4, .extra_bits = 0 },     // 3
    .{ .base = 5, .extra_bits = 1 },     // 4
    .{ .base = 7, .extra_bits = 1 },     // 5
    .{ .base = 9, .extra_bits = 2 },     // 6
    .{ .base = 13, .extra_bits = 2 },    // 7
    .{ .base = 17, .extra_bits = 3 },    // 8
    .{ .base = 25, .extra_bits = 3 },    // 9
    .{ .base = 33, .extra_bits = 4 },    // 10
    .{ .base = 49, .extra_bits = 4 },    // 11
    .{ .base = 65, .extra_bits = 5 },    // 12
    .{ .base = 97, .extra_bits = 5 },    // 13
    .{ .base = 129, .extra_bits = 6 },   // 14
    .{ .base = 193, .extra_bits = 6 },   // 15
    .{ .base = 257, .extra_bits = 7 },   // 16
    .{ .base = 385, .extra_bits = 7 },   // 17
    .{ .base = 513, .extra_bits = 8 },   // 18
    .{ .base = 769, .extra_bits = 8 },   // 19
    .{ .base = 1025, .extra_bits = 9 },  // 20
    .{ .base = 1537, .extra_bits = 9 },  // 21
    .{ .base = 2049, .extra_bits = 10 }, // 22
    .{ .base = 3073, .extra_bits = 10 }, // 23
    .{ .base = 4097, .extra_bits = 11 }, // 24
    .{ .base = 6145, .extra_bits = 11 }, // 25
    .{ .base = 8193, .extra_bits = 12 }, // 26
    .{ .base = 12289, .extra_bits = 12 }, // 27
    .{ .base = 16385, .extra_bits = 13 }, // 28
    .{ .base = 24577, .extra_bits = 13 }, // 29
};

// Huffman tree node
const HuffmanNode = struct {
    symbol: ?u16 = null, // null for internal nodes
    left: ?*HuffmanNode = null,
    right: ?*HuffmanNode = null,
};

// Huffman decoder table for faster decoding
const HuffmanDecoder = struct {
    // Fast lookup table for common short codes
    fast_table: [512]u16 = [_]u16{0} ** 512, // 9-bit lookup
    fast_mask: u16 = 511, // 2^9 - 1
    
    // Tree for longer codes
    root: ?*HuffmanNode = null,
    allocator: Allocator,
    nodes: ArrayList(HuffmanNode),

    pub fn init(allocator: Allocator) HuffmanDecoder {
        return .{
            .allocator = allocator,
            .nodes = ArrayList(HuffmanNode).init(allocator),
        };
    }

    pub fn deinit(self: *HuffmanDecoder) void {
        self.nodes.deinit();
    }

    pub fn buildFromLengths(self: *HuffmanDecoder, code_lengths: []const u8) !void {
        // Count codes of each length
        var length_count = [_]u16{0} ** 16;
        for (code_lengths) |len| {
            if (len > 0) length_count[len] += 1;
        }

        // Calculate first code for each length
        var code: u16 = 0;
        var first_code = [_]u16{0} ** 16;
        for (1..16) |bits| {
            code = (code + length_count[bits - 1]) << 1;
            first_code[bits] = code;
        }

        // Build fast lookup table and tree
        for (code_lengths, 0..) |len, symbol| {
            if (len == 0) continue;

            const sym_code = first_code[len];
            first_code[len] += 1;

            if (len <= 9) {
                // Add to fast table with all possible suffixes
                const num_entries = @as(u16, 1) << @intCast(9 - len);
                var i: u16 = 0;
                while (i < num_entries) : (i += 1) {
                    const table_index = sym_code | (i << @intCast(len));
                    self.fast_table[table_index] = @as(u16, @intCast(symbol)) | (@as(u16, @intCast(len)) << 12);
                }
            } else {
                // Add to tree for longer codes
                if (self.root == null) {
                    try self.nodes.append(.{});
                    self.root = &self.nodes.items[self.nodes.items.len - 1];
                }

                var current = self.root.?;
                var bit_pos: u8 = @intCast(len - 1);
                while (bit_pos > 0) : (bit_pos -= 1) {
                    const bit = (sym_code >> @as(u4, @intCast(bit_pos))) & 1;
                    if (bit == 0) {
                        if (current.left == null) {
                            try self.nodes.append(.{});
                            current.left = &self.nodes.items[self.nodes.items.len - 1];
                        }
                        current = current.left.?;
                    } else {
                        if (current.right == null) {
                            try self.nodes.append(.{});
                            current.right = &self.nodes.items[self.nodes.items.len - 1];
                        }
                        current = current.right.?;
                    }
                }
                current.symbol = @intCast(symbol);
            }
        }
    }
};

// Bit stream reader for deflate data
const BitReader = struct {
    data: []const u8,
    byte_pos: usize = 0,
    bit_pos: u8 = 0, // 0-7, position within current byte

    pub fn init(data: []const u8) BitReader {
        return .{ .data = data };
    }

    pub fn readBits(self: *BitReader, num_bits: u8) !u32 {
        assert(num_bits <= 32);
        
        var result: u32 = 0;
        var bits_read: u8 = 0;

        while (bits_read < num_bits) {
            if (self.byte_pos >= self.data.len) {
                return error.UnexpectedEndOfData;
            }

            const current_byte = self.data[self.byte_pos];
            const bits_in_byte = 8 - self.bit_pos;
            const bits_needed = num_bits - bits_read;
            const bits_to_read = @min(bits_needed, bits_in_byte);

            const mask = (@as(u8, 1) << @as(u3, @intCast(bits_to_read))) - 1;
            const bits = (current_byte >> @as(u3, @intCast(self.bit_pos))) & mask;
            result |= @as(u32, bits) << @as(u5, @intCast(bits_read));

            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if (self.bit_pos >= 8) {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        return result;
    }

    pub fn skipToByteBoundary(self: *BitReader) void {
        if (self.bit_pos != 0) {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    pub fn readBytes(self: *BitReader, buffer: []u8) !void {
        self.skipToByteBoundary();
        if (self.byte_pos + buffer.len > self.data.len) {
            return error.UnexpectedEndOfData;
        }
        @memcpy(buffer, self.data[self.byte_pos..self.byte_pos + buffer.len]);
        self.byte_pos += buffer.len;
    }
};

// DEFLATE decompressor
pub const DeflateDecoder = struct {
    allocator: Allocator,
    output: ArrayList(u8),
    literal_decoder: HuffmanDecoder,
    distance_decoder: HuffmanDecoder,

    pub fn init(allocator: Allocator) DeflateDecoder {
        return .{
            .allocator = allocator,
            .output = ArrayList(u8).init(allocator),
            .literal_decoder = HuffmanDecoder.init(allocator),
            .distance_decoder = HuffmanDecoder.init(allocator),
        };
    }

    pub fn deinit(self: *DeflateDecoder) void {
        self.output.deinit();
        self.literal_decoder.deinit();
        self.distance_decoder.deinit();
    }

    pub fn decode(self: *DeflateDecoder, compressed_data: []const u8) !ArrayList(u8) {
        var reader = BitReader.init(compressed_data);
        
        while (true) {
            const is_final = try reader.readBits(1) == 1;
            const block_type = try reader.readBits(2);

            switch (block_type) {
                0 => try self.decodeUncompressedBlock(&reader),
                1 => try self.decodeFixedHuffmanBlock(&reader),
                2 => try self.decodeDynamicHuffmanBlock(&reader),
                3 => return error.InvalidBlockType,
                else => unreachable,
            }

            if (is_final) break;
        }

        return self.output.clone();
    }

    fn decodeUncompressedBlock(self: *DeflateDecoder, reader: *BitReader) !void {
        reader.skipToByteBoundary();
        
        var len_bytes: [2]u8 = undefined;
        var nlen_bytes: [2]u8 = undefined;
        try reader.readBytes(&len_bytes);
        try reader.readBytes(&nlen_bytes);

        const len = std.mem.readInt(u16, len_bytes[0..2], .little);
        const nlen = std.mem.readInt(u16, nlen_bytes[0..2], .little);

        if (len != ~nlen) {
            return error.InvalidUncompressedBlock;
        }

        const old_len = self.output.items.len;
        try self.output.resize(old_len + len);
        try reader.readBytes(self.output.items[old_len..]);
    }

    fn decodeFixedHuffmanBlock(self: *DeflateDecoder, reader: *BitReader) !void {
        // Build fixed Huffman tables
        try self.literal_decoder.buildFromLengths(&FIXED_LITERAL_LENGTHS);
        try self.distance_decoder.buildFromLengths(&FIXED_DISTANCE_LENGTHS);
        
        try self.decodeHuffmanBlock(reader);
    }

    fn decodeDynamicHuffmanBlock(self: *DeflateDecoder, reader: *BitReader) !void {
        const hlit = try reader.readBits(5) + 257; // # of literal/length codes
        const hdist = try reader.readBits(5) + 1;  // # of distance codes  
        const hclen = try reader.readBits(4) + 4;  // # of code length codes

        // Code length code order
        const code_length_order = [_]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
        
        // Read code length codes
        var code_length_lengths = [_]u8{0} ** 19;
        for (0..hclen) |i| {
            code_length_lengths[code_length_order[i]] = @intCast(try reader.readBits(3));
        }

        // Build code length decoder
        var code_length_decoder = HuffmanDecoder.init(self.allocator);
        defer code_length_decoder.deinit();
        try code_length_decoder.buildFromLengths(&code_length_lengths);

        // Decode literal/length and distance code lengths
        var lengths = try self.allocator.alloc(u8, hlit + hdist);
        defer self.allocator.free(lengths);
        
        var i: usize = 0;
        while (i < lengths.len) {
            const symbol = try self.decodeSymbol(reader, &code_length_decoder);
            
            if (symbol < 16) {
                lengths[i] = @intCast(symbol);
                i += 1;
            } else if (symbol == 16) {
                if (i == 0) return error.InvalidCodeLength;
                const repeat_count = try reader.readBits(2) + 3;
                const repeat_value = lengths[i - 1];
                for (0..repeat_count) |_| {
                    if (i >= lengths.len) return error.InvalidCodeLength;
                    lengths[i] = repeat_value;
                    i += 1;
                }
            } else if (symbol == 17) {
                const repeat_count = try reader.readBits(3) + 3;
                for (0..repeat_count) |_| {
                    if (i >= lengths.len) return error.InvalidCodeLength;
                    lengths[i] = 0;
                    i += 1;
                }
            } else if (symbol == 18) {
                const repeat_count = try reader.readBits(7) + 11;
                for (0..repeat_count) |_| {
                    if (i >= lengths.len) return error.InvalidCodeLength;
                    lengths[i] = 0;
                    i += 1;
                }
            } else {
                return error.InvalidCodeLengthSymbol;
            }
        }

        // Build literal/length and distance decoders
        try self.literal_decoder.buildFromLengths(lengths[0..hlit]);
        try self.distance_decoder.buildFromLengths(lengths[hlit..hlit + hdist]);
        
        try self.decodeHuffmanBlock(reader);
    }

    fn decodeHuffmanBlock(self: *DeflateDecoder, reader: *BitReader) !void {
        while (true) {
            const symbol = try self.decodeSymbol(reader, &self.literal_decoder);
            
            if (symbol < 256) {
                // Literal byte
                try self.output.append(@intCast(symbol));
            } else if (symbol == 256) {
                // End of block
                break;
            } else if (symbol <= 285) {
                // Length/distance pair
                const length_code = symbol - 257;
                if (length_code >= LENGTH_CODES.len) {
                    return error.InvalidLengthCode;
                }
                
                const length_info = LENGTH_CODES[length_code];
                const length = length_info.base + try reader.readBits(length_info.extra_bits);
                
                const distance_symbol = try self.decodeSymbol(reader, &self.distance_decoder);
                if (distance_symbol >= DISTANCE_CODES.len) {
                    return error.InvalidDistanceCode;
                }
                
                const distance_info = DISTANCE_CODES[distance_symbol];
                const distance = distance_info.base + try reader.readBits(distance_info.extra_bits);
                
                // Copy from sliding window
                const start_pos = self.output.items.len - distance;
                for (0..length) |j| {
                    const byte = self.output.items[start_pos + (j % distance)];
                    try self.output.append(byte);
                }
            } else {
                return error.InvalidLiteralLengthSymbol;
            }
        }
    }

    fn decodeSymbol(self: *DeflateDecoder, reader: *BitReader, decoder: *HuffmanDecoder) !u16 {
        _ = self; // unused parameter
        
        // Try fast lookup first (simplified for now)
        const remaining_bits = (reader.data.len - reader.byte_pos) * 8 - reader.bit_pos;
        if (remaining_bits >= 9) {
            // Read bits for fast lookup
            var peek_value: u16 = 0;
            var temp_byte_pos = reader.byte_pos;
            var temp_bit_pos = reader.bit_pos;
            
            for (0..9) |i| {
                if (temp_byte_pos >= reader.data.len) break;
                const bit = (reader.data[temp_byte_pos] >> @as(u3, @intCast(temp_bit_pos))) & 1;
                peek_value |= @as(u16, bit) << @intCast(i);
                temp_bit_pos += 1;
                if (temp_bit_pos >= 8) {
                    temp_bit_pos = 0;
                    temp_byte_pos += 1;
                }
            }
            
            const entry = decoder.fast_table[peek_value & decoder.fast_mask];
            if (entry != 0) {
                const symbol = entry & 0xFFF;
                const code_length: u8 = @intCast((entry >> 12) & 0xF);
                // Advance reader by code_length bits
                _ = try reader.readBits(code_length);
                return symbol;
            }
        }

        // Fall back to tree traversal for longer codes
        if (decoder.root) |root| {
            var current = root;
            while (current.symbol == null) {
                const bit = try reader.readBits(1);
                current = if (bit == 0) current.left.? else current.right.?;
            }
            return current.symbol.?;
        }

        return error.InvalidHuffmanCode;
    }
};

// Public decompression function
pub fn inflate(allocator: Allocator, compressed_data: []const u8) ![]u8 {
    var decoder = DeflateDecoder.init(allocator);
    defer decoder.deinit();
    
    var result = try decoder.decode(compressed_data);
    defer result.deinit();
    
    return result.toOwnedSlice();
}

// DEFLATE encoder for PNG compression
pub const DeflateEncoder = struct {
    allocator: Allocator,
    output: ArrayList(u8),
    
    pub fn init(allocator: Allocator) DeflateEncoder {
        return .{
            .allocator = allocator,
            .output = ArrayList(u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *DeflateEncoder) void {
        self.output.deinit();
    }
    
    pub fn encode(self: *DeflateEncoder, data: []const u8) !ArrayList(u8) {
        // For PNG, we'll use a simple deflate implementation
        // Write final block with no compression (type 00)
        
        const block_size = @min(data.len, 65535); // Max uncompressed block size
        var pos: usize = 0;
        
        while (pos < data.len) {
            const remaining = data.len - pos;
            const chunk_size = @min(remaining, block_size);
            const is_final = (pos + chunk_size >= data.len);
            
            // Block header: BFINAL (1 bit) + BTYPE (2 bits) = 000 or 001 for final
            const block_header: u8 = if (is_final) 0x01 else 0x00;
            try self.output.append(block_header);
            
            // Length and NLEN (one's complement of length)
            const len: u16 = @intCast(chunk_size);
            const nlen: u16 = ~len;
            
            try self.output.appendSlice(std.mem.asBytes(&len));
            try self.output.appendSlice(std.mem.asBytes(&nlen));
            
            // Uncompressed data
            try self.output.appendSlice(data[pos..pos + chunk_size]);
            
            pos += chunk_size;
        }
        
        return self.output.clone();
    }
};

// Public compression function (simple implementation for PNG)
pub fn deflate(allocator: Allocator, data: []const u8) ![]u8 {
    var encoder = DeflateEncoder.init(allocator);
    defer encoder.deinit();
    
    var result = try encoder.encode(data);
    defer result.deinit();
    
    return result.toOwnedSlice();
}

// Basic test
test "deflate decompression" {
    // This is a basic test to ensure the module compiles
    const allocator = std.testing.allocator;
    
    // Test with empty data should fail gracefully
    const empty_data = [_]u8{};
    const result = inflate(allocator, &empty_data);
    try std.testing.expectError(error.UnexpectedEndOfData, result);
}

test "deflate round-trip compression" {
    const allocator = std.testing.allocator;
    
    // Test data
    const original_data = "Hello, World! This is a test string for deflate compression.";
    
    // Compress
    const compressed = try deflate(allocator, original_data);
    defer allocator.free(compressed);
    
    // Decompress
    const decompressed = try inflate(allocator, compressed);
    defer allocator.free(decompressed);
    
    // Verify
    try std.testing.expectEqualSlices(u8, original_data, decompressed);
}