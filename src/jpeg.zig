//! Pure Zig JPEG decoder implementation.
//! Supports baseline DCT JPEG images with common chroma subsampling.
//! Zero dependencies - implements all required algorithms internally.

const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

const Image = @import("image.zig").Image;
const Rgb = @import("color.zig").Rgb;
const Rgba = @import("color.zig").Rgba;
const convertColor = @import("color.zig").convertColor;

// JPEG markers
const Marker = enum(u16) {
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
    SOI = 0xFFD8,  // Start of Image
    EOI = 0xFFD9,  // End of Image
    SOS = 0xFFDA,  // Start of Scan
    DQT = 0xFFDB,  // Quantization Table
    DNL = 0xFFDC,  // Number of Lines
    DRI = 0xFFDD,  // Restart Interval
    DHP = 0xFFDE,  // Hierarchical Progression
    EXP = 0xFFDF,  // Expand Reference Component
    
    // Application segments
    APP0 = 0xFFE0,  // JFIF
    APP1 = 0xFFE1,  // EXIF
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

// JPEG decoder state
pub const JpegDecoder = struct {
    allocator: Allocator,
    
    // Image properties
    width: u16,
    height: u16,
    num_components: u8,
    components: [4]Component = undefined,
    
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
    
    // MCU (Minimum Coded Unit) dimensions
    mcu_width: u16 = 0,
    mcu_height: u16 = 0,
    mcu_width_in_blocks: u16 = 0,
    mcu_height_in_blocks: u16 = 0,
    
    pub fn init(allocator: Allocator) JpegDecoder {
        return .{ 
            .allocator = allocator,
            .width = 0,
            .height = 0,
            .num_components = 0,
            .scan_components = &[_]ScanComponent{},
        };
    }
    
    pub fn deinit(self: *JpegDecoder) void {
        for (&self.dc_tables) |*table| {
            if (table.*) |*t| t.deinit(self.allocator);
        }
        for (&self.ac_tables) |*table| {
            if (table.*) |*t| t.deinit(self.allocator);
        }
        if (self.scan_components.len > 0) {
            self.allocator.free(self.scan_components);
        }
    }
    
    // Parse Start of Frame (SOF0) marker
    pub fn parseSOF(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 8) return error.InvalidSOF;
        
        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidSOF;
        
        const precision = data[2];
        if (precision != 8) return error.UnsupportedJpegFormat; // Only 8-bit precision supported
        
        self.height = (@as(u16, data[3]) << 8) | data[4];
        self.width = (@as(u16, data[5]) << 8) | data[6];
        self.num_components = data[7];
        
        if (self.num_components != 1 and self.num_components != 3) {
            return error.InvalidComponentCount;
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
        
        // Calculate MCU dimensions
        if (max_h_sampling > 4 or max_v_sampling > 4) {
            return error.InvalidComponentCount;
        }
        self.mcu_width = @as(u16, max_h_sampling) * 8;
        self.mcu_height = @as(u16, max_v_sampling) * 8;
        self.mcu_width_in_blocks = (self.width + self.mcu_width - 1) / self.mcu_width;
        self.mcu_height_in_blocks = (self.height + self.mcu_height - 1) / self.mcu_height;
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
            const table_id = table_info & 0x0F;
            
            if (table_id > 3) return error.InvalidHuffmanTable;
            
            pos += 1;
            
            // Read 16 bytes of bit lengths
            var bits: [16]u8 = undefined;
            @memcpy(&bits, data[pos..pos + 16]);
            pos += 16;
            
            // Count total number of codes
            var total_codes: u16 = 0;
            for (bits) |count| {
                total_codes += count;
            }
            
            if (pos + total_codes > length) return error.InvalidDHT;
            
            // Allocate and read huffman values
            const huffval = try self.allocator.alloc(u8, total_codes);
            @memcpy(huffval, data[pos..pos + total_codes]);
            pos += total_codes;
            
            // Build Huffman table
            var table = HuffmanTable{
                .bits = bits,
                .huffval = huffval,
                .mincode = undefined,
                .maxcode = undefined,
                .valptr = undefined,
            };
            
            // Generate decoding tables
            try generateHuffmanTables(&table);
            
            // Store table
            if (table_class == 0) {
                if (self.dc_tables[table_id]) |*old_table| {
                    old_table.deinit(self.allocator);
                }
                self.dc_tables[table_id] = table;
            } else {
                if (self.ac_tables[table_id]) |*old_table| {
                    old_table.deinit(self.allocator);
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
            const table_id = table_info & 0x0F;
            
            if (table_id > 3) return error.InvalidQuantTable;
            
            pos += 1;
            
            const element_size: usize = if (precision == 0) 1 else 2;
            if (pos + 64 * element_size > length) return error.InvalidDQT;
            
            // Read quantization table
            var table: [64]u16 = undefined;
            
            if (precision == 0) {
                // 8-bit values - stored in natural order in the file
                for (0..64) |i| {
                    table[i] = data[pos + i];
                }
                pos += 64;
            } else {
                // 16-bit values - stored in natural order in the file  
                for (0..64) |i| {
                    table[i] = (@as(u16, data[pos + i * 2]) << 8) | data[pos + i * 2 + 1];
                }
                pos += 128;
            }
            
            self.quant_tables[table_id] = table;
        }
    }

    // Parse Start of Scan (SOS) marker
    pub fn parseSOS(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 6) return error.InvalidSOS;
        
        const length = (@as(u16, data[0]) << 8) | data[1];
        if (data.len < length) return error.InvalidSOS;
        
        const num_components = data[2];
        if (num_components != self.num_components) return error.InvalidSOS;
        
        self.scan_components = try self.allocator.alloc(ScanComponent, num_components);
        
        var pos: usize = 3;
        for (0..num_components) |i| {
            if (pos + 2 > length) return error.InvalidSOS;
            
            self.scan_components[i] = .{
                .component_id = data[pos],
                .dc_table_id = @intCast(data[pos + 1] >> 4),
                .ac_table_id = @intCast(data[pos + 1] & 0x0F),
            };
            
            pos += 2;
        }
        
        // Skip spectral selection and successive approximation (should be 0, 63, 0 for baseline)
        if (pos + 3 > length) return error.InvalidSOS;
    }

    // Parse Define Restart Interval (DRI) marker
    pub fn parseDRI(self: *JpegDecoder, data: []const u8) !void {
        if (data.len < 4) return error.InvalidDRI;
        
        const length = (@as(u16, data[0]) << 8) | data[1];
        if (length != 4) return error.InvalidDRI;
        
        self.restart_interval = (@as(u16, data[2]) << 8) | data[3];
    }

    // Decode a Huffman symbol
    pub fn decodeHuffmanSymbol(self: *JpegDecoder, table: *const HuffmanTable) !u8 {
        var code: i32 = 0;
        
        for (0..16) |i| {
            const bit = try self.bit_reader.getBits(1);
            code = (code << 1) | @as(i32, bit);
            
            if (code <= table.maxcode[i]) {
                if (table.valptr[i] == -1) return error.InvalidHuffmanCode;
                const index = @as(usize, @intCast(table.valptr[i] + (code - table.mincode[i])));
                if (index >= table.huffval.len) return error.InvalidHuffmanCode;
                return table.huffval[index];
            }
        }
        
        return error.InvalidHuffmanCode;
    }

    // Decode DC coefficient with overflow protection
    pub fn decodeDC(self: *JpegDecoder, table: *const HuffmanTable, prev_dc: i64) !i64 {
        const symbol = try self.decodeHuffmanSymbol(table);
        
        if (symbol == 0) {
            return prev_dc;
        }
        
        if (symbol > 11) return error.InvalidDCValue;
        
        const bits = try self.bit_reader.getBits(@intCast(symbol));
        var diff: i64 = @intCast(bits);
        
        // Convert from unsigned to signed
        if (bits < (@as(u32, 1) << @intCast(symbol - 1))) {
            diff = @as(i64, @intCast(bits)) - @as(i64, @intCast((@as(u32, 1) << @intCast(symbol)) - 1));
        }
        
        return prev_dc + diff;
    }

    // Decode AC coefficients
    pub fn decodeAC(self: *JpegDecoder, table: *const HuffmanTable, block: *[64]i32) !void {
        var k: usize = 1; // Start after DC coefficient
        
        while (k < 64) {
            const symbol = try self.decodeHuffmanSymbol(table);
            
            if (symbol == 0) {
                // End of block
                while (k < 64) {
                    block[ZIGZAG_ORDER[k]] = 0;
                    k += 1;
                }
                return;
            }
            
            const run = symbol >> 4;
            const size = symbol & 0x0F;
            
            if (size == 0) {
                if (run != 15) return error.InvalidACValue;
                // Skip 16 zeros
                for (0..16) |_| {
                    if (k >= 64) return error.InvalidACValue;
                    block[ZIGZAG_ORDER[k]] = 0;
                    k += 1;
                }
            } else {
                // Skip 'run' zeros
                for (0..run) |_| {
                    if (k >= 64) return error.InvalidACValue;
                    block[ZIGZAG_ORDER[k]] = 0;
                    k += 1;
                }
                
                if (k >= 64) return error.InvalidACValue;
                
                // Decode AC value
                const bits = try self.bit_reader.getBits(@intCast(size));
                var value: i32 = @intCast(bits);
                
                // Convert from unsigned to signed
                if (bits < (@as(u32, 1) << @intCast(size - 1))) {
                    value = @as(i32, @intCast(bits)) - @as(i32, @intCast((@as(u32, 1) << @intCast(size)) - 1));
                }
                
                block[ZIGZAG_ORDER[k]] = value;
                k += 1;
            }
        }
    }

    // Decode a single 8x8 block
    pub fn decodeBlock(self: *JpegDecoder, component_idx: usize, prev_dc: *i64) ![64]i32 {
        var block: [64]i32 = .{0} ** 64;
        
        // Find which scan component corresponds to this component
        var scan_comp_idx: usize = 0;
        for (self.scan_components, 0..) |scan_comp, i| {
            if (scan_comp.component_id == self.components[component_idx].id) {
                scan_comp_idx = i;
                break;
            }
        }
        
        const scan_comp = self.scan_components[scan_comp_idx];
        
        // Decode DC coefficient
        const dc_table = self.dc_tables[scan_comp.dc_table_id] orelse return error.MissingHuffmanTable;
        prev_dc.* = try self.decodeDC(&dc_table, prev_dc.*);
        block[0] = @intCast(std.math.clamp(prev_dc.*, std.math.minInt(i32), std.math.maxInt(i32)));
        
        // Decode AC coefficients
        const ac_table = self.ac_tables[scan_comp.ac_table_id] orelse return error.MissingHuffmanTable;
        try self.decodeAC(&ac_table, &block);
        
        // Dequantize with overflow protection
        const quant_table = self.quant_tables[self.components[component_idx].quant_table_id] orelse return error.MissingQuantTable;
        for (0..64) |i| {
            const coeff = block[i];
            const quant_val = @as(i32, @intCast(quant_table[i]));
            
            // Perform multiplication with overflow checking
            if (coeff != 0) {
                const result = @as(i64, coeff) * @as(i64, quant_val);
                // Clamp to i32 range to prevent overflow
                if (result > std.math.maxInt(i32)) {
                    block[i] = std.math.maxInt(i32);
                } else if (result < std.math.minInt(i32)) {
                    block[i] = std.math.minInt(i32);
                } else {
                    block[i] = @intCast(result);
                }
            }
        }
        
        return block;
    }

    // Decode MCU (Minimum Coded Unit) with proper interleaved block order
    pub fn decodeMCU(self: *JpegDecoder, mcu_data: *[4][][64]i32, dc_values: []i64) !void {
        // Calculate total number of blocks in MCU for interleaved decoding
        var total_blocks: usize = 0;
        var component_block_counts: [4]usize = .{0} ** 4;
        for (0..self.num_components) |comp_idx| {
            const comp = self.components[comp_idx];
            const blocks_count = @as(usize, comp.h_sampling) * comp.v_sampling;
            component_block_counts[comp_idx] = blocks_count;
            total_blocks += blocks_count;
        }
        
        // Create interleaved block order
        var block_order: [64]struct { comp_idx: usize, block_idx: usize } = undefined; // Max possible blocks in MCU
        var order_idx: usize = 0;
        
        // Standard JPEG interleaved order: cycle through components for each position
        for (0..self.num_components) |comp_idx| {
            for (0..component_block_counts[comp_idx]) |block_idx| {
                block_order[order_idx] = .{ .comp_idx = comp_idx, .block_idx = block_idx };
                order_idx += 1;
            }
        }
        
        // Decode blocks in interleaved order
        for (0..total_blocks) |i| {
            const comp_idx = block_order[i].comp_idx;
            const block_idx = block_order[i].block_idx;
            
            var block = try self.decodeBlock(comp_idx, &dc_values[comp_idx]);
            
            // Apply IDCT
            idct8x8(&block);
            
            // Level shift and clamp: IDCT produces [-128,127], we need [0,255]
            // Add 128 and clamp to valid range
            for (&block) |*val| {
                const shifted = val.* + 128;
                val.* = std.math.clamp(shifted, 0, 255);
            }
            
            // Store the decoded block
            mcu_data[comp_idx][block_idx] = block;
        }
    }
};

// Huffman table for decoding
const HuffmanTable = struct {
    // Number of codes for each bit length (1-16)
    bits: [16]u8,
    // Huffman values in order
    huffval: []u8,
    // Decoding tables
    mincode: [16]i32,
    maxcode: [16]i32,
    valptr: [16]i32,
    
    pub fn deinit(self: *HuffmanTable, allocator: Allocator) void {
        allocator.free(self.huffval);
    }
};

// Bit reader for entropy-coded segments
const BitReader = struct {
    data: []const u8,
    byte_pos: usize = 0,
    bit_pos: u3 = 0,
    bits_left: u32 = 0,
    bit_buffer: u32 = 0,
    
    pub fn init(data: []const u8) BitReader {
        return .{ .data = data };
    }
    
    pub fn resetBitBuffer(self: *BitReader) void {
        self.bit_buffer = 0;
        self.bits_left = 0;
    }
    
    pub fn getBits(self: *BitReader, n: u5) !u16 {
        while (self.bits_left < n) {
            if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
            
            const byte = self.data[self.byte_pos];
            self.byte_pos += 1;
            
            // Handle 0xFF byte stuffing - 0xFF 0x00 sequence represents actual 0xFF data
            if (byte == 0xFF) {
                if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
                const next = self.data[self.byte_pos];
                if (next == 0x00) {
                    // This is stuffed 0xFF - skip the 0x00 and use 0xFF
                    self.byte_pos += 1;
                    // byte remains 0xFF
                } else if (next >= 0xD0 and next <= 0xD7) {
                    // Restart marker - we've consumed the FF, now consume the marker byte
                    self.byte_pos += 1;
                    // Reset bit buffer state  
                    self.bit_buffer = 0;
                    self.bits_left = 0;
                    continue;
                } else {
                    // This is another marker - should not happen in entropy-coded data
                    return error.InvalidByteStuffing;
                }
            }
            
            self.bit_buffer = (self.bit_buffer << 8) | byte;
            self.bits_left += 8;
        }
        
        self.bits_left -= n;
        const result = @as(u16, @intCast((self.bit_buffer >> @intCast(self.bits_left)) & ((@as(u32, 1) << n) - 1)));
        return result;
    }
    
    pub fn peekBits(self: *BitReader, n: u5) !u16 {
        const saved_byte_pos = self.byte_pos;
        const saved_bits_left = self.bits_left;
        const saved_bit_buffer = self.bit_buffer;
        
        const result = try self.getBits(n);
        
        self.byte_pos = saved_byte_pos;
        self.bits_left = saved_bits_left;
        self.bit_buffer = saved_bit_buffer;
        
        return result;
    }
};

// Zigzag scan order for 8x8 blocks
const ZIGZAG_ORDER = [64]u8{
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
   10, 19, 23, 32, 39, 45, 52, 54,
   20, 22, 33, 38, 46, 51, 55, 60,
   21, 34, 37, 47, 50, 56, 59, 61,
   35, 36, 48, 49, 57, 58, 62, 63
};

// Parse JPEG file and decode image
pub fn decode(allocator: Allocator, data: []const u8) !JpegDecoder {
    var decoder = JpegDecoder.init(allocator);
    errdefer decoder.deinit();
    
    // Check for JPEG SOI marker
    if (data.len < 2 or data[0] != 0xFF or data[1] != 0xD8) {
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
            const length = (@as(u16, data[pos]) << 8) | data[pos + 1];
            if (length < 2) return error.InvalidMarker; // Invalid length
            pos += length;
            continue;
        };
        
        switch (marker) {
            .SOI => {
                pos += 2;
                continue; // Already handled
            },
            .EOI => break,
            .SOF0 => {
                try decoder.parseSOF(data[pos + 2..]);
                // parseSOF will handle advancing position internally
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            },
            .DHT => {
                try decoder.parseDHT(data[pos + 2..]);
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            },
            .DQT => {
                try decoder.parseDQT(data[pos + 2..]);
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            },
            .SOS => {
                try decoder.parseSOS(data[pos + 2..]);
                // After SOS, we have the entropy-coded data
                const header_len = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + header_len;
                
                // Find the end of entropy-coded data (next marker)
                var scan_end = pos;
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
                
                decoder.bit_reader = BitReader.init(data[pos..scan_end]);
                return decoder;
            },
            .DRI => {
                try decoder.parseDRI(data[pos + 2..]);
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            },
            .APP0, .APP1, .APP2, .APP3, .APP4, .APP5, .APP6, .APP7,
            .APP8, .APP9, .APP10, .APP11, .APP12, .APP13, .APP14, .APP15,
            .COM => {
                // Skip these markers
                if (pos + 4 > data.len) break;
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            },
            else => {
                // Skip unknown markers with length
                if (pos + 4 > data.len) break;
                const length = (@as(u16, data[pos + 2]) << 8) | data[pos + 3];
                pos += 2 + length;
            }
        }
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
    InvalidComponentCount,
    InvalidHuffmanTable,
    InvalidQuantTable,
    NoScanData,
    UnexpectedEndOfData,
    InvalidByteStuffing,
    OutOfMemory,
    InvalidHuffmanCode,
    InvalidDCValue,
    InvalidACValue,
    MissingHuffmanTable,
    MissingQuantTable,
    NotImplemented,
};


// Generate Huffman decoding tables
fn generateHuffmanTables(table: *HuffmanTable) !void {
    var code: i32 = 0;
    var k: usize = 0;
    
    for (0..16) |i| {
        if (table.bits[i] == 0) {
            table.maxcode[i] = -1;
            table.mincode[i] = 0x7FFFFFFF;
            table.valptr[i] = -1;
        } else {
            table.valptr[i] = @intCast(k);
            table.mincode[i] = code;
            for (0..table.bits[i]) |_| {
                code += 1;
                k += 1;
            }
            table.maxcode[i] = code - 1;
        }
        code <<= 1;
    }
}

// IDCT constants - based on IJG implementation
const CONST_BITS = 13;
const CONST_SCALE = 1 << CONST_BITS; // 8192
const CONST_HALF = CONST_SCALE >> 1; // 4096
const PASS1_BITS = 2;
const PASS1_SCALE = 1 << PASS1_BITS; // 4

// Fixed-point constants for IDCT (scaled by CONST_SCALE = 8192)
// These match the IJG reference implementation
const FIX_1_414213562 = 11585; // sqrt(2) * CONST_SCALE
const FIX_1_847759065 = 15137; // (sqrt(2) + sqrt(2) * cos(3π/8)) * CONST_SCALE  
const FIX_1_082392200 = 8867;  // sqrt(2) * cos(3π/8) * CONST_SCALE
const FIX_2_613125930 = 21407; // sqrt(2) * (1 + cos(3π/8)) * CONST_SCALE

// Back to our working integer IDCT - focus on other issues
fn idct8x8(block: *[64]i32) void {
    // Pass 1: process columns
    var temp: [64]i32 = undefined;
    
    for (0..8) |i| {
        const col_offset = i;
        
        // Check if all AC terms are zero
        var ac_zero = true;
        for (1..8) |row| {
            if (block[row * 8 + col_offset] != 0) {
                ac_zero = false;
                break;
            }
        }
        
        if (ac_zero) {
            // AC terms all zero; just propagate the DC term
            const dc_val = block[col_offset] << PASS1_BITS; // Scale for pass 1
            for (0..8) |row| {
                temp[row * 8 + col_offset] = dc_val;
            }
        } else {
            // General case: do the full IDCT
            const tmp0 = block[0 * 8 + col_offset];
            const tmp1 = block[2 * 8 + col_offset];
            const tmp2 = block[4 * 8 + col_offset];
            const tmp3 = block[6 * 8 + col_offset];
            
            const tmp10 = tmp0 + tmp2;
            const tmp11 = tmp0 - tmp2;
            const tmp13 = tmp1 + tmp3;
            const tmp12 = ((tmp1 - tmp3) * FIX_1_414213562 + CONST_HALF) >> CONST_BITS;
            
            const tmp6 = tmp10 + tmp13;
            const tmp5 = tmp10 - tmp13;
            const tmp4 = tmp11 + tmp12;
            const tmp7 = tmp11 - tmp12;
            
            // Odd part
            const tmp1_odd = block[1 * 8 + col_offset];
            const tmp2_odd = block[3 * 8 + col_offset];
            const tmp3_odd = block[5 * 8 + col_offset];
            const tmp0_odd = block[7 * 8 + col_offset];
            
            const z13 = tmp1_odd + tmp3_odd;
            const z10 = tmp1_odd - tmp3_odd;
            const z11 = tmp2_odd + tmp0_odd;
            const z12 = tmp2_odd - tmp0_odd;
            
            const tmp7_odd = z11 + z13;
            const tmp11_odd = ((z11 - z13) * FIX_1_414213562 + CONST_HALF) >> CONST_BITS;
            
            const z5 = ((z10 + z12) * FIX_1_847759065 + CONST_HALF) >> CONST_BITS;
            const tmp10_odd = ((z12 * FIX_1_082392200 + CONST_HALF) >> CONST_BITS) - z5;
            const tmp12_odd = z5 - ((z10 * FIX_2_613125930 + CONST_HALF) >> CONST_BITS);
            
            const tmp6_odd = tmp12_odd - tmp7_odd;
            const tmp5_odd = tmp11_odd - tmp6_odd;
            const tmp4_odd = tmp10_odd + tmp5_odd;
            
            // Final output for column (with pass 1 scaling)
            temp[0 * 8 + col_offset] = (tmp6 + tmp7_odd) << PASS1_BITS;
            temp[1 * 8 + col_offset] = (tmp4 + tmp6_odd) << PASS1_BITS;
            temp[2 * 8 + col_offset] = (tmp5 + tmp5_odd) << PASS1_BITS;
            temp[3 * 8 + col_offset] = (tmp7 + tmp4_odd) << PASS1_BITS;
            temp[4 * 8 + col_offset] = (tmp7 - tmp4_odd) << PASS1_BITS;
            temp[5 * 8 + col_offset] = (tmp5 - tmp5_odd) << PASS1_BITS;
            temp[6 * 8 + col_offset] = (tmp4 - tmp6_odd) << PASS1_BITS;
            temp[7 * 8 + col_offset] = (tmp6 - tmp7_odd) << PASS1_BITS;
        }
    }
    
    // Pass 2: process rows
    for (0..8) |i| {
        const row_offset = i * 8;
        
        // Check if all AC terms are zero
        var ac_zero = true;
        for (1..8) |col| {
            if (temp[row_offset + col] != 0) {
                ac_zero = false;
                break;
            }
        }
        
        if (ac_zero) {
            // AC terms all zero; just propagate the DC term  
            const dc_val = (temp[row_offset] + 16) >> 5; // Match the scaling in full IDCT
            const clamped = std.math.clamp(dc_val, -128, 127);
            for (0..8) |col| {
                block[row_offset + col] = clamped;
            }
        } else {
            // General case: do the full IDCT
            const tmp0 = temp[row_offset + 0];
            const tmp1 = temp[row_offset + 2];
            const tmp2 = temp[row_offset + 4];
            const tmp3 = temp[row_offset + 6];
            
            const tmp10 = tmp0 + tmp2;
            const tmp11 = tmp0 - tmp2;
            const tmp13 = tmp1 + tmp3;
            const tmp12 = ((tmp1 - tmp3) * FIX_1_414213562 + CONST_HALF) >> CONST_BITS;
            
            const tmp6 = tmp10 + tmp13;
            const tmp5 = tmp10 - tmp13;
            const tmp4 = tmp11 + tmp12;
            const tmp7 = tmp11 - tmp12;
            
            // Odd part
            const tmp1_odd = temp[row_offset + 1];
            const tmp2_odd = temp[row_offset + 3];
            const tmp3_odd = temp[row_offset + 5];
            const tmp0_odd = temp[row_offset + 7];
            
            const z13 = tmp1_odd + tmp3_odd;
            const z10 = tmp1_odd - tmp3_odd;
            const z11 = tmp2_odd + tmp0_odd;
            const z12 = tmp2_odd - tmp0_odd;
            
            const tmp7_odd = z11 + z13;
            const tmp11_odd = ((z11 - z13) * FIX_1_414213562 + CONST_HALF) >> CONST_BITS;
            
            const z5 = ((z10 + z12) * FIX_1_847759065 + CONST_HALF) >> CONST_BITS;
            const tmp10_odd = ((z12 * FIX_1_082392200 + CONST_HALF) >> CONST_BITS) - z5;
            const tmp12_odd = z5 - ((z10 * FIX_2_613125930 + CONST_HALF) >> CONST_BITS);
            
            const tmp6_odd = tmp12_odd - tmp7_odd;
            const tmp5_odd = tmp11_odd - tmp6_odd;
            const tmp4_odd = tmp10_odd + tmp5_odd;
            
            // Final output with original scaling but corrected constants
            block[row_offset + 0] = std.math.clamp((tmp6 + tmp7_odd + 16) >> 5, -128, 127);
            block[row_offset + 1] = std.math.clamp((tmp4 + tmp6_odd + 16) >> 5, -128, 127);
            block[row_offset + 2] = std.math.clamp((tmp5 + tmp5_odd + 16) >> 5, -128, 127);
            block[row_offset + 3] = std.math.clamp((tmp7 + tmp4_odd + 16) >> 5, -128, 127);
            block[row_offset + 4] = std.math.clamp((tmp7 - tmp4_odd + 16) >> 5, -128, 127);
            block[row_offset + 5] = std.math.clamp((tmp5 - tmp5_odd + 16) >> 5, -128, 127);
            block[row_offset + 6] = std.math.clamp((tmp4 - tmp6_odd + 16) >> 5, -128, 127);
            block[row_offset + 7] = std.math.clamp((tmp6 - tmp7_odd + 16) >> 5, -128, 127);
        }
    }
}


// YCbCr to RGB conversion
fn ycbcrToRgb(y: i32, cb: i32, cr: i32) Rgb {
    // Convert from JPEG YCbCr to RGB using ITU-R BT.601 standard
    // Y, Cb, Cr are all in range [0, 255] after level shift
    const y_f = @as(f32, @floatFromInt(y));
    const cb_f = @as(f32, @floatFromInt(cb)) - 128.0;  // Center around 0
    const cr_f = @as(f32, @floatFromInt(cr)) - 128.0;  // Center around 0
    
    // Standard ITU-R BT.601 conversion
    const r_f = y_f + 1.402 * cr_f;
    const g_f = y_f - 0.344136 * cb_f - 0.714136 * cr_f;
    const b_f = y_f + 1.772 * cb_f;
    
    return Rgb{
        .r = clampU8(@intFromFloat(@round(r_f))),
        .g = clampU8(@intFromFloat(@round(g_f))),
        .b = clampU8(@intFromFloat(@round(b_f))),
    };
}

// Clamp value to u8 range
fn clampU8(value: i32) u8 {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return @intCast(value);
}

// Upsample chroma component for 4:2:0 subsampling using JPEG standard box filter
fn upsampleChroma420(input: []const [64]i32, output: *[256]i32, h_blocks: u4, v_blocks: u4, max_h: u4, max_v: u4) void {
    // For 4:2:0, input is typically 1 block (8x8), output should be max_h*8 x max_v*8
    assert(h_blocks == 1 and v_blocks == 1);
    assert(input.len == 1);
    
    const src_block = &input[0];
    const dst_width = @as(usize, max_h) * 8;
    const dst_height = @as(usize, max_v) * 8;
    
    // Box filter upsampling (pixel duplication) - JPEG standard approach
    // Each 8x8 chroma block gets upsampled to 16x16 luma resolution
    for (0..dst_height) |dst_y| {
        for (0..dst_width) |dst_x| {
            // Map destination coordinates to source coordinates
            // For 4:2:0, each chroma sample covers a 2x2 area of luma samples
            const src_y = dst_y / (dst_height / 8);
            const src_x = dst_x / (dst_width / 8);
            
            // Clamp to ensure we don't go out of bounds
            const src_y_clamped = @min(src_y, 7);
            const src_x_clamped = @min(src_x, 7);
            
            const src_idx = @as(usize, src_y_clamped) * 8 + @as(usize, src_x_clamped);
            const dst_idx = dst_y * dst_width + dst_x;
            
            // Simple pixel duplication (box filter)
            output[dst_idx] = src_block[src_idx];
        }
    }
}

// Convert MCU to pixels with proper chroma subsampling handling
fn convertMCUToPixels(comptime T: type, 
                     mcu_data: *const [4][][64]i32, 
                     components: []const Component, 
                     num_components: u8,
                     max_h: u4, 
                     max_v: u4, 
                     img: *Image(T), 
                     mcu_x: usize, 
                     mcu_y: usize, 
                     img_width: u16, 
                     img_height: u16) void {
    
    if (num_components == 1) {
        // Grayscale - simple case
        const y_blocks = mcu_data[0];
        var block_idx: usize = 0;
        
        for (0..components[0].v_sampling) |block_v| {
            for (0..components[0].h_sampling) |block_h| {
                const block = &y_blocks[block_idx];
                
                for (0..8) |y| {
                    for (0..8) |x| {
                        const px = mcu_x * (@as(usize, max_h) * 8) + block_h * 8 + x;
                        const py = mcu_y * (@as(usize, max_v) * 8) + block_v * 8 + y;
                        
                        if (px < img_width and py < img_height) {
                            const gray = clampU8(block[y * 8 + x]);
                            const rgb = Rgb{ .r = gray, .g = gray, .b = gray };
                            img.at(py, px).* = convertColor(T, rgb);
                        }
                    }
                }
                block_idx += 1;
            }
        }
    } else if (num_components == 3) {
        // Color image with potential chroma subsampling
        const y_component = &components[0];
        const cb_component = &components[1];
        const cr_component = &components[2];
        
        // Check if this is 4:2:0 subsampling
        if (y_component.h_sampling == 2 and y_component.v_sampling == 2 and
            cb_component.h_sampling == 1 and cb_component.v_sampling == 1 and
            cr_component.h_sampling == 1 and cr_component.v_sampling == 1) {
            
            // 4:2:0 subsampling - need to upsample chroma
            const mcu_pixels = @as(usize, max_h) * 8 * @as(usize, max_v) * 8;
            var cb_upsampled: [256]i32 = undefined; // Should be mcu_pixels but need compile-time size
            var cr_upsampled: [256]i32 = undefined; // Should be mcu_pixels but need compile-time size
            
            // Verify we don't exceed our fixed size
            assert(mcu_pixels <= 256);
            
            upsampleChroma420(mcu_data[1], &cb_upsampled, cb_component.h_sampling, cb_component.v_sampling, max_h, max_v);
            upsampleChroma420(mcu_data[2], &cr_upsampled, cr_component.h_sampling, cr_component.v_sampling, max_h, max_v);
            
            // Now convert Y (2x2 blocks) with upsampled chroma
            var y_block_idx: usize = 0;
            for (0..2) |block_v| { // Y has 2x2 blocks
                for (0..2) |block_h| {
                    const y_block = &mcu_data[0][y_block_idx];
                    
                    for (0..8) |y| {
                        for (0..8) |x| {
                            const mcu_width_pixels = @as(usize, max_h) * 8;
                            const mcu_height_pixels = @as(usize, max_v) * 8;
                            const px = mcu_x * mcu_width_pixels + block_h * 8 + x;
                            const py = mcu_y * mcu_height_pixels + block_v * 8 + y;
                            
                            if (px < img_width and py < img_height) {
                                const y_val = y_block[y * 8 + x];
                                
                                // Get corresponding chroma values (upsampled)
                                const chroma_idx = (block_v * 8 + y) * mcu_width_pixels + (block_h * 8 + x);
                                const cb_val = cb_upsampled[chroma_idx];
                                const cr_val = cr_upsampled[chroma_idx];
                                
                                const rgb = ycbcrToRgb(y_val, cb_val, cr_val);
                                img.at(py, px).* = convertColor(T, rgb);
                            }
                        }
                    }
                    y_block_idx += 1;
                }
            }
        } else if (y_component.h_sampling == 1 and y_component.v_sampling == 1 and
                   cb_component.h_sampling == 1 and cb_component.v_sampling == 1 and
                   cr_component.h_sampling == 1 and cr_component.v_sampling == 1) {
            
            // 4:4:4 - no subsampling, all components have same resolution
            const y_block = &mcu_data[0][0];
            const cb_block = &mcu_data[1][0]; 
            const cr_block = &mcu_data[2][0];
            
            for (0..8) |y| {
                for (0..8) |x| {
                    const px = mcu_x * 8 + x;
                    const py = mcu_y * 8 + y;
                    
                    if (px < img_width and py < img_height) {
                        const rgb = ycbcrToRgb(
                            y_block[y * 8 + x],
                            cb_block[y * 8 + x],
                            cr_block[y * 8 + x]
                        );
                        img.at(py, px).* = convertColor(T, rgb);
                    }
                }
            }
        }
        // TODO: Add support for 4:2:2 and other subsampling modes
    }
}

// Decode entire image
pub fn decodeImage(comptime T: type) !Image(T) {
    // This function would be called after parsing all markers
    // For now, return error to indicate incomplete implementation
    return error.NotImplemented;
}

// Load JPEG file following PNG pattern
pub fn loadJpeg(comptime T: type, allocator: Allocator, file_path: []const u8) !Image(T) {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    
    const file_size = try file.getEndPos();
    const data = try allocator.alloc(u8, file_size);
    defer allocator.free(data);
    
    _ = try file.read(data);
    
    var decoder = try decode(allocator, data);
    defer decoder.deinit();
    
    // Create output image
    var img = try Image(T).initAlloc(allocator, decoder.height, decoder.width);
    errdefer img.deinit(allocator);
    
    // Calculate max sampling factors
    var max_h: u4 = 0;
    var max_v: u4 = 0;
    for (0..decoder.num_components) |i| {
        max_h = @max(max_h, decoder.components[i].h_sampling);
        max_v = @max(max_v, decoder.components[i].v_sampling);
    }
    
    // Allocate MCU blocks organized by component
    var mcu_data: [4][][64]i32 = undefined; // Max 4 components
    defer {
        for (0..decoder.num_components) |i| {
            allocator.free(mcu_data[i]);
        }
    }
    
    for (0..decoder.num_components) |i| {
        const blocks_per_component = @as(usize, decoder.components[i].h_sampling) * decoder.components[i].v_sampling;
        mcu_data[i] = try allocator.alloc([64]i32, blocks_per_component);
    }
    
    // DC values for each component
    var dc_values = [_]i64{0} ** 4;
    
    // Decode MCUs
    var mcu_count: usize = 0;
    const total_mcus = decoder.mcu_width_in_blocks * decoder.mcu_height_in_blocks;
    
    for (0..decoder.mcu_height_in_blocks) |mcu_y| {
        for (0..decoder.mcu_width_in_blocks) |mcu_x| {
            try decoder.decodeMCU(&mcu_data, &dc_values);
            
            // Convert MCU to pixels using the new chroma subsampling-aware function
            convertMCUToPixels(T, &mcu_data, decoder.components[0..decoder.num_components], decoder.num_components, max_h, max_v, &img, mcu_x, mcu_y, decoder.width, decoder.height);
            
            mcu_count += 1;
            
            // Handle restart markers if needed
            if (decoder.restart_interval > 0 and mcu_count % decoder.restart_interval == 0 and mcu_count < total_mcus) {
                // Reset DC values
                dc_values = [_]i64{0} ** 4;
                // Reset bit buffer to handle restart marker
                decoder.bit_reader.resetBitBuffer();
            }
        }
    }
    
    return img;
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

test "YCbCr to RGB conversion" {
    const testing = std.testing;
    
    // Test grayscale (Y=128, Cb=128, Cr=128)
    const gray = ycbcrToRgb(128, 128, 128);
    try testing.expectEqual(@as(u8, 128), gray.r);
    try testing.expectEqual(@as(u8, 128), gray.g);
    try testing.expectEqual(@as(u8, 128), gray.b);
    
    // Test white (Y=255, Cb=128, Cr=128)
    const white = ycbcrToRgb(255, 128, 128);
    try testing.expectEqual(@as(u8, 255), white.r);
    try testing.expectEqual(@as(u8, 255), white.g);
    try testing.expectEqual(@as(u8, 255), white.b);
    
    // Test black (Y=0, Cb=128, Cr=128)
    const black = ycbcrToRgb(0, 128, 128);
    try testing.expectEqual(@as(u8, 0), black.r);
    try testing.expectEqual(@as(u8, 0), black.g);
    try testing.expectEqual(@as(u8, 0), black.b);
}