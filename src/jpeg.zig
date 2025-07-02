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
                // 8-bit values
                for (0..64) |i| {
                    table[i] = data[pos + i];
                }
                pos += 64;
            } else {
                // 16-bit values
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

    // Decode DC coefficient
    pub fn decodeDC(self: *JpegDecoder, table: *const HuffmanTable, prev_dc: i32) !i32 {
        const symbol = try self.decodeHuffmanSymbol(table);
        
        if (symbol == 0) {
            return prev_dc;
        }
        
        if (symbol > 11) return error.InvalidDCValue;
        
        const bits = try self.bit_reader.getBits(@intCast(symbol));
        var diff: i32 = @intCast(bits);
        
        // Convert from unsigned to signed
        if (bits < (@as(u32, 1) << @intCast(symbol - 1))) {
            diff = @as(i32, @intCast(bits)) - @as(i32, @intCast((@as(u32, 1) << @intCast(symbol)) - 1));
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
                    block[k] = 0;
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
                    block[k] = 0;
                    k += 1;
                }
            } else {
                // Skip 'run' zeros
                for (0..run) |_| {
                    if (k >= 64) return error.InvalidACValue;
                    block[k] = 0;
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
                
                block[k] = value;
                k += 1;
            }
        }
    }

    // Decode a single 8x8 block
    pub fn decodeBlock(self: *JpegDecoder, component_idx: usize, prev_dc: i32) ![64]i32 {
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
        block[0] = try self.decodeDC(&dc_table, prev_dc);
        
        // Decode AC coefficients
        const ac_table = self.ac_tables[scan_comp.ac_table_id] orelse return error.MissingHuffmanTable;
        try self.decodeAC(&ac_table, &block);
        
        // Dequantize
        const quant_table = self.quant_tables[self.components[component_idx].quant_table_id] orelse return error.MissingQuantTable;
        for (0..64) |i| {
            block[ZIGZAG_ORDER[i]] *= @intCast(quant_table[i]);
        }
        
        return block;
    }

    // Decode MCU (Minimum Coded Unit)
    pub fn decodeMCU(self: *JpegDecoder, mcu_blocks: [][]i32, dc_values: []i32) !void {
        var block_idx: usize = 0;
        
        for (0..self.num_components) |comp_idx| {
            const comp = self.components[comp_idx];
            const blocks_h = comp.h_sampling;
            const blocks_v = comp.v_sampling;
            
            for (0..blocks_v) |_| {
                for (0..blocks_h) |_| {
                    var block = try self.decodeBlock(comp_idx, dc_values[comp_idx]);
                    dc_values[comp_idx] = block[0]; // Update DC value
                    
                    // Apply IDCT
                    idct8x8(&block);
                    
                    // Level shift for color components
                    if (self.num_components == 3) {
                        for (&block) |*val| {
                            val.* += 128;
                        }
                    }
                    
                    @memcpy(mcu_blocks[block_idx], &block);
                    block_idx += 1;
                }
            }
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
    
    pub fn getBits(self: *BitReader, n: u5) !u16 {
        while (self.bits_left < n) {
            if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
            
            const byte = self.data[self.byte_pos];
            self.byte_pos += 1;
            
            // Handle 0xFF byte stuffing
            if (byte == 0xFF) {
                if (self.byte_pos >= self.data.len) return error.UnexpectedEndOfData;
                const next = self.data[self.byte_pos];
                if (next != 0x00) return error.InvalidByteStuffing;
                self.byte_pos += 1;
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
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
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
                    if (data[scan_end] == 0xFF and data[scan_end + 1] != 0x00) {
                        break;
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

// IDCT constants
const IDCT_SCALE = 2048; // 2^11 for fixed-point arithmetic
const IDCT_HALF = 1024;  // IDCT_SCALE / 2

// Fast integer IDCT based on Loeffler algorithm
fn idct8x8(block: *[64]i32) void {
    var temp: [64]i32 = undefined;
    
    // Process rows
    for (0..8) |i| {
        const row_offset = i * 8;
        
        // Even part
        const tmp0 = block[row_offset + 0];
        const tmp1 = block[row_offset + 2];
        const tmp2 = block[row_offset + 4];
        const tmp3 = block[row_offset + 6];
        
        const tmp10 = tmp0 + tmp2;
        const tmp11 = tmp0 - tmp2;
        const tmp13 = tmp1 + tmp3;
        const tmp12 = mulFixedPoint(tmp1 - tmp3, 1414) - tmp13; // 1.414213562
        
        const e0 = tmp10 + tmp13;
        const e3 = tmp10 - tmp13;
        const e1 = tmp11 + tmp12;
        const e2 = tmp11 - tmp12;
        
        // Odd part
        const tmp4 = block[row_offset + 1];
        const tmp5 = block[row_offset + 3];
        const tmp6 = block[row_offset + 5];
        const tmp7 = block[row_offset + 7];
        
        const z13 = tmp6 + tmp5;
        const z10 = tmp6 - tmp5;
        const z11 = tmp4 + tmp7;
        const z12 = tmp4 - tmp7;
        
        const tmp7_new = z11 + z13;
        const tmp11_new = mulFixedPoint(z11 - z13, 1414); // 1.414213562
        
        const z5 = mulFixedPoint(z10 + z12, 1847); // 1.847759065
        const tmp10_new = mulFixedPoint(z12, 1082) - z5; // 1.082392200
        const tmp12_new = mulFixedPoint(z10, -2613) + z5; // -2.613125930
        
        const tmp6_new = tmp12_new - tmp7_new;
        const tmp5_new = tmp11_new - tmp6_new;
        const tmp4_new = tmp10_new + tmp5_new;
        
        // Final output
        temp[row_offset + 0] = e0 + tmp7_new;
        temp[row_offset + 7] = e0 - tmp7_new;
        temp[row_offset + 1] = e1 + tmp6_new;
        temp[row_offset + 6] = e1 - tmp6_new;
        temp[row_offset + 2] = e2 + tmp5_new;
        temp[row_offset + 5] = e2 - tmp5_new;
        temp[row_offset + 4] = e3 + tmp4_new;
        temp[row_offset + 3] = e3 - tmp4_new;
    }
    
    // Process columns
    for (0..8) |i| {
        // Even part
        const tmp0 = temp[i + 0 * 8];
        const tmp1 = temp[i + 2 * 8];
        const tmp2 = temp[i + 4 * 8];
        const tmp3 = temp[i + 6 * 8];
        
        const tmp10 = tmp0 + tmp2;
        const tmp11 = tmp0 - tmp2;
        const tmp13 = tmp1 + tmp3;
        const tmp12 = mulFixedPoint(tmp1 - tmp3, 1414) - tmp13;
        
        const e0 = tmp10 + tmp13;
        const e3 = tmp10 - tmp13;
        const e1 = tmp11 + tmp12;
        const e2 = tmp11 - tmp12;
        
        // Odd part
        const tmp4 = temp[i + 1 * 8];
        const tmp5 = temp[i + 3 * 8];
        const tmp6 = temp[i + 5 * 8];
        const tmp7 = temp[i + 7 * 8];
        
        const z13 = tmp6 + tmp5;
        const z10 = tmp6 - tmp5;
        const z11 = tmp4 + tmp7;
        const z12 = tmp4 - tmp7;
        
        const tmp7_new = z11 + z13;
        const tmp11_new = mulFixedPoint(z11 - z13, 1414);
        
        const z5 = mulFixedPoint(z10 + z12, 1847);
        const tmp10_new = mulFixedPoint(z12, 1082) - z5;
        const tmp12_new = mulFixedPoint(z10, -2613) + z5;
        
        const tmp6_new = tmp12_new - tmp7_new;
        const tmp5_new = tmp11_new - tmp6_new;
        const tmp4_new = tmp10_new + tmp5_new;
        
        // Final output with rounding and shifting
        block[i + 0 * 8] = descale(e0 + tmp7_new);
        block[i + 7 * 8] = descale(e0 - tmp7_new);
        block[i + 1 * 8] = descale(e1 + tmp6_new);
        block[i + 6 * 8] = descale(e1 - tmp6_new);
        block[i + 2 * 8] = descale(e2 + tmp5_new);
        block[i + 5 * 8] = descale(e2 - tmp5_new);
        block[i + 4 * 8] = descale(e3 + tmp4_new);
        block[i + 3 * 8] = descale(e3 - tmp4_new);
    }
}

// Fixed-point multiplication with rounding
fn mulFixedPoint(a: i32, b: i32) i32 {
    return @divTrunc(a * b + IDCT_HALF, IDCT_SCALE);
}

// Descale and clamp to valid range
fn descale(x: i32) i32 {
    const shifted = @divTrunc(x + 8, 16); // Round and shift by 4 bits
    return shifted;
}

// YCbCr to RGB conversion
fn ycbcrToRgb(y: i32, cb: i32, cr: i32) Rgb {
    // Convert from JPEG YCbCr to RGB
    // Y is in range [0, 255]
    // Cb and Cr are in range [-128, 127] after level shift
    const y_shifted = y;
    const cb_shifted = cb - 128;
    const cr_shifted = cr - 128;
    
    // Fixed-point conversion (scaled by 256)
    const r = y_shifted + ((cr_shifted * 359) >> 8); // 1.402 * 256 = 359
    const g = y_shifted - ((cb_shifted * 88) >> 8) - ((cr_shifted * 183) >> 8); // 0.344 * 256 = 88, 0.714 * 256 = 183
    const b = y_shifted + ((cb_shifted * 454) >> 8); // 1.772 * 256 = 454
    
    return Rgb{
        .r = clampU8(r),
        .g = clampU8(g),
        .b = clampU8(b),
    };
}

// Clamp value to u8 range
fn clampU8(value: i32) u8 {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return @intCast(value);
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
    
    // Decode image data
    const max_h = decoder.components[0].h_sampling;
    const max_v = decoder.components[0].v_sampling;
    
    // Allocate MCU blocks
    var total_blocks: usize = 0;
    for (0..decoder.num_components) |i| {
        total_blocks += decoder.components[i].h_sampling * decoder.components[i].v_sampling;
    }
    
    var mcu_blocks = try allocator.alloc([]i32, total_blocks);
    defer allocator.free(mcu_blocks);
    
    for (0..total_blocks) |i| {
        mcu_blocks[i] = try allocator.alloc(i32, 64);
    }
    defer for (mcu_blocks) |block| {
        allocator.free(block);
    };
    
    // DC values for each component
    var dc_values = [_]i32{0} ** 4;
    
    // Decode MCUs
    var mcu_count: usize = 0;
    const total_mcus = decoder.mcu_width_in_blocks * decoder.mcu_height_in_blocks;
    
    for (0..decoder.mcu_height_in_blocks) |mcu_y| {
        for (0..decoder.mcu_width_in_blocks) |mcu_x| {
            try decoder.decodeMCU(mcu_blocks, &dc_values);
            
            // Convert MCU blocks to pixels
            if (decoder.num_components == 1) {
                // Grayscale
                const block = mcu_blocks[0];
                for (0..8) |y| {
                    for (0..8) |x| {
                        const px = mcu_x * 8 + x;
                        const py = mcu_y * 8 + y;
                        if (px < decoder.width and py < decoder.height) {
                            const gray = clampU8(block[y * 8 + x]);
                            const rgb = Rgb{ .r = gray, .g = gray, .b = gray };
                            img.at(py, px).* = convertColor(T, rgb);
                        }
                    }
                }
            } else if (decoder.num_components == 3) {
                // YCbCr color image
                // For now, handle only 4:4:4 (no subsampling)
                if (max_h == 1 and max_v == 1) {
                    const y_block = mcu_blocks[0];
                    const cb_block = mcu_blocks[1];
                    const cr_block = mcu_blocks[2];
                    
                    for (0..8) |y| {
                        for (0..8) |x| {
                            const px = mcu_x * 8 + x;
                            const py = mcu_y * 8 + y;
                            if (px < decoder.width and py < decoder.height) {
                                const rgb = ycbcrToRgb(
                                    y_block[y * 8 + x],
                                    cb_block[y * 8 + x],
                                    cr_block[y * 8 + x]
                                );
                                img.at(py, px).* = convertColor(T, rgb);
                            }
                        }
                    }
                } else {
                    // TODO: Handle chroma subsampling
                    return error.UnsupportedJpegFormat;
                }
            }
            
            mcu_count += 1;
            
            // Handle restart markers if needed
            if (decoder.restart_interval > 0 and mcu_count % decoder.restart_interval == 0 and mcu_count < total_mcus) {
                // Reset DC values
                dc_values = [_]i32{0} ** 4;
                // TODO: Find and skip restart marker in bit stream
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