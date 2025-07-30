//! BDF (Bitmap Distribution Format) font parser for zignal
//!
//! This module provides zero-dependency parsing of BDF font files,
//! enabling support for Unicode bitmap fonts like GNU Unifont.

const std = @import("std");
const testing = std.testing;
const BitmapFont = @import("font.zig").BitmapFont;

/// Errors that can occur during BDF parsing
pub const BdfError = error{
    InvalidFormat,
    MissingStartFont,
    MissingEndFont,
    InvalidVersion,
    MissingSize,
    MissingFontBoundingBox,
    MissingChars,
    InvalidCharCount,
    MissingEncoding,
    InvalidEncoding,
    MissingBitmap,
    InvalidBitmapData,
    UnexpectedEndOfFile,
    LineTooLong,
    AllocationFailed,
};

/// Maximum line length in BDF files
const MAX_LINE_LENGTH = 1024;

/// BDF font information parsed from file
pub const BdfFont = struct {
    /// Font name
    name: []const u8,
    /// Point size
    point_size: u16,
    /// X resolution in DPI
    x_resolution: u16,
    /// Y resolution in DPI
    y_resolution: u16,
    /// Font bounding box
    bounding_box: struct {
        width: i16,
        height: i16,
        x_offset: i16,
        y_offset: i16,
    },
    /// Number of glyphs
    glyph_count: u32,
    /// Font properties (optional)
    properties: ?std.StringHashMap([]const u8),
    /// Allocator used for this font
    allocator: std.mem.Allocator,

    /// Free all allocated memory
    pub fn deinit(self: *BdfFont) void {
        self.allocator.free(self.name);
        if (self.properties) |*props| {
            var iter = props.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            props.deinit();
        }
    }
};

/// A single glyph from a BDF font
pub const BdfGlyph = struct {
    /// Character name
    name: []const u8,
    /// Encoding (Unicode code point)
    encoding: u32,
    /// Scalable width
    swidth: struct { x: i16, y: i16 },
    /// Device width
    dwidth: struct { x: i16, y: i16 },
    /// Glyph bounding box
    bbx: struct {
        width: u16,
        height: u16,
        x_offset: i16,
        y_offset: i16,
    },
    /// Bitmap data (each row is one element)
    bitmap: []u32,
    /// Allocator used for this glyph
    allocator: std.mem.Allocator,

    /// Free all allocated memory
    pub fn deinit(self: *BdfGlyph) void {
        self.allocator.free(self.name);
        if (self.bitmap.len > 0) {
            self.allocator.free(self.bitmap);
        }
    }
};

/// Parser state for BDF files
fn BdfParser(comptime ReaderType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        reader: ReaderType,
        line_buffer: [MAX_LINE_LENGTH]u8,
        current_line: ?[]const u8,
        line_number: u32,

        const Self = @This();

        /// Read next line from the BDF file
        fn readLine(self: *Self) !?[]const u8 {
            if (try self.reader.readUntilDelimiterOrEof(&self.line_buffer, '\n')) |line| {
                self.line_number += 1;
                // Trim carriage return if present
                const trimmed = if (line.len > 0 and line[line.len - 1] == '\r')
                    line[0 .. line.len - 1]
                else
                    line;
                self.current_line = trimmed;
                return trimmed;
            }
            self.current_line = null;
            return null;
        }

        /// Parse a line expecting a specific keyword
        fn expectKeyword(self: *Self, keyword: []const u8) ![]const u8 {
            const line = try self.readLine() orelse return BdfError.UnexpectedEndOfFile;

            if (!std.mem.startsWith(u8, line, keyword)) {
                std.debug.print("BDF parse error: Expected '{s}' but got '{s}' at line {}\n", .{ keyword, line, self.line_number });
                return BdfError.InvalidFormat;
            }

            // Skip keyword and any spaces
            var rest = line[keyword.len..];
            while (rest.len > 0 and rest[0] == ' ') {
                rest = rest[1..];
            }

            return rest;
        }

        /// Parse an integer from a string
        fn parseInt(str: []const u8, comptime T: type) !T {
            return std.fmt.parseInt(T, std.mem.trim(u8, str, " \t"), 10);
        }

        /// Parse the BDF header
        fn parseHeader(self: *Self) !BdfFont {
            // Parse STARTFONT
            const version_str = try self.expectKeyword("STARTFONT");
            if (!std.mem.eql(u8, version_str, "2.1") and !std.mem.eql(u8, version_str, "2.2")) {
                return BdfError.InvalidVersion;
            }

            var font = BdfFont{
                .name = &[_]u8{}, // Empty slice initially
                .point_size = 0,
                .x_resolution = 0,
                .y_resolution = 0,
                .bounding_box = .{ .width = 0, .height = 0, .x_offset = 0, .y_offset = 0 },
                .glyph_count = 0,
                .properties = null,
                .allocator = self.allocator,
            };

            // Parse header fields
            while (try self.readLine()) |line| {
                if (std.mem.startsWith(u8, line, "FONT ")) {
                    const name = std.mem.trim(u8, line[5..], " \t");
                    font.name = try self.allocator.dupe(u8, name);
                } else if (std.mem.startsWith(u8, line, "SIZE ")) {
                    var parts = std.mem.tokenizeAny(u8, line[5..], " \t");
                    font.point_size = try parseInt(parts.next() orelse return BdfError.MissingSize, u16);
                    font.x_resolution = try parseInt(parts.next() orelse return BdfError.MissingSize, u16);
                    font.y_resolution = try parseInt(parts.next() orelse return BdfError.MissingSize, u16);
                } else if (std.mem.startsWith(u8, line, "FONTBOUNDINGBOX ")) {
                    var parts = std.mem.tokenizeAny(u8, line[16..], " \t");
                    font.bounding_box.width = try parseInt(parts.next() orelse return BdfError.MissingFontBoundingBox, i16);
                    font.bounding_box.height = try parseInt(parts.next() orelse return BdfError.MissingFontBoundingBox, i16);
                    font.bounding_box.x_offset = try parseInt(parts.next() orelse return BdfError.MissingFontBoundingBox, i16);
                    font.bounding_box.y_offset = try parseInt(parts.next() orelse return BdfError.MissingFontBoundingBox, i16);
                } else if (std.mem.startsWith(u8, line, "STARTPROPERTIES ")) {
                    try self.parseProperties(&font);
                } else if (std.mem.startsWith(u8, line, "CHARS ")) {
                    font.glyph_count = try parseInt(line[6..], u32);
                    break;
                } else if (std.mem.startsWith(u8, line, "COMMENT") or
                    std.mem.startsWith(u8, line, "METRICSSET") or
                    std.mem.startsWith(u8, line, "SWIDTH") or
                    std.mem.startsWith(u8, line, "DWIDTH") or
                    std.mem.startsWith(u8, line, "VVECTOR"))
                {
                    // Skip these lines - they're not critical for basic rendering
                    continue;
                }
            }

            // Validate required fields
            if (font.name.len == 0) {
                std.debug.print("BDF parse error: Missing FONT name\n", .{});
                return BdfError.InvalidFormat;
            }
            if (font.glyph_count == 0) {
                std.debug.print("BDF parse error: Missing or zero CHARS count\n", .{});
                return BdfError.InvalidCharCount;
            }

            return font;
        }

        /// Parse font properties section
        fn parseProperties(self: *Self, font: *BdfFont) !void {
            font.properties = std.StringHashMap([]const u8).init(self.allocator);

            while (try self.readLine()) |line| {
                if (std.mem.eql(u8, line, "ENDPROPERTIES")) {
                    break;
                }

                // Parse property name and value
                if (std.mem.indexOf(u8, line, " ")) |space_idx| {
                    const key = try self.allocator.dupe(u8, line[0..space_idx]);
                    const value = try self.allocator.dupe(u8, std.mem.trim(u8, line[space_idx + 1 ..], " \t\""));
                    try font.properties.?.put(key, value);
                }
            }
        }

        /// Parse a single glyph
        fn parseGlyph(self: *Self) !?BdfGlyph {
            const line = self.current_line orelse try self.readLine() orelse return null;

            if (!std.mem.startsWith(u8, line, "STARTCHAR ")) {
                if (std.mem.eql(u8, line, "ENDFONT")) {
                    return null;
                }
                // std.debug.print("BDF glyph parse error: Expected 'STARTCHAR' but got '{s}' at line {}\n", .{ line, self.line_number });
                return BdfError.InvalidFormat;
            }

            var glyph = BdfGlyph{
                .name = undefined,
                .encoding = undefined,
                .swidth = .{ .x = 0, .y = 0 },
                .dwidth = .{ .x = 0, .y = 0 },
                .bbx = .{ .width = 0, .height = 0, .x_offset = 0, .y_offset = 0 },
                .bitmap = &[_]u32{}, // Empty slice initially
                .allocator = self.allocator,
            };

            // Parse character name
            glyph.name = try self.allocator.dupe(u8, std.mem.trim(u8, line[10..], " \t"));

            // Track whether bitmap was parsed
            var has_bitmap = false;

            // Parse glyph fields
            while (try self.readLine()) |glyph_line| {
                // std.debug.print("Glyph line: '{s}'\n", .{glyph_line});
                if (std.mem.startsWith(u8, glyph_line, "ENCODING ")) {
                    glyph.encoding = try parseInt(glyph_line[9..], u32);
                } else if (std.mem.startsWith(u8, glyph_line, "SWIDTH ")) {
                    var parts = std.mem.tokenizeAny(u8, glyph_line[7..], " \t");
                    glyph.swidth.x = try parseInt(parts.next() orelse "0", i16);
                    glyph.swidth.y = try parseInt(parts.next() orelse "0", i16);
                } else if (std.mem.startsWith(u8, glyph_line, "DWIDTH ")) {
                    var parts = std.mem.tokenizeAny(u8, glyph_line[7..], " \t");
                    glyph.dwidth.x = try parseInt(parts.next() orelse "0", i16);
                    glyph.dwidth.y = try parseInt(parts.next() orelse "0", i16);
                } else if (std.mem.startsWith(u8, glyph_line, "BBX ")) {
                    var parts = std.mem.tokenizeAny(u8, glyph_line[4..], " \t");
                    glyph.bbx.width = try parseInt(parts.next() orelse return BdfError.InvalidFormat, u16);
                    glyph.bbx.height = try parseInt(parts.next() orelse return BdfError.InvalidFormat, u16);
                    glyph.bbx.x_offset = try parseInt(parts.next() orelse return BdfError.InvalidFormat, i16);
                    glyph.bbx.y_offset = try parseInt(parts.next() orelse return BdfError.InvalidFormat, i16);

                    // If DWIDTH not specified, default to BBX width
                    if (glyph.dwidth.x == 0) {
                        glyph.dwidth.x = @intCast(glyph.bbx.width);
                    }

                    // Debug print
                    // std.debug.print("Glyph BBX: width={}, height={}\n", .{ glyph.bbx.width, glyph.bbx.height });
                } else if (std.mem.startsWith(u8, glyph_line, "BITMAP")) {
                    try self.parseBitmap(&glyph);
                    has_bitmap = true;
                    break;
                }
            }

            // Ensure bitmap was parsed
            if (!has_bitmap) {
                self.allocator.free(glyph.name);
                return BdfError.MissingBitmap;
            }

            // Skip to ENDCHAR
            while (try self.readLine()) |end_line| {
                if (std.mem.eql(u8, end_line, "ENDCHAR")) {
                    // Clear current_line so next parseGlyph starts fresh
                    self.current_line = null;
                    break;
                }
            }

            return glyph;
        }

        /// Parse bitmap data for a glyph
        fn parseBitmap(self: *Self, glyph: *BdfGlyph) !void {
            glyph.bitmap = try self.allocator.alloc(u32, glyph.bbx.height);

            for (0..glyph.bbx.height) |row| {
                const line = try self.readLine() orelse return BdfError.UnexpectedEndOfFile;
                if (std.mem.eql(u8, line, "ENDCHAR")) {
                    std.debug.print("BDF bitmap parse error: Found ENDCHAR after {} rows, expected {} rows\n", .{ row, glyph.bbx.height });
                    return BdfError.InvalidBitmapData;
                }

                // Parse hexadecimal bitmap data - handle both upper and lower case
                const parsed_value = std.fmt.parseInt(u32, line, 16) catch {
                    std.debug.print("BDF bitmap parse error: Invalid hex data '{s}' at row {}\n", .{ line, row });
                    return BdfError.InvalidBitmapData;
                };

                // BDF hex values need to be left-aligned to 32 bits
                // For example, "42" (8-bit) should become 0x42000000, not 0x00000042
                const hex_chars = line.len;
                const shift_amount = (8 - hex_chars) * 4; // 4 bits per hex char
                glyph.bitmap[row] = parsed_value << @intCast(shift_amount);
            }
        }
    };
}

/// Load a BDF font from a reader
pub fn loadBdfFont(allocator: std.mem.Allocator, reader: anytype) !BdfLoadResult {
    const Parser = BdfParser(@TypeOf(reader));
    var parser = Parser{
        .allocator = allocator,
        .reader = reader,
        .line_buffer = undefined,
        .current_line = null,
        .line_number = 0,
    };

    // Parse header
    var font = try parser.parseHeader();
    errdefer font.deinit();

    // Allocate glyph array - for ASCII only, allocate smaller array
    const initial_size = @min(font.glyph_count, 128);
    var glyphs = try allocator.alloc(BdfGlyph, initial_size);
    errdefer allocator.free(glyphs);

    // Parse glyphs
    var glyph_idx: usize = 0;
    var parsed_count: usize = 0;
    parser.current_line = null; // Clear current line so parseGlyph reads next line
    while (try parser.parseGlyph()) |glyph| : (parsed_count += 1) {
        // For now, only store ASCII glyphs to keep memory reasonable
        if (glyph.encoding <= 127) {
            if (glyph_idx >= glyphs.len) {
                return BdfError.InvalidCharCount;
            }
            glyphs[glyph_idx] = glyph;
            glyph_idx += 1;
        } else {
            // Free non-ASCII glyphs immediately
            var temp_glyph = glyph;
            temp_glyph.deinit();
        }

        // Bail out if we've parsed enough characters
        if (parsed_count >= font.glyph_count) {
            break;
        }
    }

    // Resize glyphs array to actual count
    if (glyph_idx < glyphs.len) {
        glyphs = allocator.realloc(glyphs, glyph_idx) catch glyphs[0..glyph_idx];
    }

    return .{ .font = font, .glyphs = glyphs };
}

/// Convert BDF glyphs to a BitmapFont compatible format
/// This creates a font that can be used with zignal's text rendering
pub fn convertToBitmapFont(allocator: std.mem.Allocator, bdf_font: BdfFont, glyphs: []const BdfGlyph) !BitmapFont {
    // For now, we'll create a simple ASCII subset font
    // TODO: Extend to support full Unicode range

    // Find the range of ASCII characters we have
    var min_ascii: u8 = 255;
    var max_ascii: u8 = 0;
    var ascii_count: usize = 0;

    for (glyphs) |glyph| {
        if (glyph.encoding <= 127) {
            min_ascii = @min(min_ascii, @as(u8, @intCast(glyph.encoding)));
            max_ascii = @max(max_ascii, @as(u8, @intCast(glyph.encoding)));
            ascii_count += 1;
        }
    }

    if (ascii_count == 0) {
        min_ascii = 32;
        max_ascii = 32;
    }

    // Calculate bitmap data size
    const char_count = @as(usize, max_ascii - min_ascii + 1);
    const char_height = @as(usize, @intCast(@abs(bdf_font.bounding_box.height)));
    const char_width = @as(usize, @intCast(@abs(bdf_font.bounding_box.width)));
    // For fonts wider than 8 pixels, we need multiple bytes per row
    const bytes_per_row = (char_width + 7) / 8; // Round up to nearest byte
    const data_size = char_count * char_height * bytes_per_row;

    // Allocate bitmap data
    var bitmap_data = try allocator.alloc(u8, data_size);
    errdefer allocator.free(bitmap_data);

    // Clear bitmap data
    @memset(bitmap_data, 0);

    // Create glyph map and data arrays
    var glyph_map = std.AutoHashMap(u32, usize).init(allocator);
    errdefer glyph_map.deinit();

    var glyph_data_list = std.ArrayList(GlyphData).init(allocator);
    defer glyph_data_list.deinit();

    // Convert each glyph
    for (glyphs) |glyph| {
        if (glyph.encoding >= min_ascii and glyph.encoding <= max_ascii) {
            const char_idx = glyph.encoding - min_ascii;
            const data_offset = char_idx * char_height * bytes_per_row;

            // Convert BDF bitmap to our format
            for (0..@min(glyph.bbx.height, char_height)) |row| {
                if (row < glyph.bitmap.len) {
                    // BDF uses MSB first, we use LSB first
                    const bdf_row = glyph.bitmap[row];

                    // Convert all bits, possibly spanning multiple bytes
                    for (0..bytes_per_row) |byte_idx| {
                        var our_byte: u8 = 0;
                        const start_bit = byte_idx * 8;
                        const end_bit = @min(start_bit + 8, glyph.bbx.width);

                        for (start_bit..end_bit) |bit| {
                            if ((bdf_row >> @intCast(31 - bit)) & 1 != 0) {
                                our_byte |= @as(u8, 1) << @intCast(bit - start_bit);
                            }
                        }

                        bitmap_data[data_offset + row * bytes_per_row + byte_idx] = our_byte;
                    }
                }
            }

            // Store glyph data
            try glyph_map.put(glyph.encoding, glyph_data_list.items.len);
            try glyph_data_list.append(GlyphData{
                .width = @intCast(glyph.bbx.width),
                .height = @intCast(glyph.bbx.height),
                .x_offset = glyph.bbx.x_offset,
                .y_offset = glyph.bbx.y_offset,
                .device_width = glyph.dwidth.x,
            });
        }
    }

    return BitmapFont{
        .char_width = @intCast(char_width),
        .char_height = @intCast(char_height),
        .first_char = min_ascii,
        .last_char = max_ascii,
        .data = bitmap_data,
        .glyph_map = glyph_map,
        .glyph_data = try glyph_data_list.toOwnedSlice(),
    };
}

/// Import GlyphData from font module for consistency
pub const GlyphData = @import("font.zig").GlyphData;

/// Result of loading a BDF font
pub const BdfLoadResult = struct {
    font: BdfFont,
    glyphs: []BdfGlyph,
};

test "BDF parser - parse simple font header" {
    const test_bdf =
        \\STARTFONT 2.1
        \\FONT -Misc-Fixed-Medium-R-Normal--8-80-75-75-C-50-ISO10646-1
        \\SIZE 8 75 75
        \\FONTBOUNDINGBOX 5 8 0 -1
        \\STARTPROPERTIES 3
        \\FONT_ASCENT 7
        \\FONT_DESCENT 1
        \\DEFAULT_CHAR 0
        \\ENDPROPERTIES
        \\CHARS 2
        \\STARTCHAR space
        \\ENCODING 32
        \\SWIDTH 500 0
        \\DWIDTH 5 0
        \\BBX 5 8 0 -1
        \\BITMAP
        \\00
        \\00
        \\00
        \\00
        \\00
        \\00
        \\00
        \\00
        \\ENDCHAR
        \\STARTCHAR A
        \\ENCODING 65
        \\SWIDTH 500 0
        \\DWIDTH 5 0
        \\BBX 5 8 0 -1
        \\BITMAP
        \\20
        \\50
        \\50
        \\88
        \\F8
        \\88
        \\88
        \\00
        \\ENDCHAR
        \\ENDFONT
    ;

    var stream = std.io.fixedBufferStream(test_bdf);
    const result = try loadBdfFont(testing.allocator, stream.reader());
    defer {
        var font = result.font;
        font.deinit();
        for (result.glyphs) |*glyph| {
            glyph.deinit();
        }
        testing.allocator.free(result.glyphs);
    }

    // Test font header
    try testing.expectEqualStrings("-Misc-Fixed-Medium-R-Normal--8-80-75-75-C-50-ISO10646-1", result.font.name);
    try testing.expectEqual(@as(u16, 8), result.font.point_size);
    try testing.expectEqual(@as(u16, 75), result.font.x_resolution);
    try testing.expectEqual(@as(u16, 75), result.font.y_resolution);
    try testing.expectEqual(@as(i16, 5), result.font.bounding_box.width);
    try testing.expectEqual(@as(i16, 8), result.font.bounding_box.height);
    try testing.expectEqual(@as(u32, 2), result.font.glyph_count);

    // Test properties
    try testing.expect(result.font.properties != null);
    try testing.expectEqualStrings("7", result.font.properties.?.get("FONT_ASCENT").?);
    try testing.expectEqualStrings("1", result.font.properties.?.get("FONT_DESCENT").?);

    // Test glyphs
    try testing.expectEqual(@as(usize, 2), result.glyphs.len);

    // Test space character
    try testing.expectEqualStrings("space", result.glyphs[0].name);
    try testing.expectEqual(@as(u32, 32), result.glyphs[0].encoding);
    try testing.expectEqual(@as(u16, 5), result.glyphs[0].bbx.width);
    try testing.expectEqual(@as(u16, 8), result.glyphs[0].bbx.height);

    // Test 'A' character
    try testing.expectEqualStrings("A", result.glyphs[1].name);
    try testing.expectEqual(@as(u32, 65), result.glyphs[1].encoding);
    try testing.expectEqual(@as(u32, 0x20), result.glyphs[1].bitmap[0]);
    try testing.expectEqual(@as(u32, 0x50), result.glyphs[1].bitmap[1]);
    try testing.expectEqual(@as(u32, 0xF8), result.glyphs[1].bitmap[4]);
}

/// Load a BDF font from a file path
pub fn loadBdfFontFromFile(allocator: std.mem.Allocator, path: []const u8) !BdfLoadResult {
    // Read entire file into memory
    const file_contents = try std.fs.cwd().readFileAlloc(allocator, path, 50 * 1024 * 1024); // 50MB max
    defer allocator.free(file_contents);

    // Create a fixed buffer stream from the contents
    var stream = std.io.fixedBufferStream(file_contents);
    return try loadBdfFont(allocator, stream.reader());
}

/// Load a BDF font and convert it to BitmapFont format
pub fn loadBitmapFontFromBdfFile(allocator: std.mem.Allocator, path: []const u8) !BitmapFont {
    // Load BDF font
    const result = try loadBdfFontFromFile(allocator, path);

    // Save what we need before cleanup
    const converted = try convertToBitmapFont(allocator, result.font, result.glyphs);

    // Clean up BDF data
    var font = result.font;
    font.deinit();
    for (result.glyphs) |*glyph| {
        glyph.deinit();
    }
    allocator.free(result.glyphs);

    return converted;
}

test "BDF to BitmapFont conversion" {
    const test_bdf =
        \\STARTFONT 2.1
        \\FONT test
        \\SIZE 8 75 75
        \\FONTBOUNDINGBOX 8 8 0 0
        \\CHARS 1
        \\STARTCHAR A
        \\ENCODING 65
        \\SWIDTH 500 0
        \\DWIDTH 8 0
        \\BBX 8 8 0 0
        \\BITMAP
        \\18
        \\24
        \\42
        \\42
        \\7E
        \\42
        \\42
        \\00
        \\ENDCHAR
        \\ENDFONT
    ;

    var stream = std.io.fixedBufferStream(test_bdf);
    const result = try loadBdfFont(testing.allocator, stream.reader());
    defer {
        var font = result.font;
        font.deinit();
        for (result.glyphs) |*glyph| {
            glyph.deinit();
        }
        testing.allocator.free(result.glyphs);
    }

    // Convert to BitmapFont
    var converted = try convertToBitmapFont(testing.allocator, result.font, result.glyphs);
    defer converted.deinit(testing.allocator);

    // Test converted font
    try testing.expectEqual(@as(u8, 8), converted.char_height);
    try testing.expectEqual(@as(u8, 65), converted.first_char);
    try testing.expectEqual(@as(u8, 65), converted.last_char);

    // Test that 'A' was converted correctly
    const char_data = converted.getCharData('A');
    try testing.expect(char_data != null);
    try testing.expectEqual(@as(usize, 8), char_data.?.len);

    // Check bitmap conversion (BDF MSB to our LSB format)
    // BDF: 0x18 = 00011000 -> LSB: 00011000 = 0x18
    try testing.expectEqual(@as(u8, 0x18), char_data.?[0]);
    // BDF: 0x24 = 00100100 -> LSB: 00100100 = 0x24
    try testing.expectEqual(@as(u8, 0x24), char_data.?[1]);
}
