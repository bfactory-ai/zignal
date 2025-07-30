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

/// Options for loading BDF fonts
pub const BdfLoadOptions = struct {
    /// Load all characters in the font (default: false, loads only ASCII)
    load_all: bool = false,
    /// Specific Unicode ranges to load (null = use default behavior)
    ranges: ?[]const UnicodeRange = null,
    /// Maximum characters to load (0 = no limit)
    max_chars: usize = 0,
};

/// A Unicode character range
pub const UnicodeRange = struct {
    start: u21,
    end: u21,
};

/// Common Unicode ranges for convenience
pub const unicode_ranges = struct {
    /// Basic Latin (ASCII)
    pub const ascii = UnicodeRange{ .start = 0x0000, .end = 0x007F };

    /// Latin-1 Supplement
    pub const latin1_supplement = UnicodeRange{ .start = 0x0080, .end = 0x00FF };

    /// Full Latin-1 (ASCII + Latin-1 Supplement)
    pub const latin1 = UnicodeRange{ .start = 0x0000, .end = 0x00FF };

    /// Greek and Coptic
    pub const greek = UnicodeRange{ .start = 0x0370, .end = 0x03FF };

    /// Cyrillic
    pub const cyrillic = UnicodeRange{ .start = 0x0400, .end = 0x04FF };

    /// Arabic
    pub const arabic = UnicodeRange{ .start = 0x0600, .end = 0x06FF };

    /// Hebrew
    pub const hebrew = UnicodeRange{ .start = 0x0590, .end = 0x05FF };

    /// Hiragana
    pub const hiragana = UnicodeRange{ .start = 0x3040, .end = 0x309F };

    /// Katakana
    pub const katakana = UnicodeRange{ .start = 0x30A0, .end = 0x30FF };

    /// CJK Unified Ideographs (main block)
    pub const cjk_unified = UnicodeRange{ .start = 0x4E00, .end = 0x9FFF };

    /// Hangul Syllables (Korean)
    pub const hangul = UnicodeRange{ .start = 0xAC00, .end = 0xD7AF };

    /// Emoji & Pictographs
    pub const emoji = UnicodeRange{ .start = 0x1F300, .end = 0x1F9FF };

    /// Mathematical Operators
    pub const math = UnicodeRange{ .start = 0x2200, .end = 0x22FF };

    /// Box Drawing
    pub const box_drawing = UnicodeRange{ .start = 0x2500, .end = 0x257F };

    /// Block Elements
    pub const block_elements = UnicodeRange{ .start = 0x2580, .end = 0x259F };

    /// Common Western European languages (Latin-1 + Latin Extended-A)
    pub const western_european = [_]UnicodeRange{
        latin1,
        UnicodeRange{ .start = 0x0100, .end = 0x017F }, // Latin Extended-A
    };

    /// Common East Asian languages
    pub const east_asian = [_]UnicodeRange{
        hiragana,
        katakana,
        cjk_unified,
        hangul,
    };
};

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
    /// Font ascent (distance from baseline to top of font)
    font_ascent: i16,
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
                .font_ascent = 0,
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
                return BdfError.InvalidFormat;
            }
            if (font.glyph_count == 0) {
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
                    const key = line[0..space_idx];
                    const value = std.mem.trim(u8, line[space_idx + 1 ..], " \t\"");

                    // Handle specific properties we need
                    if (std.mem.eql(u8, key, "FONT_ASCENT")) {
                        font.font_ascent = try parseInt(value, i16);
                    }

                    // Store all properties in the map
                    const key_copy = try self.allocator.dupe(u8, key);
                    const value_copy = try self.allocator.dupe(u8, value);
                    try font.properties.?.put(key_copy, value_copy);
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
                if (std.mem.startsWith(u8, glyph_line, "ENCODING ")) {
                    const encoding_str = std.mem.trim(u8, glyph_line[9..], " \t");

                    // Check for negative encoding values (like ENCODING -1)
                    if (std.mem.startsWith(u8, encoding_str, "-")) {
                        // Skip glyphs with negative encodings
                        // These are typically .notdef or other special glyphs
                        // Clean up allocated name
                        self.allocator.free(glyph.name);
                        // Skip to ENDCHAR
                        while (try self.readLine()) |skip_line| {
                            if (std.mem.eql(u8, skip_line, "ENDCHAR")) {
                                break;
                            }
                        }
                        // Clear current line so next parseGlyph will read fresh
                        self.current_line = null;
                        // Try parsing the next glyph
                        return try self.parseGlyph();
                    }

                    glyph.encoding = try parseInt(encoding_str, u32);
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
                    return BdfError.InvalidBitmapData;
                }

                // Parse hexadecimal bitmap data - handle both upper and lower case
                const parsed_value = std.fmt.parseInt(u32, line, 16) catch {
                    return BdfError.InvalidBitmapData;
                };

                // BDF hex values need to be left-aligned to 32 bits
                // For example, "42" (8-bit) should become 0x42000000, not 0x00000042
                const hex_chars = line.len;
                // Ensure we don't underflow when hex_chars > 8
                const shift_amount = if (hex_chars < 8) (8 - hex_chars) * 4 else 0;
                glyph.bitmap[row] = parsed_value << @intCast(shift_amount);
            }
        }
    };
}

/// Load a BDF font from a reader with default options (ASCII only)
pub fn loadBdfFont(allocator: std.mem.Allocator, reader: anytype) !BdfLoadResult {
    return loadBdfFontWithOptions(allocator, reader, .{});
}

/// Load a BDF font from a reader with custom options
pub fn loadBdfFontWithOptions(allocator: std.mem.Allocator, reader: anytype, options: BdfLoadOptions) !BdfLoadResult {
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

    // Determine initial allocation size based on options
    const initial_size = if (options.load_all)
        font.glyph_count
    else if (options.max_chars > 0)
        @min(font.glyph_count, options.max_chars)
    else
        @min(font.glyph_count, 128); // Default to ASCII size

    var glyphs = std.ArrayList(BdfGlyph).init(allocator);
    defer glyphs.deinit();
    try glyphs.ensureTotalCapacity(initial_size);

    // Parse glyphs
    var parsed_count: usize = 0;
    parser.current_line = null; // Clear current line so parseGlyph reads next line
    while (try parser.parseGlyph()) |glyph| : (parsed_count += 1) {
        // if (parsed_count == 65413) {
        // }
        // if (parsed_count >= 65000) {
        // } else if (parsed_count % 1000 == 0) {
        // }
        // Determine if we should keep this glyph
        var should_keep = false;

        if (options.load_all) {
            should_keep = true;
        } else if (options.ranges) |ranges| {
            // Check if glyph is in any specified range
            for (ranges) |range| {
                if (glyph.encoding >= range.start and glyph.encoding <= range.end) {
                    should_keep = true;
                    break;
                }
            }
        } else {
            // Default: only keep ASCII
            should_keep = glyph.encoding <= 127;
        }

        if (should_keep) {
            try glyphs.append(glyph);

            // Check max_chars limit
            if (options.max_chars > 0 and glyphs.items.len >= options.max_chars) {
                break;
            }
        } else {
            // Free unused glyphs immediately
            var temp_glyph = glyph;
            temp_glyph.deinit();
        }

        // Bail out if we've parsed all characters
        if (parsed_count >= font.glyph_count) {
            break;
        }
    }

    return .{ .font = font, .glyphs = try glyphs.toOwnedSlice() };
}

/// Convert BDF glyphs to a BitmapFont compatible format
/// This creates a font that can be used with zignal's text rendering
pub fn convertToBitmapFont(allocator: std.mem.Allocator, bdf_font: BdfFont, glyphs: []const BdfGlyph) !BitmapFont {

    // Check if all glyphs fit in ASCII range
    var all_ascii = true;
    var min_ascii: u8 = 255;
    var max_ascii: u8 = 0;

    for (glyphs) |glyph| {
        if (glyph.encoding > 127) {
            all_ascii = false;
            break;
        }
        // Only update min/max if we know it's in ASCII range
        if (glyph.encoding <= 127) {
            min_ascii = @min(min_ascii, @as(u8, @intCast(glyph.encoding)));
            max_ascii = @max(max_ascii, @as(u8, @intCast(glyph.encoding)));
        }
    }

    if (glyphs.len == 0) {
        min_ascii = 32;
        max_ascii = 32;
    }

    const char_height = @as(usize, @intCast(@abs(bdf_font.bounding_box.height)));
    const char_width = @as(usize, @intCast(@abs(bdf_font.bounding_box.width)));
    const bytes_per_row = (char_width + 7) / 8; // Round up to nearest byte

    // Calculate bitmap data size and allocate
    var bitmap_data: []u8 = undefined;
    var glyph_map: ?std.AutoHashMap(u32, usize) = null;
    var glyph_data_array: ?[]GlyphData = null;

    if (all_ascii and glyphs.len > 0) {
        // For ASCII-only fonts, use contiguous array for efficiency
        const char_count = @as(usize, max_ascii - min_ascii + 1);
        const data_size = char_count * char_height * bytes_per_row;
        bitmap_data = try allocator.alloc(u8, data_size);
        errdefer allocator.free(bitmap_data);
        @memset(bitmap_data, 0);

        // Check if we need per-glyph data for variable widths
        var need_glyph_data = false;
        for (glyphs) |glyph| {
            if (glyph.bbx.width != char_width) {
                need_glyph_data = true;
                break;
            }
        }

        // If glyphs have variable widths, create glyph data
        if (need_glyph_data) {
            var glyph_data_list = std.ArrayList(GlyphData).init(allocator);
            defer glyph_data_list.deinit();

            var map = std.AutoHashMap(u32, usize).init(allocator);
            errdefer map.deinit();

            // Create glyph data for each character in the range
            // Only iterate if we have a valid range
            if (min_ascii <= max_ascii) {
                for (min_ascii..max_ascii + 1) |code| {
                    // Find the glyph for this code
                    var found_glyph: ?BdfGlyph = null;
                    for (glyphs) |glyph| {
                        if (glyph.encoding == code) {
                            found_glyph = glyph;
                            break;
                        }
                    }

                    if (found_glyph) |glyph| {
                        const idx = glyph_data_list.items.len;
                        try map.put(@intCast(code), idx);

                        // Convert BDF baseline-relative y_offset to top-relative
                        // BDF: y_offset is from baseline to bottom of glyph bbox
                        // We want: y_offset from top of line to top of glyph
                        const adjusted_y_offset = bdf_font.font_ascent - (glyph.bbx.y_offset + @as(i16, @intCast(glyph.bbx.height)));

                        try glyph_data_list.append(GlyphData{
                            .width = @intCast(glyph.bbx.width),
                            .height = @intCast(char_height), // Use font height, not glyph height
                            .x_offset = glyph.bbx.x_offset,
                            .y_offset = adjusted_y_offset,
                            .device_width = glyph.dwidth.x,
                            .bitmap_offset = 0, // Not used for ASCII fonts
                        });
                    }
                }
            }

            glyph_map = map;
            glyph_data_array = try glyph_data_list.toOwnedSlice();
        }

        // Convert each glyph
        for (glyphs) |glyph| {

            // Skip glyphs outside our ASCII range
            if (glyph.encoding < min_ascii or glyph.encoding > max_ascii) {
                continue;
            }

            // Additional safety check: ensure encoding fits in usize for ASCII fonts
            if (glyph.encoding > 255) {
                continue;
            }

            const char_idx = glyph.encoding - min_ascii;
            const data_offset = char_idx * char_height * bytes_per_row;

            // Convert BDF bitmap to our format
            for (0..@min(glyph.bbx.height, char_height)) |row| {
                if (row < glyph.bitmap.len) {
                    const bdf_row = glyph.bitmap[row];

                    for (0..bytes_per_row) |byte_idx| {
                        var our_byte: u8 = 0;
                        const start_bit = byte_idx * 8;
                        const end_bit = @min(start_bit + 8, glyph.bbx.width);

                        if (end_bit > start_bit) {
                            for (start_bit..end_bit) |bit| {
                                // BDF data is left-aligned to 32 bits, so we always shift from bit 31
                                // Ensure we don't exceed 31 bits
                                if (bit < 32) {
                                    if ((bdf_row >> @intCast(31 - bit)) & 1 != 0) {
                                        our_byte |= @as(u8, 1) << @intCast(bit - start_bit);
                                    }
                                }
                            }
                        }

                        bitmap_data[data_offset + row * bytes_per_row + byte_idx] = our_byte;
                    }
                }
            }
        }
    } else {
        // For Unicode fonts, use sparse storage with glyph map
        // Calculate total bitmap size needed
        var total_bitmap_size: usize = 0;
        for (glyphs) |glyph| {
            const glyph_size = @as(usize, @intCast(glyph.bbx.height)) * bytes_per_row;
            total_bitmap_size += glyph_size;
        }

        bitmap_data = try allocator.alloc(u8, total_bitmap_size);
        errdefer allocator.free(bitmap_data);

        var map = std.AutoHashMap(u32, usize).init(allocator);
        errdefer map.deinit();

        var glyph_data_list = std.ArrayList(GlyphData).init(allocator);
        defer glyph_data_list.deinit();

        var bitmap_offset: usize = 0;

        // Convert each glyph
        for (glyphs) |glyph| {
            // Store glyph location in map
            try map.put(glyph.encoding, glyph_data_list.items.len);

            // Convert BDF baseline-relative y_offset to top-relative
            // BDF: y_offset is from baseline to bottom of glyph bbox
            // We want: y_offset from top of line to top of glyph
            const adjusted_y_offset = bdf_font.font_ascent - (glyph.bbx.y_offset + @as(i16, @intCast(glyph.bbx.height)));

            // Store glyph metadata
            try glyph_data_list.append(GlyphData{
                .width = @intCast(glyph.bbx.width),
                .height = @intCast(glyph.bbx.height),
                .x_offset = glyph.bbx.x_offset,
                .y_offset = adjusted_y_offset,
                .device_width = glyph.dwidth.x,
                .bitmap_offset = bitmap_offset,
            });

            // Convert bitmap data
            const glyph_bytes_per_row = (glyph.bbx.width + 7) / 8;
            for (0..glyph.bbx.height) |row| {
                if (row < glyph.bitmap.len) {
                    const bdf_row = glyph.bitmap[row];

                    for (0..glyph_bytes_per_row) |byte_idx| {
                        var our_byte: u8 = 0;
                        const start_bit = byte_idx * 8;
                        const end_bit = @min(start_bit + 8, glyph.bbx.width);

                        if (end_bit > start_bit) {
                            for (start_bit..end_bit) |bit| {
                                // BDF data is left-aligned to 32 bits, so we always shift from bit 31
                                // Ensure we don't exceed 31 bits
                                if (bit < 32) {
                                    if ((bdf_row >> @intCast(31 - bit)) & 1 != 0) {
                                        our_byte |= @as(u8, 1) << @intCast(bit - start_bit);
                                    }
                                }
                            }
                        }

                        bitmap_data[bitmap_offset + row * glyph_bytes_per_row + byte_idx] = our_byte;
                    }
                }
            }

            bitmap_offset += @as(usize, @intCast(glyph.bbx.height)) * glyph_bytes_per_row;
        }

        glyph_map = map;
        glyph_data_array = try glyph_data_list.toOwnedSlice();

        // Set first/last char to 0 for Unicode fonts
        min_ascii = 0;
        max_ascii = 0;
    }

    return BitmapFont{
        .char_width = @intCast(char_width),
        .char_height = @intCast(char_height),
        .first_char = min_ascii,
        .last_char = max_ascii,
        .data = bitmap_data,
        .glyph_map = glyph_map,
        .glyph_data = glyph_data_array,
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

/// Load a BDF font from a file path with default options
pub fn loadBdfFontFromFile(allocator: std.mem.Allocator, path: []const u8) !BdfLoadResult {
    return loadBdfFontFromFileWithOptions(allocator, path, .{});
}

/// Load a BDF font from a file path with custom options
pub fn loadBdfFontFromFileWithOptions(allocator: std.mem.Allocator, path: []const u8, options: BdfLoadOptions) !BdfLoadResult {
    // Read entire file into memory
    const file_contents = try std.fs.cwd().readFileAlloc(allocator, path, 50 * 1024 * 1024); // 50MB max
    defer allocator.free(file_contents);

    // Create a fixed buffer stream from the contents
    var stream = std.io.fixedBufferStream(file_contents);
    return try loadBdfFontWithOptions(allocator, stream.reader(), options);
}

/// Load a BDF font and convert it to BitmapFont format with default options
pub fn loadBitmapFontFromBdfFile(allocator: std.mem.Allocator, path: []const u8) !BitmapFont {
    return loadBitmapFontFromBdfFileWithOptions(allocator, path, .{});
}

/// Load a BDF font and convert it to BitmapFont format with custom options
pub fn loadBitmapFontFromBdfFileWithOptions(allocator: std.mem.Allocator, path: []const u8, options: BdfLoadOptions) !BitmapFont {
    // Load BDF font
    const result = try loadBdfFontFromFileWithOptions(allocator, path, options);

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
