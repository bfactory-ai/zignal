//! Line layout shared by every text drawing entry point: paragraphs, word wrap,
//! alignment and spacing, for bitmap and vector fonts alike. Widths come from
//! `Font.getTextBounds`, so vector kerning is honored when deciding breaks.

const std = @import("std");

const Font = @import("../font.zig").Font;
const Rectangle = @import("../geometry.zig").Rectangle;

pub const TextAlign = enum { left, center, right };
pub const VerticalAlign = enum { top, middle, bottom };

/// How `Canvas.drawTextBox` arranges lines inside its box.
pub const TextLayout = struct {
    halign: TextAlign = .left,
    valign: VerticalAlign = .top,
    /// Break lines at spaces so none exceeds the available width; a word wider than the
    /// width breaks between codepoints. `\n` always breaks.
    wrap: bool = false,
    /// Multiplier on the font's line height; values at or below zero stack the lines.
    line_spacing: f32 = 1,
    /// Extra pixels between consecutive glyphs; negative tightens.
    letter_spacing: f32 = 0,

    pub const default: TextLayout = .{};
};

/// Width of `slice` drawn on one line: its advance plus `letter_spacing` between its
/// codepoints, never negative.
pub fn lineWidth(font: Font, slice: []const u8, size: f32, letter_spacing: f32) f32 {
    if (slice.len == 0) return 0;
    const glyphs = std.unicode.utf8CountCodepoints(slice) catch slice.len;
    const gaps: f32 = @floatFromInt(glyphs -| 1);
    return @max(0, font.getTextBounds(slice, size).r + letter_spacing * gaps);
}

/// Baseline-to-baseline distance under `layout`.
pub fn lineAdvance(font: Font, size: f32, layout: TextLayout) f32 {
    return font.lineHeight(size) * layout.line_spacing;
}

/// The lines of `text` at `size`: paragraphs split on `\n`, then wrapped greedily to
/// `max_width` when `layout.wrap` is set. A wrapped line ends at a word boundary, the
/// spaces at the break are consumed, and a word wider than `max_width` breaks between
/// codepoints. Empty text yields one empty line, like `getTextBounds`.
pub const Lines = struct {
    pub const Line = struct {
        text: []const u8,
        /// `lineWidth` of the text without trailing spaces.
        width: f32,
    };

    font: Font,
    text: []const u8,
    size: f32,
    max_width: ?f32,
    layout: TextLayout,
    pos: usize = 0,
    finished: bool = false,

    pub fn init(font: Font, text: []const u8, size: f32, max_width: ?f32, layout: TextLayout) Lines {
        return .{ .font = font, .text = text, .size = size, .max_width = max_width, .layout = layout };
    }

    pub fn next(self: *Lines) ?Line {
        if (self.finished) return null;
        const rest = self.text[self.pos..];
        const newline = std.mem.indexOfScalar(u8, rest, '\n');
        const paragraph = if (newline) |n| rest[0..n] else rest;

        var line = paragraph;
        var consumed = paragraph.len + @intFromBool(newline != null);
        if (self.max_width) |max_width| if (self.layout.wrap) {
            const fitted = self.fit(paragraph, max_width);
            if (fitted.len < paragraph.len) {
                line = paragraph[0..fitted.len];
                consumed = fitted.consumed;
            }
        };
        self.pos += consumed;
        if (newline == null and consumed == paragraph.len) self.finished = true;
        return .{
            .text = line,
            .width = self.width(std.mem.trimEnd(u8, line, " ")),
        };
    }

    fn width(self: Lines, slice: []const u8) f32 {
        return lineWidth(self.font, slice, self.size, self.layout.letter_spacing);
    }

    const Fit = struct { len: usize, consumed: usize };

    /// The longest prefix of `paragraph` ending at a word boundary that fits `max_width`,
    /// plus the spaces after it; falling back to the longest fitting prefix of the first
    /// word, never shorter than one codepoint.
    fn fit(self: Lines, paragraph: []const u8, max_width: f32) Fit {
        var accepted: usize = 0;
        var i: usize = 0;
        while (true) {
            var word_start = i;
            while (word_start < paragraph.len and paragraph[word_start] == ' ') word_start += 1;
            if (word_start == paragraph.len) return .{ .len = paragraph.len, .consumed = paragraph.len };
            const word_end = std.mem.indexOfScalarPos(u8, paragraph, word_start, ' ') orelse paragraph.len;
            if (self.width(paragraph[0..word_end]) > max_width) break;
            accepted = word_end;
            i = word_end;
        }
        if (accepted == 0) {
            const word_end = std.mem.indexOfScalar(u8, paragraph, ' ') orelse paragraph.len;
            var view: std.unicode.Utf8View = .initUnchecked(paragraph[0..word_end]);
            var iter = view.iterator();
            _ = iter.nextCodepointSlice() orelse return .{ .len = paragraph.len, .consumed = paragraph.len };
            accepted = iter.i;
            while (iter.nextCodepointSlice()) |_| {
                if (self.width(paragraph[0..iter.i]) > max_width) break;
                accepted = iter.i;
            }
        }
        var consumed = accepted;
        while (consumed < paragraph.len and paragraph[consumed] == ' ') consumed += 1;
        return .{ .len = accepted, .consumed = consumed };
    }
};

/// Box the lines of `text` occupy from their top-left corner: the widest line by the
/// number of lines times the line advance.
pub fn measure(font: Font, text: []const u8, size: f32, max_width: ?f32, layout: TextLayout) Rectangle(f32) {
    var lines: Lines = .init(font, text, size, max_width, layout);
    var width: f32 = 0;
    var count: usize = 0;
    while (lines.next()) |line| {
        width = @max(width, line.width);
        count += 1;
    }
    return .{ .l = 0, .t = 0, .r = width, .b = @as(f32, @floatFromInt(count)) * lineAdvance(font, size, layout) };
}

const testing = std.testing;
const font8x8 = @import("font8x8.zig");

fn expectLines(font: Font, text: []const u8, max_width: ?f32, layout: TextLayout, expected: []const []const u8) !void {
    var lines: Lines = .init(font, text, 8, max_width, layout);
    for (expected) |want| {
        const line = lines.next() orelse return error.TestExpectedMoreLines;
        try testing.expectEqualStrings(want, line.text);
        try testing.expectEqual(lineWidth(font, std.mem.trimEnd(u8, want, " "), 8, layout.letter_spacing), line.width);
    }
    try testing.expectEqual(null, lines.next());
}

test "paragraphs without wrapping" {
    const font: Font = .{ .bitmap = font8x8.basic };
    try expectLines(font, "", null, .default, &.{""});
    try expectLines(font, "ab", null, .default, &.{"ab"});
    try expectLines(font, "ab\ncd", null, .default, &.{ "ab", "cd" });
    try expectLines(font, "ab\n", null, .default, &.{ "ab", "" });
    try expectLines(font, "\n\nx", null, .default, &.{ "", "", "x" });
    // Width limits are ignored unless wrapping is on.
    try expectLines(font, "aaa bbb ccc", 64, .default, &.{"aaa bbb ccc"});
}

test "word wrap" {
    const font: Font = .{ .bitmap = font8x8.basic };
    const wrap: TextLayout = .{ .wrap = true };
    // 8 px per glyph: "aaa bbb" is 56 px, "aaa bbb ccc" 88.
    try expectLines(font, "aaa bbb ccc", 64, wrap, &.{ "aaa bbb", "ccc" });
    try expectLines(font, "aaa bbb ccc", 56, wrap, &.{ "aaa bbb", "ccc" });
    try expectLines(font, "aaa bbb ccc", 55, wrap, &.{ "aaa", "bbb", "ccc" });
    // Spaces at the break are consumed, trailing ones are kept out of the width.
    try expectLines(font, "aaa   bbb  ", 40, wrap, &.{ "aaa", "bbb  " });
    // A word wider than the box breaks between codepoints, at least one per line.
    try expectLines(font, "abcdefgh ij", 24, wrap, &.{ "abc", "def", "gh", "ij" });
    try expectLines(font, "abc", 4, wrap, &.{ "a", "b", "c" });
    try expectLines(font, "ééé", 16, wrap, &.{ "éé", "é" });
    // Explicit breaks still apply inside wrapped text.
    try expectLines(font, "aaa bbb\nccc ddd", 40, wrap, &.{ "aaa", "bbb", "ccc", "ddd" });
    try expectLines(font, "aaa\n\nbbb", 40, wrap, &.{ "aaa", "", "bbb" });
}

test "spacing and measure" {
    const font: Font = .{ .bitmap = font8x8.basic };
    try testing.expectEqual(24, lineWidth(font, "abc", 8, 0));
    try testing.expectEqual(28, lineWidth(font, "abc", 8, 2));
    try testing.expectEqual(20, lineWidth(font, "abc", 8, -2));
    try testing.expectEqual(0, lineWidth(font, "abc", 8, -20));
    try testing.expectEqual(0, lineWidth(font, "", 8, 5));
    try testing.expectEqual(8, lineWidth(font, "a", 8, 5));

    const spaced: TextLayout = .{ .wrap = true, .line_spacing = 1.5, .letter_spacing = 2 };
    try testing.expectEqual(12, lineAdvance(font, 8, spaced));
    // "aaa bbb" would be 68 px with tracking, so it wraps at 64.
    const box = measure(font, "aaa bbb ccc", 8, 64, spaced);
    try testing.expectEqual(Rectangle(f32){ .l = 0, .t = 0, .r = 28, .b = 36 }, box);
    try testing.expectEqual(Rectangle(f32){ .l = 0, .t = 0, .r = 0, .b = 8 }, measure(font, "", 8, null, .default));
    try testing.expectEqual(font.getTextBounds("ab\ncd", 8), measure(font, "ab\ncd", 8, null, .default));
}

test "vector widths follow kerning" {
    const synthetic = @import("truetype/synthetic.zig");
    var buf: [synthetic.buffer_size]u8 = undefined;
    const font: Font = .{ .vector = synthetic.font(&buf, .{}) };
    // A then B kerns by -30 units; at 1000 px per em that is 800 - 30 + 800.
    try testing.expectEqual(font.getTextBounds("AB", 1000).r, lineWidth(font, "AB", 1000, 0));
    try testing.expectEqual(1570 + 10, lineWidth(font, "AB", 1000, 10));
    var lines: Lines = .init(font, "AB AB", 1000, 1600, .{ .wrap = true });
    try testing.expectEqualStrings("AB", lines.next().?.text);
    try testing.expectEqualStrings("AB", lines.next().?.text);
    try testing.expectEqual(null, lines.next());
}
