//! Font system for zignal
//!
//! This module provides bitmap font rendering capabilities including:
//! - Default 8x8 bitmap font
//! - BDF font loading with Unicode support
//! - Variable-width font support
//!
//! The font system is organized into subdirectories for better modularity.

const std = @import("std");

// Core font types
pub const BitmapFont = @import("font/BitmapFont.zig");
pub const GlyphData = @import("font/GlyphData.zig");

// Font loading options
pub const LoadOptions = struct {
    /// Load all characters in the font (default: false, loads only ASCII)
    load_all: bool = false,
    /// Specific Unicode ranges to load (null = use default behavior)
    ranges: ?[]const unicode.Range = null,
    /// Maximum characters to load (null = no limit)
    max_chars: ?usize = null,
};

// Default font
const default_8x8 = @import("font/default_8x8.zig");
pub const default_font_8x8 = default_8x8.font;

// Unicode utilities
pub const unicode = @import("font/unicode.zig");

// Format detection
pub const FontFormat = @import("font/format.zig").FontFormat;

// BDF font support
pub const bdf = @import("font/bdf.zig");
