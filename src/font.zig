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

// Default font
const default_8x8 = @import("font/default_8x8.zig");
pub const default_font_8x8 = default_8x8.font;

// Unicode utilities
pub const unicode = @import("font/unicode.zig");

// BDF font support
pub const bdf = @import("font/bdf.zig");
