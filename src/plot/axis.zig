//! Axis module for the plot system
//!
//! This module provides axis-related functionality including:
//! - Range calculations with nice number rounding
//! - Tick generation with customizable intervals
//! - Coordinate transformations between data and pixel space
//! - Support for different axis scales (linear, logarithmic, etc.)

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Types of axis scales supported
pub const AxisScale = enum {
    linear,
    logarithmic, // TODO: Implement
    datetime, // TODO: Implement
    categorical, // TODO: Implement
};

/// Range of values for an axis
pub const Range = struct {
    min: f32,
    max: f32,

    /// Calculate range from data with optional padding
    pub fn fromData(data: []const f32, padding_pct: f32) Range {
        if (data.len == 0) return .{ .min = 0, .max = 1 };

        var min = data[0];
        var max = data[0];
        for (data[1..]) |val| {
            min = @min(min, val);
            max = @max(max, val);
        }

        // Add padding
        const range = max - min;
        const padding = range * padding_pct;
        return .{
            .min = min - padding,
            .max = max + padding,
        };
    }

    /// Get nice round numbers for the range
    pub fn nice(self: Range) Range {
        const range = self.max - self.min;
        const magnitude = std.math.pow(f32, 10, @floor(std.math.log10(range)));
        const normalized = range / magnitude;

        // Find nice interval
        const nice_interval: f32 = if (normalized <= 1) 1 else if (normalized <= 2) 2 else if (normalized <= 5) 5 else 10;

        const interval = nice_interval * magnitude;
        return .{
            .min = @floor(self.min / interval) * interval,
            .max = @ceil(self.max / interval) * interval,
        };
    }

    /// Generate nice tick positions for this range
    pub fn generateTicks(self: Range, allocator: Allocator, target_count: usize) ![]f32 {
        const range = self.max - self.min;
        if (range <= 0) return try allocator.alloc(f32, 0);

        // Calculate nice tick interval
        const rough_interval = range / @as(f32, @floatFromInt(target_count));
        const magnitude = std.math.pow(f32, 10, @floor(std.math.log10(rough_interval)));
        const normalized = rough_interval / magnitude;

        const tick_interval = blk: {
            if (normalized <= 1) break :blk magnitude;
            if (normalized <= 2) break :blk 2 * magnitude;
            if (normalized <= 5) break :blk 5 * magnitude;
            break :blk 10 * magnitude;
        };

        // Generate ticks
        const first_tick = @ceil(self.min / tick_interval) * tick_interval;
        const tick_count = @as(usize, @intFromFloat(@floor((self.max - first_tick) / tick_interval))) + 1;

        var ticks = try allocator.alloc(f32, tick_count);
        for (0..tick_count) |i| {
            ticks[i] = first_tick + @as(f32, @floatFromInt(i)) * tick_interval;
        }

        return ticks;
    }
};

/// Axis structure that encapsulates scale, range, and transformations
pub const Axis = struct {
    scale: AxisScale = .linear,
    range: Range,
    pixel_min: f32,
    pixel_max: f32,
    inverted: bool = false,

    /// Initialize an axis with data range and pixel range
    pub fn init(range: Range, pixel_min: f32, pixel_max: f32) Axis {
        return .{
            .range = range,
            .pixel_min = pixel_min,
            .pixel_max = pixel_max,
        };
    }

    /// Initialize an axis from data with automatic range calculation
    pub fn fromData(data: []const f32, pixel_min: f32, pixel_max: f32, padding_pct: f32) Axis {
        const range = Range.fromData(data, padding_pct).nice();
        return init(range, pixel_min, pixel_max);
    }

    /// Convert data coordinate to pixel coordinate
    pub fn dataToPixel(self: Axis, value: f32) f32 {
        const t = (value - self.range.min) / (self.range.max - self.range.min);
        if (self.inverted) {
            return self.pixel_max - t * (self.pixel_max - self.pixel_min);
        } else {
            return self.pixel_min + t * (self.pixel_max - self.pixel_min);
        }
    }

    /// Convert pixel coordinate to data coordinate
    pub fn pixelToData(self: Axis, pixel: f32) f32 {
        const t = if (self.inverted)
            (self.pixel_max - pixel) / (self.pixel_max - self.pixel_min)
        else
            (pixel - self.pixel_min) / (self.pixel_max - self.pixel_min);
        return self.range.min + t * (self.range.max - self.range.min);
    }

    /// Generate tick positions in data space
    pub fn generateTicks(self: Axis, allocator: Allocator, target_count: usize) ![]f32 {
        return try self.range.generateTicks(allocator, target_count);
    }

    /// Format a tick value to a string
    pub fn formatTickValue(value: f32, buffer: []u8) []const u8 {
        // Simple formatting - could be improved
        const abs_val = @abs(value);

        // Format based on magnitude
        if (abs_val >= 100) {
            return std.fmt.bufPrint(buffer, "{d:.0}", .{value}) catch "?";
        } else if (abs_val >= 10) {
            return std.fmt.bufPrint(buffer, "{d:.1}", .{value}) catch "?";
        } else if (abs_val >= 1) {
            return std.fmt.bufPrint(buffer, "{d:.1}", .{value}) catch "?";
        } else {
            return std.fmt.bufPrint(buffer, "{d:.2}", .{value}) catch "?";
        }
    }
};

// Tests
test "Range calculations" {
    const testing = std.testing;

    const data = [_]f32{ -5, 3, 8, -2, 15 };
    const range = Range.fromData(&data, 0.1);

    try testing.expect(range.min < -5);
    try testing.expect(range.max > 15);

    const nice_range = range.nice();
    try testing.expectEqual(@as(f32, -10), nice_range.min);
    try testing.expectEqual(@as(f32, 20), nice_range.max);
}

test "Axis coordinate transformations" {
    const testing = std.testing;

    const axis = Axis.init(.{ .min = 0, .max = 10 }, 100, 500);

    // Test data to pixel
    try testing.expectEqual(@as(f32, 100), axis.dataToPixel(0));
    try testing.expectEqual(@as(f32, 300), axis.dataToPixel(5));
    try testing.expectEqual(@as(f32, 500), axis.dataToPixel(10));

    // Test pixel to data
    try testing.expectEqual(@as(f32, 0), axis.pixelToData(100));
    try testing.expectEqual(@as(f32, 5), axis.pixelToData(300));
    try testing.expectEqual(@as(f32, 10), axis.pixelToData(500));
}

test "Inverted axis" {
    const testing = std.testing;

    var axis = Axis.init(.{ .min = 0, .max = 10 }, 100, 500);
    axis.inverted = true;

    // For inverted axis (like Y-axis in plots), pixel coordinates are flipped
    try testing.expectEqual(@as(f32, 500), axis.dataToPixel(0));
    try testing.expectEqual(@as(f32, 300), axis.dataToPixel(5));
    try testing.expectEqual(@as(f32, 100), axis.dataToPixel(10));
}
