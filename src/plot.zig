//! Plot module for creating data visualizations using zignal's Canvas
//!
//! This module provides a high-level plotting API for creating various types
//! of data visualizations including line plots, scatter plots, bar charts,
//! and more. It builds on top of zignal's Canvas module for rendering.

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const Image = @import("image.zig").Image;
const Canvas = @import("canvas.zig").Canvas;
const DrawMode = @import("canvas.zig").DrawMode;
const Rgb = @import("color.zig").Rgb;
const Point2d = @import("geometry/Point.zig").Point2d;
const Rectangle = @import("geometry.zig").Rectangle;
const BitmapFont = @import("font.zig").BitmapFont;
const default_font_8x8 = @import("font.zig").default_font_8x8;

/// Margins around the plot area
pub const Margins = struct {
    left: f32 = 60,
    right: f32 = 40,
    top: f32 = 40,
    bottom: f32 = 60,
};

/// Types of plots that can be rendered
pub const SeriesType = enum {
    line,
    scatter,
    bar,
    area,
};

/// Marker types for scatter plots
pub const MarkerType = enum {
    circle,
    square,
    triangle,
    cross,
    plus,
    diamond,
};

/// Styling options for a data series
pub const SeriesStyle = struct {
    color: Rgb = .{ .r = 0, .g = 0, .b = 255 },
    line_width: usize = 2,
    marker_type: MarkerType = .circle,
    marker_size: f32 = 4,
    fill_alpha: f32 = 0.3,
};

/// A data series to be plotted
pub const Series = struct {
    type: SeriesType,
    x_data: []const f32,
    y_data: []const f32,
    style: SeriesStyle,
    label: ?[]const u8 = null,
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

/// Main plot structure
pub const Plot = struct {
    allocator: Allocator,
    image: Image(Rgb),
    canvas: Canvas(Rgb),

    // Plot area (excluding margins)
    plot_area: Rectangle(f32),

    // Data ranges
    x_range: Range,
    y_range: Range,
    auto_range: bool = true,

    // Styling
    background_color: Rgb = .{ .r = 255, .g = 255, .b = 255 },
    grid_color: Rgb = .{ .r = 200, .g = 200, .b = 200 },
    axis_color: Rgb = .{ .r = 0, .g = 0, .b = 0 },

    // Text properties
    title: ?[]const u8 = null,
    x_label: ?[]const u8 = null,
    y_label: ?[]const u8 = null,
    font: BitmapFont = default_font_8x8,
    font_scale: f32 = 1.5,

    // Data series
    series: std.ArrayList(Series),

    // Layout
    margins: Margins = .{},
    show_grid: bool = true,
    show_legend: bool = false,

    /// Initialize a new plot with given dimensions
    pub fn init(allocator: Allocator, width: usize, height: usize) !Plot {
        var image = try Image(Rgb).initAlloc(allocator, height, width);
        errdefer image.deinit(allocator);

        const canvas = Canvas(Rgb).init(allocator, image);

        const plot_area = Rectangle(f32){
            .l = @as(f32, @floatFromInt(width)) * 0.1,
            .t = @as(f32, @floatFromInt(height)) * 0.1,
            .r = @as(f32, @floatFromInt(width)) * 0.9,
            .b = @as(f32, @floatFromInt(height)) * 0.9,
        };

        return .{
            .allocator = allocator,
            .image = image,
            .canvas = canvas,
            .plot_area = plot_area,
            .x_range = .{ .min = 0, .max = 1 },
            .y_range = .{ .min = 0, .max = 1 },
            .series = std.ArrayList(Series).init(allocator),
        };
    }

    /// Deinitialize the plot
    pub fn deinit(self: *Plot) void {
        self.series.deinit();
        self.image.deinit(self.allocator);
    }

    /// Set the plot title
    pub fn setTitle(self: *Plot, title: []const u8) void {
        self.title = title;
    }

    /// Set the X-axis label
    pub fn setXLabel(self: *Plot, label: []const u8) void {
        self.x_label = label;
    }

    /// Set the Y-axis label
    pub fn setYLabel(self: *Plot, label: []const u8) void {
        self.y_label = label;
    }

    /// Set the X-axis range manually
    pub fn setXRange(self: *Plot, min: f32, max: f32) void {
        self.x_range = .{ .min = min, .max = max };
        self.auto_range = false;
    }

    /// Set the Y-axis range manually
    pub fn setYRange(self: *Plot, min: f32, max: f32) void {
        self.y_range = .{ .min = min, .max = max };
        self.auto_range = false;
    }

    /// Enable/disable grid
    pub fn showGrid(self: *Plot, show: bool) void {
        self.show_grid = show;
    }

    /// Enable/disable legend
    pub fn showLegend(self: *Plot, show: bool) void {
        self.show_legend = show;
    }

    /// Add a line plot
    pub fn addLine(self: *Plot, x_data: []const f32, y_data: []const f32, style: SeriesStyle) !void {
        assert(x_data.len == y_data.len);
        try self.series.append(.{
            .type = .line,
            .x_data = x_data,
            .y_data = y_data,
            .style = style,
            .label = null,
        });
    }

    /// Add a line plot with label
    pub fn addLineWithLabel(self: *Plot, x_data: []const f32, y_data: []const f32, label: []const u8, style: SeriesStyle) !void {
        assert(x_data.len == y_data.len);
        try self.series.append(.{
            .type = .line,
            .x_data = x_data,
            .y_data = y_data,
            .style = style,
            .label = label,
        });
    }

    /// Add a scatter plot
    pub fn addScatter(self: *Plot, x_data: []const f32, y_data: []const f32, style: SeriesStyle) !void {
        assert(x_data.len == y_data.len);
        try self.series.append(.{
            .type = .scatter,
            .x_data = x_data,
            .y_data = y_data,
            .style = style,
            .label = null,
        });
    }

    /// Add a scatter plot with label
    pub fn addScatterWithLabel(self: *Plot, x_data: []const f32, y_data: []const f32, label: []const u8, style: SeriesStyle) !void {
        assert(x_data.len == y_data.len);
        try self.series.append(.{
            .type = .scatter,
            .x_data = x_data,
            .y_data = y_data,
            .style = style,
            .label = label,
        });
    }

    /// Update plot area based on margins
    fn updatePlotArea(self: *Plot) void {
        const width = @as(f32, @floatFromInt(self.image.cols));
        const height = @as(f32, @floatFromInt(self.image.rows));

        // Adjust left margin if Y-axis label is present
        var left_margin = self.margins.left;
        if (self.y_label != null) {
            // Add extra space for Y-axis label (text height * scale + padding)
            left_margin += 20 * self.font_scale;
        }

        self.plot_area = .{
            .l = left_margin,
            .t = self.margins.top,
            .r = width - self.margins.right,
            .b = height - self.margins.bottom,
        };
    }

    /// Calculate data ranges from all series
    fn calculateDataRanges(self: *Plot) void {
        if (!self.auto_range) return;

        var x_min: f32 = std.math.floatMax(f32);
        var x_max: f32 = -std.math.floatMax(f32);
        var y_min: f32 = std.math.floatMax(f32);
        var y_max: f32 = -std.math.floatMax(f32);

        for (self.series.items) |series| {
            for (series.x_data) |x| {
                x_min = @min(x_min, x);
                x_max = @max(x_max, x);
            }
            for (series.y_data) |y| {
                y_min = @min(y_min, y);
                y_max = @max(y_max, y);
            }
        }

        // Apply nice rounding
        self.x_range = Range.fromData(&.{ x_min, x_max }, 0.05).nice();
        self.y_range = Range.fromData(&.{ y_min, y_max }, 0.05).nice();
    }

    /// Convert data coordinates to pixel coordinates
    fn dataToPixel(self: Plot, x: f32, y: f32) Point2d(f32) {
        const px = self.plot_area.l + (x - self.x_range.min) /
            (self.x_range.max - self.x_range.min) * (self.plot_area.r - self.plot_area.l);
        const py = self.plot_area.b - (y - self.y_range.min) /
            (self.y_range.max - self.y_range.min) * (self.plot_area.b - self.plot_area.t);
        return .init2d(px, py);
    }

    /// Draw the background
    fn drawBackground(self: Plot) void {
        self.canvas.fill(self.background_color);
    }

    /// Draw the grid
    fn drawGrid(self: Plot) !void {
        if (!self.show_grid) return;

        // Generate tick positions
        const x_ticks = try self.x_range.generateTicks(self.allocator, 8);
        defer self.allocator.free(x_ticks);
        const y_ticks = try self.y_range.generateTicks(self.allocator, 8);
        defer self.allocator.free(y_ticks);

        // Vertical grid lines
        for (x_ticks) |tick_val| {
            const px = self.dataToPixel(tick_val, 0).x();
            if (px >= self.plot_area.l and px <= self.plot_area.r) {
                self.canvas.drawLine(.init2d(px, self.plot_area.t), .init2d(px, self.plot_area.b), self.grid_color, 1, .fast);
            }
        }

        // Horizontal grid lines
        for (y_ticks) |tick_val| {
            const py = self.dataToPixel(0, tick_val).y();
            if (py >= self.plot_area.t and py <= self.plot_area.b) {
                self.canvas.drawLine(.init2d(self.plot_area.l, py), .init2d(self.plot_area.r, py), self.grid_color, 1, .fast);
            }
        }
    }

    /// Draw the axes
    fn drawAxes(self: Plot) void {
        // X-axis
        self.canvas.drawLine(.init2d(self.plot_area.l, self.plot_area.b), .init2d(self.plot_area.r, self.plot_area.b), self.axis_color, 2, .fast);

        // Y-axis
        self.canvas.drawLine(.init2d(self.plot_area.l, self.plot_area.t), .init2d(self.plot_area.l, self.plot_area.b), self.axis_color, 2, .fast);
    }

    /// Draw tick marks and labels
    fn drawTicks(self: Plot) !void {
        var buffer: [32]u8 = undefined;
        const tick_size: f32 = 5;

        // Generate tick positions
        const x_ticks = try self.x_range.generateTicks(self.allocator, 8);
        defer self.allocator.free(x_ticks);
        const y_ticks = try self.y_range.generateTicks(self.allocator, 8);
        defer self.allocator.free(y_ticks);

        // X-axis ticks and labels
        for (x_ticks) |tick_val| {
            const px = self.dataToPixel(tick_val, 0).x();
            if (px >= self.plot_area.l and px <= self.plot_area.r) {
                // Draw tick mark
                self.canvas.drawLine(.init2d(px, self.plot_area.b), .init2d(px, self.plot_area.b + tick_size), self.axis_color, 1, .fast);

                // Draw label
                const label = formatTickValue(tick_val, &buffer);
                const text_width = @as(f32, @floatFromInt(label.len * 8));
                self.canvas.drawText(label, .init2d(px - text_width / 2, self.plot_area.b + tick_size + 5), self.font, self.axis_color, 1, .fast);
            }
        }

        // Y-axis ticks and labels
        for (y_ticks) |tick_val| {
            const py = self.dataToPixel(0, tick_val).y();
            if (py >= self.plot_area.t and py <= self.plot_area.b) {
                // Draw tick mark
                self.canvas.drawLine(.init2d(self.plot_area.l - tick_size, py), .init2d(self.plot_area.l, py), self.axis_color, 1, .fast);

                // Draw label
                const label = formatTickValue(tick_val, &buffer);
                const text_width = @as(f32, @floatFromInt(label.len * 8));
                self.canvas.drawText(label, .init2d(self.plot_area.l - tick_size - text_width - 5, py - 4), self.font, self.axis_color, 1, .fast);
            }
        }
    }

    /// Format a tick value to a string
    fn formatTickValue(value: f32, buffer: []u8) []const u8 {
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

    /// Draw text labels
    fn drawLabels(self: Plot) !void {
        // Title
        if (self.title) |title| {
            const x = (self.plot_area.l + self.plot_area.r) / 2 -
                @as(f32, @floatFromInt(title.len * 8)) * self.font_scale / 2;
            const y = self.margins.top / 2 - 4 * self.font_scale;
            self.canvas.drawText(title, .init2d(x, y), self.font, self.axis_color, self.font_scale, .fast);
        }

        // X-axis label
        if (self.x_label) |label| {
            const x = (self.plot_area.l + self.plot_area.r) / 2 -
                @as(f32, @floatFromInt(label.len * 8)) * self.font_scale / 2;
            const y = self.plot_area.b + self.margins.bottom / 2;
            self.canvas.drawText(label, .init2d(x, y), self.font, self.axis_color, self.font_scale, .fast);
        }

        // Y-axis label (rotated 90 degrees counter-clockwise)
        if (self.y_label) |label| {
            // Create a temporary image for the text
            const text_width = @as(usize, @intFromFloat(@as(f32, @floatFromInt(label.len * 8)) * self.font_scale));
            const text_height = @as(usize, @intFromFloat(8 * self.font_scale));

            var temp_image = try Image(Rgb).initAlloc(self.allocator, text_height + 4, text_width + 4);
            defer temp_image.deinit(self.allocator);

            // Create canvas and fill with background color
            const temp_canvas = Canvas(Rgb).init(self.allocator, temp_image);
            temp_canvas.fill(self.background_color);

            // Draw text horizontally on temp image
            temp_canvas.drawText(label, .init2d(2, 2), self.font, self.axis_color, self.font_scale, .fast);

            // Rotate 90 degrees counter-clockwise
            var rotated_image: Image(Rgb) = .empty;
            try temp_image.rotate(self.allocator, -std.math.pi / 2.0, &rotated_image);
            defer rotated_image.deinit(self.allocator);

            // Calculate position for rotated text (centered vertically on Y-axis)
            // Position between left edge and plot area
            const x = (self.plot_area.l - 20 * self.font_scale) / 2 - @as(f32, @floatFromInt(rotated_image.cols)) / 2;
            const y = (self.plot_area.t + self.plot_area.b) / 2 - @as(f32, @floatFromInt(rotated_image.rows)) / 2;

            // Copy rotated text to main image
            const dest_x = @as(usize, @intFromFloat(@max(0, x)));
            const dest_y = @as(usize, @intFromFloat(@max(0, y)));

            for (0..rotated_image.rows) |r| {
                for (0..rotated_image.cols) |c| {
                    const src_pixel = rotated_image.at(r, c).*;
                    // Only copy non-background pixels
                    if (src_pixel.r != self.background_color.r or
                        src_pixel.g != self.background_color.g or
                        src_pixel.b != self.background_color.b)
                    {
                        if (dest_y + r < self.image.rows and dest_x + c < self.image.cols) {
                            self.image.at(dest_y + r, dest_x + c).* = src_pixel;
                        }
                    }
                }
            }
        }
    }

    /// Draw a line series
    fn drawLineSeries(self: Plot, series: Series) void {
        if (series.x_data.len < 2) return;

        var i: usize = 0;
        while (i < series.x_data.len - 1) : (i += 1) {
            const p1 = self.dataToPixel(series.x_data[i], series.y_data[i]);
            const p2 = self.dataToPixel(series.x_data[i + 1], series.y_data[i + 1]);

            self.canvas.drawLine(p1, p2, series.style.color, series.style.line_width, .soft);
        }
    }

    /// Draw a scatter series
    fn drawScatterSeries(self: Plot, series: Series) void {
        for (series.x_data, series.y_data) |x, y| {
            const p = self.dataToPixel(x, y);
            const size = series.style.marker_size;

            switch (series.style.marker_type) {
                .circle => {
                    self.canvas.fillCircle(p, size, series.style.color, .soft);
                },
                .square => {
                    const half = size / 2;
                    self.canvas.fillRectangle(.{
                        .l = p.x() - half,
                        .t = p.y() - half,
                        .r = p.x() + half,
                        .b = p.y() + half,
                    }, series.style.color, .fast);
                },
                .triangle => {
                    // Equilateral triangle pointing up
                    const h = size * 0.866; // height = size * sqrt(3)/2
                    const half_base = size / 2;
                    const points = [_]Point2d(f32){
                        .init2d(p.x(), p.y() - h * 0.67), // top
                        .init2d(p.x() - half_base, p.y() + h * 0.33), // bottom left
                        .init2d(p.x() + half_base, p.y() + h * 0.33), // bottom right
                    };
                    self.canvas.fillPolygon(&points, series.style.color, .soft) catch {};
                },
                .cross => {
                    // × shape
                    const half = size / 2;
                    const line_width = @max(1, @as(usize, @intFromFloat(size / 4)));
                    self.canvas.drawLine(.init2d(p.x() - half, p.y() - half), .init2d(p.x() + half, p.y() + half), series.style.color, line_width, .soft);
                    self.canvas.drawLine(.init2d(p.x() + half, p.y() - half), .init2d(p.x() - half, p.y() + half), series.style.color, line_width, .soft);
                },
                .plus => {
                    // + shape
                    const half = size / 2;
                    const line_width = @max(1, @as(usize, @intFromFloat(size / 4)));
                    self.canvas.drawLine(.init2d(p.x() - half, p.y()), .init2d(p.x() + half, p.y()), series.style.color, line_width, .soft);
                    self.canvas.drawLine(.init2d(p.x(), p.y() - half), .init2d(p.x(), p.y() + half), series.style.color, line_width, .soft);
                },
                .diamond => {
                    // Diamond shape
                    const half = size / 2;
                    const points = [_]Point2d(f32){
                        .init2d(p.x(), p.y() - half), // top
                        .init2d(p.x() + half, p.y()), // right
                        .init2d(p.x(), p.y() + half), // bottom
                        .init2d(p.x() - half, p.y()), // left
                    };
                    self.canvas.fillPolygon(&points, series.style.color, .soft) catch {};
                },
            }
        }
    }

    /// Draw the legend
    fn drawLegend(self: Plot) void {
        if (!self.show_legend) return;

        // Count series with labels
        var labeled_count: usize = 0;
        for (self.series.items) |series| {
            if (series.label != null) labeled_count += 1;
        }
        if (labeled_count == 0) return;

        // Legend dimensions
        const padding: f32 = 10;
        const item_height: f32 = 20;
        const sample_width: f32 = 30;
        const text_offset: f32 = 5;

        // Calculate legend size
        var max_label_width: f32 = 0;
        for (self.series.items) |series| {
            if (series.label) |label| {
                const label_width = @as(f32, @floatFromInt(label.len * 8));
                max_label_width = @max(max_label_width, label_width);
            }
        }

        const legend_width = padding * 2 + sample_width + text_offset + max_label_width;
        const legend_height = padding * 2 + @as(f32, @floatFromInt(labeled_count)) * item_height;

        // Position legend in top-right corner of plot area
        const legend_x = self.plot_area.r - legend_width - 10;
        const legend_y = self.plot_area.t + 10;

        // Draw legend background
        self.canvas.fillRectangle(.{
            .l = legend_x,
            .t = legend_y,
            .r = legend_x + legend_width,
            .b = legend_y + legend_height,
        }, Rgb{ .r = 255, .g = 255, .b = 255 }, .fast);

        // Draw legend border
        self.canvas.drawRectangle(.{
            .l = legend_x,
            .t = legend_y,
            .r = legend_x + legend_width,
            .b = legend_y + legend_height,
        }, Rgb{ .r = 200, .g = 200, .b = 200 }, 1, .fast);

        // Draw legend items
        var y_offset: f32 = legend_y + padding;
        for (self.series.items) |series| {
            const label = series.label orelse continue;

            const sample_x = legend_x + padding;
            const sample_y = y_offset + item_height / 2;

            // Draw sample based on series type
            switch (series.type) {
                .line => {
                    self.canvas.drawLine(.init2d(sample_x, sample_y), .init2d(sample_x + sample_width, sample_y), series.style.color, series.style.line_width, .soft);
                },
                .scatter => {
                    const marker_x = sample_x + sample_width / 2;
                    // Draw a single marker
                    switch (series.style.marker_type) {
                        .circle => {
                            self.canvas.fillCircle(.init2d(marker_x, sample_y), series.style.marker_size, series.style.color, .soft);
                        },
                        .square => {
                            const half = series.style.marker_size / 2;
                            self.canvas.fillRectangle(.{
                                .l = marker_x - half,
                                .t = sample_y - half,
                                .r = marker_x + half,
                                .b = sample_y + half,
                            }, series.style.color, .fast);
                        },
                        .triangle => {
                            const size = series.style.marker_size;
                            const h = size * 0.866;
                            const half_base = size / 2;
                            const points = [_]Point2d(f32){
                                .init2d(marker_x, sample_y - h * 0.67),
                                .init2d(marker_x - half_base, sample_y + h * 0.33),
                                .init2d(marker_x + half_base, sample_y + h * 0.33),
                            };
                            self.canvas.fillPolygon(&points, series.style.color, .soft) catch {};
                        },
                        .cross => {
                            const half = series.style.marker_size / 2;
                            const line_width = @max(1, @as(usize, @intFromFloat(series.style.marker_size / 4)));
                            self.canvas.drawLine(.init2d(marker_x - half, sample_y - half), .init2d(marker_x + half, sample_y + half), series.style.color, line_width, .soft);
                            self.canvas.drawLine(.init2d(marker_x + half, sample_y - half), .init2d(marker_x - half, sample_y + half), series.style.color, line_width, .soft);
                        },
                        .plus => {
                            const half = series.style.marker_size / 2;
                            const line_width = @max(1, @as(usize, @intFromFloat(series.style.marker_size / 4)));
                            self.canvas.drawLine(.init2d(marker_x - half, sample_y), .init2d(marker_x + half, sample_y), series.style.color, line_width, .soft);
                            self.canvas.drawLine(.init2d(marker_x, sample_y - half), .init2d(marker_x, sample_y + half), series.style.color, line_width, .soft);
                        },
                        .diamond => {
                            const half = series.style.marker_size / 2;
                            const points = [_]Point2d(f32){
                                .init2d(marker_x, sample_y - half),
                                .init2d(marker_x + half, sample_y),
                                .init2d(marker_x, sample_y + half),
                                .init2d(marker_x - half, sample_y),
                            };
                            self.canvas.fillPolygon(&points, series.style.color, .soft) catch {};
                        },
                    }
                },
                else => {},
            }

            // Draw label text
            const text_x = sample_x + sample_width + text_offset;
            const text_y = sample_y - 4; // Center text vertically
            self.canvas.drawText(label, .init2d(text_x, text_y), self.font, self.axis_color, 1, .fast);

            y_offset += item_height;
        }
    }

    /// Render the plot
    pub fn render(self: *Plot) !void {
        // Update layout
        self.updatePlotArea();
        self.calculateDataRanges();

        // Draw components in order
        self.drawBackground();
        try self.drawGrid();
        self.drawAxes();
        try self.drawTicks();
        try self.drawLabels();

        // Draw all series
        for (self.series.items) |series| {
            switch (series.type) {
                .line => self.drawLineSeries(series),
                .scatter => self.drawScatterSeries(series),
                else => {}, // TODO: Implement other types
            }
        }

        // Draw legend if enabled
        self.drawLegend();
    }

    /// Save the plot to a PNG file
    pub fn save(self: Plot, path: []const u8) !void {
        try self.image.save(self.allocator, path);
    }
};

test "Plot initialization" {
    const allocator = testing.allocator;

    var plot = try Plot.init(allocator, 800, 600);
    defer plot.deinit();

    try testing.expectEqual(@as(usize, 800), plot.image.cols);
    try testing.expectEqual(@as(usize, 600), plot.image.rows);
}

test "Plot with line series" {
    const allocator = testing.allocator;

    var plot = try Plot.init(allocator, 400, 300);
    defer plot.deinit();

    const x_data = [_]f32{ 0, 1, 2, 3, 4 };
    const y_data = [_]f32{ 0, 1, 4, 9, 16 };

    try plot.addLine(&x_data, &y_data, .{
        .color = .{ .r = 255, .g = 0, .b = 0 },
        .line_width = 2,
    });

    plot.setTitle("Test Plot");
    plot.setXLabel("X Values");
    plot.setYLabel("Y Values");

    try plot.render();

    // Verify some pixels are not white (plot was drawn)
    var non_white_pixels: usize = 0;
    for (plot.image.data) |pixel| {
        if (pixel.r != 255 or pixel.g != 255 or pixel.b != 255) {
            non_white_pixels += 1;
        }
    }

    try testing.expect(non_white_pixels > 100);
}

test "Range calculations" {
    const data = [_]f32{ -5, 3, 8, -2, 15 };
    const range = Range.fromData(&data, 0.1);

    try testing.expect(range.min < -5);
    try testing.expect(range.max > 15);

    const nice_range = range.nice();
    try testing.expectEqual(@as(f32, -10), nice_range.min);
    try testing.expectEqual(@as(f32, 20), nice_range.max);
}
