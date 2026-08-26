//! This module provides a Canvas for drawing various shapes and lines on images.

const std = @import("std");
const assert = std.debug.assert;
const clamp = std.math.clamp;

const convertColor = @import("../color.zig").convertColor;
const isColor = @import("../color.zig").isColor;
const Rgb = @import("../color.zig").Rgb(u8);
const Rgba = @import("../color.zig").Rgba(u8);
const Blending = @import("../color.zig").Blending;
const BitmapFont = @import("../font.zig").BitmapFont;
const Font = @import("../font.zig").Font;
const VectorFont = @import("../font.zig").VectorFont;
const Outline = @import("../font.zig").Outline;
const GlyphCache = @import("../font.zig").GlyphCache;
const TextLayout = @import("../font.zig").TextLayout;
const text_layout = @import("../font.zig").layout;
const Rectangle = @import("../geometry.zig").Rectangle;
const Point = @import("../geometry/Point.zig").Point;
const Image = @import("../image.zig").Image;
const assignPixel = @import("../image.zig").assignPixel;
const as = @import("../meta.zig").as;

// The pixel-type independent machinery, by concern.
const polygons = @import("polygon.zig");
const area_block = polygons.area_block;
const Edge = polygons.Edge;
const Crossing = polygons.Crossing;
const polygonEdges = polygons.polygonEdges;
const EdgeSweep = polygons.EdgeSweep;
const scanlineCrossings = polygons.scanlineCrossings;
const sortCrossings = polygons.sortCrossings;
const SpanIter = polygons.SpanIter;
const CoverageCell = polygons.CoverageCell;
const touchBlocks = polygons.touchBlocks;
const accumulateEdge = polygons.accumulateEdge;
const strokes = @import("stroke.zig");
const StrokeBuilder = strokes.StrokeBuilder;
const perpendicular = strokes.perpendicular;
const signedArea = strokes.signedArea;
const unitDirection = strokes.unitDirection;
const curves = @import("curve.zig");
const bezier_max_segments_count = curves.bezier_max_segments_count;
const spline_max_segments_count = curves.spline_max_segments_count;
const spline_min_segments_count = curves.spline_min_segments_count;
const quadratic_min_segments_count = curves.quadratic_min_segments_count;
const pixels_per_segment_soft = curves.pixels_per_segment_soft;
const pixels_per_segment_fast = curves.pixels_per_segment_fast;
const pixels_per_segment_quadratic = curves.pixels_per_segment_quadratic;
const CubicBezier = curves.CubicBezier;
const evalQuadraticBezier = curves.evalQuadraticBezier;
const evalCubicBezier = curves.evalCubicBezier;
const estimateQuadraticBezierLength = curves.estimateQuadraticBezierLength;
const estimateCubicBezierLength = curves.estimateCubicBezierLength;
const bezierSegments = curves.bezierSegments;
const tessellateBezier = curves.tessellateBezier;
const calculateSmoothControlPoints = curves.calculateSmoothControlPoints;
const splineEdge = curves.splineEdge;
const tessellateSpline = curves.tessellateSpline;
const arcs = @import("arc.zig");
const antialias_edge_offset = arcs.antialias_edge_offset;
const ArcRange = arcs.ArcRange;
const ringCoverage = arcs.ringCoverage;
const calculateArcCoverage = arcs.calculateArcCoverage;
const fillArcRing = arcs.fillArcRing;

/// Rendering quality mode for drawing operations
pub const DrawMode = enum {
    /// Fast rendering - hard edges, maximum performance
    fast,
    /// Soft rendering - antialiased edges, better quality
    soft,
};

/// Rendering style shared by all primitives: `mode` controls antialiasing only;
/// `blending` controls compositing and applies in both modes. The presets mirror
/// the classic `DrawMode` behavior: `.soft` composites, `.fast` overwrites.
pub const DrawOptions = struct {
    mode: DrawMode,
    blending: Blending,

    pub const soft: DrawOptions = .{ .mode = .soft, .blending = .normal };
    pub const fast: DrawOptions = .{ .mode = .fast, .blending = .none };
};

/// How the contours passed to `fillPolygons` combine.
pub const FillRule = enum {
    /// Inside where a ray from the point crosses an odd number of edges.
    even_odd,
    /// Inside where the contours wind around the point a nonzero net number of times;
    /// the rule glyph outlines are designed for.
    nonzero,
};

/// Coverage sink for `Canvas(u8)` masks: keeps the larger of the existing value and the
/// new coverage, so shapes accumulated over several calls never lighten each other.
const CoverageMax = struct {
    inline fn cover(_: CoverageMax, dest: *u8, alpha: f32) void {
        const coverage: u8 = @round(@min(alpha, 1) * 255);
        dest.* = @max(dest.*, coverage);
    }
};

/// How text glyphs are painted.
const GlyphStyle = union(enum) {
    fill,
    /// Stroke width in pixels.
    outline: f32,

    /// How far past the ink the paint reaches.
    fn reach(style: GlyphStyle) f32 {
        return switch (style) {
            .fill => 0,
            .outline => |width| width / 2,
        };
    }
};

/// A drawing context for an image, providing methods to draw shapes and lines.
pub fn Canvas(comptime T: type) type {
    return struct {
        image: Image(T),
        allocator: std.mem.Allocator,

        const Self = @This();

        // Drawing-related constants
        /// Vertical samples per pixel row for antialiased polygon fills
        const polygon_subscanlines = 8;
        /// Below this many edges the fast fill tests them all on every row; the sweep's
        /// setup only pays off above it.
        const few_edges = 64;
        /// Widest rectangle outline whose wall coverage profile is precomputed.
        const max_ring_profile = 64;
        /// Stack scratch (bytes) for the lines of a text block: 64 lines, longer texts spill
        /// to the heap.
        const lines_scratch_size = 64 * @sizeOf(text_layout.Lines.Line);
        /// Stack scratch (bytes) for polygon fills: edges, crossings and sweep order for 256
        /// vertices, 1024 sweep rows, a 1024-pixel coverage row and a 12k-cell area
        /// accumulator (a 110x110 shape); larger inputs spill to the heap.
        const polygon_scratch_size = 256 * (@sizeOf(Edge) + @sizeOf(Crossing) + 2 * @sizeOf(u32)) + 1024 * (@sizeOf(u32) + @sizeOf(CoverageCell)) + 12 * 1024 * (@sizeOf(f32) + 1);
        /// Stack scratch (bytes) for flattening a glyph outline; spills to heap beyond
        const glyph_scratch_size = 1024 * @sizeOf(Point(2, f32)) + 64 * @sizeOf([]const Point(2, f32));
        /// Stack scratch (points) for spline polygon tessellation; spills to heap beyond
        const spline_polygon_stack_buffer_size = 400;

        /// Creates a drawing canvas from an image.
        pub fn init(allocator: std.mem.Allocator, image: Image(T)) Self {
            return .{ .image = image, .allocator = allocator };
        }

        /// Clamps a floating-point coordinate to image bounds and converts to a u32 pixel index.
        inline fn clampToImageBounds(coord: f32, max_size: u32) u32 {
            return @trunc(clamp(coord, 0, as(f32, max_size)));
        }

        /// Clamps a rectangle to image bounds and returns integer pixel coordinates.
        /// Returns null if the rectangle is completely outside the image.
        inline fn clampRectToImage(self: Self, rect: Rectangle(f32)) ?Rectangle(u32) {
            const left = clampToImageBounds(rect.l, self.image.cols);
            const top = clampToImageBounds(rect.t, self.image.rows);
            const right = clampToImageBounds(rect.r, self.image.cols);
            const bottom = clampToImageBounds(rect.b, self.image.rows);

            if (left >= right or top >= bottom) {
                return null;
            }

            return .{ .l = left, .t = top, .r = right, .b = bottom };
        }

        /// Fills the entire canvas with a solid color.
        pub fn fill(self: Self, color: anytype) void {
            self.image.fill(convertColor(T, color));
        }

        /// Gets a reference to the pixel at the given coordinates.
        /// Panics if coordinates are out of bounds.
        pub inline fn at(self: Self, row: u32, col: u32) *T {
            return self.image.at(row, col);
        }

        /// Gets a reference to the pixel at the given coordinates, or null if out of bounds.
        pub inline fn atOrNull(self: Self, row: i32, col: i32) ?*T {
            return self.image.atOrNull(row, col);
        }

        /// Returns the number of rows (height) in the canvas.
        pub inline fn rows(self: Self) u32 {
            return self.image.rows;
        }

        /// Returns the number of columns (width) in the canvas.
        pub inline fn cols(self: Self) u32 {
            return self.image.cols;
        }

        /// Returns the total number of pixels in the canvas (rows * cols).
        pub inline fn size(self: Self) usize {
            return self.image.size();
        }

        /// Returns true if and only if this canvas and `other` have the same number of rows and columns.
        /// It does not compare pixel data or types.
        pub inline fn hasSameShape(self: Self, other: anytype) bool {
            return self.image.hasSameShape(other.image);
        }

        /// Creates a view (sub-canvas) of this canvas within the specified rectangle.
        /// The view shares memory with the parent canvas - changes are reflected in both.
        /// Coordinates are automatically clipped to the canvas bounds.
        pub fn view(self: Self, rect: Rectangle(u32)) Self {
            return .{
                .image = self.image.view(rect),
                .allocator = self.allocator,
            };
        }

        /// Clamps a horizontal span to the image and returns its pixel slice, or null when
        /// fully outside.
        fn spanSlice(self: Self, x1: f32, x2: f32, y: f32) ?[]T {
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);

            if (y < 0 or y >= frows) return null;
            if (x2 < 0 or x1 >= fcols) return null;

            const row: u32 = @trunc(y);
            const start: u32 = @floor(@max(0, x1));
            const end: u32 = @ceil(@min(fcols - 1, x2));

            if (start > end) return null;

            const offset = row * self.image.stride + start;
            return self.image.data[offset .. offset + end - start + 1];
        }

        /// Writes a horizontal span: memset for plain overwrites, per-pixel composite otherwise.
        fn fillSpan(self: Self, x1: f32, x2: f32, y: f32, paint: Paint) void {
            const span = self.spanSlice(x1, x2, y) orelse return;
            if (paint.overwrite) @memset(span, paint.solid) else for (span) |*px| paint.put(px);
        }

        /// A color prepared for writing: pre-converted to the canvas type and to Rgba, with the
        /// overwrite decision hoisted out of pixel loops.
        const Paint = struct {
            solid: T,
            rgba: Rgba,
            blending: Blending,
            overwrite: bool,
            /// Normal blending of an opaque color: over an opaque pixel it is `mixOpaque`.
            opaque_normal: bool,

            fn init(color: anytype, blending: Blending) Paint {
                comptime assert(isColor(@TypeOf(color)));
                const rgba = convertColor(Rgba, color);
                const opaque_normal = blending == .normal and rgba.a == 255;
                return .{
                    .solid = convertColor(T, color),
                    .rgba = rgba,
                    .blending = blending,
                    // `.normal` with an opaque color degenerates to the overlay.
                    .overwrite = blending == .none or opaque_normal,
                    .opaque_normal = opaque_normal,
                };
            }

            /// Writes the color at full coverage.
            inline fn put(p: Paint, dest: *T) void {
                if (p.overwrite) dest.* = p.solid else assignPixel(dest, p.rgba, p.blending);
            }

            /// Writes the color scaled by `alpha` coverage; full coverage takes the overwrite path.
            inline fn cover(p: Paint, dest: *T, alpha: f32) void {
                if (alpha >= 1) return p.put(dest);
                if (alpha <= 0) return;
                if (p.opaqueOver(dest)) return p.mixOpaque(dest, @trunc(255 * alpha));
                assignPixel(dest, p.rgba.fade(alpha), p.blending);
            }

            /// `cover` with 8-bit coverage, as stored in masks; the byte is used as is rather
            /// than round-tripped through a float.
            inline fn coverByte(p: Paint, dest: *T, value: u8) void {
                if (value == 255) return p.put(dest);
                if (value == 0) return;
                if (p.opaqueOver(dest)) return p.mixOpaque(dest, value);
                assignPixel(dest, p.rgba.fade(as(f32, value) / 255), p.blending);
            }

            /// Whether normal blending of this paint over `dest` reduces to `mixOpaque`.
            inline fn opaqueOver(p: Paint, dest: *const T) bool {
                if (comptime T != Rgb and T != Rgba) return false;
                return p.opaque_normal and (T == Rgb or dest.a == 255);
            }

            /// Normal blending of the opaque paint over an opaque pixel: `blendColors`'
            /// arithmetic, in its order, without the generic conversions around it.
            inline fn mixOpaque(p: Paint, dest: *T, a8: u8) void {
                const t: u32 = a8;
                const w: u32 = 255 - t;
                inline for (.{ "r", "g", "b" }) |channel| {
                    const mixed = @as(u32, @field(p.rgba, channel)) * t + @as(u32, @field(dest, channel)) * w;
                    @field(dest, channel) = @intCast((mixed + 127) / 255);
                }
            }
        };

        /// Draws a line between two points. `.fast` uses Bresenham (width 1) or a
        /// rectangle+caps; `.soft` antialiases via Wu (width 1) or distance-based rendering.
        pub fn drawLine(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype, width: u32, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            switch (opts.mode) {
                .fast => if (width == 1)
                    self.drawLineBresenham(p1, p2, color, opts.blending)
                else
                    self.drawLineRectangle(p1, p2, width, color, opts),
                .soft => if (width == 1)
                    self.drawLineXiaolinWu(p1, p2, color, opts.blending)
                else
                    self.drawLineDistance(p1, p2, width, color, opts),
            }
        }

        /// Bresenham's algorithm: hard-edged 1-pixel lines in integer arithmetic.
        fn drawLineBresenham(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype, blending: Blending) void {
            var x1: i32 = @trunc(p1.x());
            var y1: i32 = @trunc(p1.y());
            const x2: i32 = @trunc(p2.x());
            const y2: i32 = @trunc(p2.y());

            const paint: Paint = .init(color, blending);

            // Special case for horizontal lines - use fillSpan for better performance
            if (y1 == y2) {
                const min_x = @min(x1, x2);
                const max_x = @max(x1, x2);
                self.fillSpan(@floatFromInt(min_x), @floatFromInt(max_x), @floatFromInt(y1), paint);
                return;
            }

            // Special case for vertical lines - direct pixel access
            if (x1 == x2) {
                const min_y = @min(y1, y2);
                const max_y = @max(y1, y2);
                var y = min_y;
                while (y <= max_y) : (y += 1) {
                    if (self.atOrNull(y, x1)) |pixel| paint.put(pixel);
                }
                return;
            }

            // General case - standard Bresenham algorithm
            const dx: i32 = @intCast(@abs(x2 - x1));
            const dy: i32 = @intCast(@abs(y2 - y1));
            const sx: i32 = if (x1 < x2) 1 else -1;
            const sy: i32 = if (y1 < y2) 1 else -1;
            var err = dx - dy;

            while (true) {
                if (self.atOrNull(y1, x1)) |pixel| paint.put(pixel);

                if (x1 == x2 and y1 == y2) break;

                const e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x1 += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y1 += sy;
                }
            }
        }

        /// Xiaolin Wu's algorithm: antialiased 1-pixel lines, a pixel pair per step along the
        /// major axis split by fractional coverage.
        fn drawLineXiaolinWu(self: Self, p1: Point(2, f32), p2: Point(2, f32), color: anytype, blending: Blending) void {
            const paint: Paint = .init(color, blending);

            var x1 = p1.x();
            var y1 = p1.y();
            var x2 = p2.x();
            var y2 = p2.y();

            // Special case for perfectly horizontal lines
            if (@abs(y2 - y1) < 0.01) {
                const y = @round(y1);
                const min_x = @min(x1, x2);
                const max_x = @max(x1, x2);

                // Handle fractional endpoints with antialiasing
                const left_x = @floor(min_x);
                const right_x = @ceil(max_x);

                // Left endpoint antialiasing
                if (min_x > left_x) self.wuPlot(false, left_x, y, min_x - left_x, paint);

                // Middle solid part - use fillSpan for performance
                const solid_start = @ceil(min_x);
                const solid_end = @floor(max_x);
                if (solid_end >= solid_start) {
                    self.fillSpan(solid_start, solid_end, y, paint);
                }

                // Right endpoint antialiasing
                if (max_x < right_x) self.wuPlot(false, right_x, y, right_x - max_x, paint);

                return;
            }

            const steep = @abs(y2 - y1) > @abs(x2 - x1);
            if (steep) {
                std.mem.swap(f32, &x1, &y1);
                std.mem.swap(f32, &x2, &y2);
            }
            if (x1 > x2) {
                std.mem.swap(f32, &x1, &x2);
                std.mem.swap(f32, &y1, &y2);
            }

            const dx = x2 - x1;
            const dy = y2 - y1;
            const gradient = if (dx == 0) 1.0 else dy / dx;

            // Endpoints: the pixel pair straddling the line, weighted by how much of the
            // endpoint's column the line covers.
            const x_px1 = @round(x1);
            const y_end1 = y1 + gradient * (x_px1 - x1);
            self.wuPair(steep, x_px1, y_end1, rfpart(x1 + 0.5), paint);
            const x_px2 = @round(x2);
            const y_end2 = y2 + gradient * (x_px2 - x2);
            self.wuPair(steep, x_px2, y_end2, fpart(x2 + 0.5), paint);

            var intery = y_end1 + gradient;
            var x = x_px1 + 1;
            while (x < x_px2) : (x += 1) {
                self.wuPair(steep, x, intery, 1, paint);
                intery += gradient;
            }
        }

        /// The two pixels of a Wu line at major-axis position `x`, split by the fractional
        /// part of `y`, both weighted by `gap`.
        inline fn wuPair(self: Self, steep: bool, x: f32, y: f32, gap: f32, paint: Paint) void {
            const y_px = @floor(y);
            self.wuPlot(steep, x, y_px, rfpart(y) * gap, paint);
            self.wuPlot(steep, x, y_px + 1, fpart(y) * gap, paint);
        }

        /// One Wu-line pixel at `alpha` coverage; a steep line has its axes swapped.
        inline fn wuPlot(self: Self, steep: bool, x: f32, y: f32, alpha: f32, paint: Paint) void {
            const px = if (steep) self.atOrNull(@floor(x), @floor(y)) else self.atOrNull(@floor(y), @floor(x));
            if (px) |dest| paint.cover(dest, alpha);
        }

        /// Thick `.fast` line: a filled rectangle along the line plus round caps (a single
        /// filled circle for a zero-length line).
        fn drawLineRectangle(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: u32, color: anytype, opts: DrawOptions) void {
            // For thick lines, draw as a filled rectangle
            const dx = p2.x() - p1.x();
            const dy = p2.y() - p1.y();
            const line_length = @sqrt(dx * dx + dy * dy);
            const half_width: f32 = as(f32, width) / 2.0;

            if (line_length == 0) {
                // Single point - draw a filled circle
                self.fillCircle(p1, half_width, color, opts);
                return;
            }

            // Calculate perpendicular vector for thick line
            const perp_x = -dy / line_length * half_width;
            const perp_y = dx / line_length * half_width;

            // Create rectangle corners
            const corners = [_]Point(2, f32){
                .init(.{ p1.x() - perp_x, p1.y() - perp_y }),
                .init(.{ p1.x() + perp_x, p1.y() + perp_y }),
                .init(.{ p2.x() + perp_x, p2.y() + perp_y }),
                .init(.{ p2.x() - perp_x, p2.y() - perp_y }),
            };

            // Fill rectangle using scanline algorithm (no anti-aliasing)
            self.fillPolygon(&corners, color, opts) catch return;

            // Add rounded caps using solid circles
            self.fillCircle(p1, half_width, color, opts);
            self.fillCircle(p2, half_width, color, opts);
        }

        /// Distance-based antialiased rendering for thick lines.
        /// Calculates the perpendicular distance from each pixel to the line segment and
        /// applies smooth alpha falloff at edges. End caps fall out naturally from the
        /// distance test.
        fn drawLineDistance(self: Self, p1: Point(2, f32), p2: Point(2, f32), width: u32, color: anytype, opts: DrawOptions) void {
            const half_width: f32 = as(f32, width) / 2.0;

            const dx = p2.x() - p1.x();
            const dy = p2.y() - p1.y();
            const length_sq = dx * dx + dy * dy;
            if (length_sq == 0) {
                self.fillCircle(p1, half_width, color, opts);
                return;
            }

            // Axis-aligned fast path: per-row (horizontal) or per-col (vertical) alpha is
            // uniform, so we can skip the per-pixel sqrt and use `fillSpan` (memset)
            // for fully-covered interior rows of horizontal lines. End-cap AA still comes from
            // `fillCircle`. Body bbox is tight to the line endpoints so caps don't double-blend.
            if (dx == 0 or dy == 0) {
                self.drawAxisAlignedThickLine(p1, p2, half_width, color, opts);
                return;
            }

            const paint: Paint = .init(color, opts.blending);

            const inv_length_sq = 1.0 / length_sq;
            const line_rect: Rectangle(f32) = .{
                .l = @min(p1.x(), p2.x()) - half_width,
                .t = @min(p1.y(), p2.y()) - half_width,
                .r = @max(p1.x(), p2.x()) + half_width + 1,
                .b = @max(p1.y(), p2.y()) + half_width + 1,
            };
            const bbox = self.clampRectToImage(line_rect) orelse return;

            // Per row, only the pixels within reach: the band around the line (its horizontal
            // cross-section widens as the line flattens) plus the two end caps.
            const reach = half_width + antialias_edge_offset;
            const band_half = reach * @sqrt(length_sq) / @abs(dy);
            const seg_lo = @min(p1.x(), p2.x()) - reach;
            const seg_hi = @max(p1.x(), p2.x()) + reach;
            const col_min = as(f32, bbox.l);
            const col_max = as(f32, bbox.r - 1);
            for (bbox.t..bbox.b) |r| {
                const py = as(f32, r);
                const dpy = py - p1.y();
                const x_on_line = p1.x() + dpy * dx / dy;
                var lo = @max(x_on_line - band_half, seg_lo);
                var hi = @min(x_on_line + band_half, seg_hi);
                inline for (.{ p1, p2 }) |cap| {
                    const dcy = py - cap.y();
                    if (@abs(dcy) <= reach) {
                        const half = @sqrt(reach * reach - dcy * dcy);
                        lo = @min(lo, cap.x() - half);
                        hi = @max(hi, cap.x() + half);
                    }
                }
                lo = @max(lo, col_min);
                hi = @min(hi, col_max);
                if (hi < lo) continue;
                const col_lo: u32 = @ceil(lo);
                const col_hi: u32 = @as(u32, @floor(hi)) + 1;
                for (col_lo..col_hi) |c| {
                    const px = as(f32, c);
                    const dpx = px - p1.x();
                    const t = clamp((dpx * dx + dpy * dy) * inv_length_sq, 0, 1);
                    const dist_x = dpx - t * dx;
                    const dist_y = dpy - t * dy;
                    const dist = @sqrt(dist_x * dist_x + dist_y * dist_y);
                    if (dist > half_width + antialias_edge_offset) continue;
                    var alpha: f32 = 1.0;
                    if (dist > half_width - antialias_edge_offset) {
                        alpha = half_width + antialias_edge_offset - dist;
                    }
                    paint.cover(&self.image.data[r * self.image.stride + c], alpha);
                }
            }
        }

        /// Horizontal and vertical thick lines: the body is an axis-aligned box whose ends fall
        /// on pixel boundaries (the round caps cover the rest), so only its long edges are
        /// antialiased. The caps are drawn separately to avoid double-blending them.
        fn drawAxisAlignedThickLine(self: Self, p1: Point(2, f32), p2: Point(2, f32), half_width: f32, color: anytype, opts: DrawOptions) void {
            const paint: Paint = .init(color, opts.blending);
            const body: Rectangle(f32) = if (p1.y() == p2.y()) .{
                .l = @floor(@min(p1.x(), p2.x())) - 0.5,
                .t = p1.y() - half_width,
                .r = @floor(@max(p1.x(), p2.x())) + 0.5,
                .b = p1.y() + half_width,
            } else .{
                .l = p1.x() - half_width,
                .t = @floor(@min(p1.y(), p2.y())) - 0.5,
                .r = p1.x() + half_width,
                .b = @floor(@max(p1.y(), p2.y())) + 0.5,
            };
            self.fillBoxSoft(body, paint);
            self.fillCircle(p1, half_width, color, opts);
            self.fillCircle(p2, half_width, color, opts);
        }

        /// Writes a pixel at integer (row, col), compositing `color` with `blending`
        /// (via an Rgba round-trip on non-Rgba canvases). Out-of-bounds is silently ignored.
        pub inline fn setPixel(self: Self, row: u32, col: u32, color: anytype, blending: Blending) void {
            if (self.image.atOrNull(row, col)) |dest| Paint.init(color, blending).put(dest);
        }

        /// Floors `point` to an integer pixel cell and writes via `setPixel`. Bridges
        /// float-coordinate drawing primitives to the integer pixel grid.
        pub fn setPoint(self: Self, point: Point(2, f32), color: anytype, blending: Blending) void {
            if (self.image.atOrNull(@floor(point.y()), @floor(point.x()))) |dest| Paint.init(color, blending).put(dest);
        }

        /// Draws another image onto this canvas at the given top-left position.
        /// Supports alpha blending for RGBA images with the normal blend mode.
        /// For rotation, scaling, or custom blend modes, users should access the canvas's image field directly.
        pub fn drawImage(self: Self, source: anytype, position: Point(2, f32), source_rect_opt: ?Rectangle(u32), blend_mode: Blending) void {
            const full_rect = source.getRectangle();
            const src_rect = full_rect.intersect(source_rect_opt orelse full_rect) orelse return;

            const origin_x: i32 = @round(position.x());
            const origin_y: i32 = @round(position.y());

            // `assignPixel` only blends Rgba sources; the rest are copied whatever the mode.
            for (src_rect.t..src_rect.b) |src_r| {
                const dest_y = origin_y + @as(i32, @intCast(src_r - src_rect.t));
                for (src_rect.l..src_rect.r) |src_c| {
                    const dest_x = origin_x + @as(i32, @intCast(src_c - src_rect.l));
                    if (self.atOrNull(dest_y, dest_x)) |dest_pixel| assignPixel(dest_pixel, source.at(src_r, src_c).*, blend_mode);
                }
            }
        }

        /// Returns the fractional part of a floating-point number.
        /// Used in Wu's anti-aliasing algorithm to calculate pixel coverage.
        /// Example: fpart(3.7) = 0.7, fpart(-2.3) = 0.7
        fn fpart(x: f32) f32 {
            return x - @floor(x);
        }

        /// Returns the reverse fractional part (1 - fractional part).
        /// Used in Wu's anti-aliasing algorithm for complementary pixel coverage.
        /// Example: rfpart(3.7) = 0.3, rfpart(-2.3) = 0.3
        fn rfpart(x: f32) f32 {
            return 1 - fpart(x);
        }

        /// Draws the outline of a polygon: its edges in order, closed from the last vertex back
        /// to the first. Widths above 1 get round joins; width 1 draws pixel lines.
        pub fn drawPolygon(self: Self, polygon: []const Point(2, f32), color: anytype, width: u32, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len == 0) return;
            self.strokePath(&.{polygon}, true, width, color, opts);
        }

        /// Draws the outline of a rectangle, `width` pixels centered on its edges (the last
        /// covered row and column, since `r` and `b` are exclusive), with square corners.
        /// Width 1 draws pixel lines; wider outlines are rasterized as the ring between two
        /// axis-aligned rectangles, exact and in one pass.
        pub fn drawRectangle(self: Self, rect: Rectangle(f32), color: anytype, width: u32, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;
            const l = rect.l;
            const t = rect.t;
            const r = rect.r - 1;
            const b = rect.b - 1;
            if (r < l or b < t) return;
            if (width == 1) {
                const points: []const Point(2, f32) = &.{ .init(.{ l, t }), .init(.{ r, t }), .init(.{ r, b }), .init(.{ l, b }) };
                return self.drawPolygon(points, color, width, opts);
            }
            const half = as(f32, width) / 2;
            const edges: Rectangle(f32) = .{ .l = l, .t = t, .r = r, .b = b };
            const outer = edges.grow(half);
            const inner = edges.shrink(half);
            const paint: Paint = .init(color, opts.blending);
            // Widths past the rectangle's own size leave nothing to cut out.
            const hole: ?Rectangle(f32) = if (inner.isEmpty()) null else inner;
            switch (opts.mode) {
                .fast => self.fillRingFast(outer, hole, paint),
                .soft => if (hole) |h| self.fillRingSoft(outer, h, paint) else self.fillBoxSoft(outer, paint),
            }
        }

        /// Length of the overlap between the unit pixel span centered on `c` and `[a, b]`.
        inline fn pixelOverlap(a: f32, b: f32, c: f32) f32 {
            return @max(0, @min(b, c + 0.5) - @max(a, c - 0.5));
        }

        /// Hard-edged axis-aligned box, less the non-empty `inner` when given: the pixels
        /// whose centers lie in the outer rectangle but not the inner one.
        fn fillRingFast(self: Self, outer: Rectangle(f32), inner: ?Rectangle(f32), paint: Paint) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            var y = @max(@ceil(outer.t), 0);
            const y_end = @min(@floor(outer.b), frows - 1);
            const hole = inner orelse {
                while (y <= y_end) : (y += 1) self.fillSpan(@ceil(outer.l), @floor(outer.r), y, paint);
                return;
            };
            while (y <= y_end) : (y += 1) {
                if (y >= hole.t and y <= hole.b) {
                    self.fillSpan(@ceil(outer.l), @ceil(hole.l) - 1, y, paint);
                    self.fillSpan(@floor(hole.r) + 1, @floor(outer.r), y, paint);
                } else {
                    self.fillSpan(@ceil(outer.l), @floor(outer.r), y, paint);
                }
            }
        }

        /// Antialiased axis-aligned box: a pixel's coverage is the product of its overlaps
        /// with the box's row and column spans, so fractional edges are exact. A wide box
        /// spans the core of each row; a narrow one (a vertical line, a wall) walks each
        /// column instead, its coverage fixed down the column except on the fringe rows.
        fn fillBoxSoft(self: Self, box: Rectangle(f32), paint: Paint) void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const y_start = @max(@ceil(box.t - 0.5), 0);
            const y_end = @min(@floor(box.b + 0.5), frows - 1);
            if (y_end < y_start) return;
            var buf: [max_ring_profile]f32 = undefined;
            const columns = self.ringProfile(&buf, box.l - 0.5, box.r + 0.5, box, null) orelse {
                var y = y_start;
                while (y <= y_end) : (y += 1) {
                    const vy = pixelOverlap(box.t, box.b, y);
                    if (vy > 0) self.boxRow(box, y, vy, paint);
                }
                return;
            };
            // Rows fully inside the box, between the fringe rows.
            const core_t = @max(@ceil(box.t + 0.5), y_start);
            const core_b = @min(@floor(box.b - 0.5), y_end);
            const first: usize = @trunc(y_start);
            const end: usize = @as(usize, @trunc(y_end)) + 1;
            const core_first: usize = @trunc(core_t);
            const core_end: usize = if (core_b >= core_t) @as(usize, @trunc(core_b)) + 1 else core_first;
            for (columns.coverage, columns.first..) |cx, c| {
                if (cx <= 0) continue;
                for (first..core_first) |r| self.coverCell(r, c, pixelOverlap(box.t, box.b, as(f32, r)) * cx, paint);
                for (core_first..core_end) |r| paint.cover(&self.image.data[r * self.image.stride + c], cx);
                for (core_end..end) |r| self.coverCell(r, c, pixelOverlap(box.t, box.b, as(f32, r)) * cx, paint);
            }
        }

        inline fn coverCell(self: Self, row: usize, col: usize, coverage: f32, paint: Paint) void {
            if (coverage > 0) paint.cover(&self.image.data[row * self.image.stride + col], coverage);
        }

        /// One row of an antialiased box, `vy` being the row's vertical overlap with it: a
        /// row fully inside paints its uniform core with a span and only the fringe by formula.
        inline fn boxRow(self: Self, box: Rectangle(f32), y: f32, vy: f32, paint: Paint) void {
            const row: u32 = @trunc(y);
            const row_px = self.image.data[row * self.image.stride ..][0..self.image.cols];
            if (vy >= 1) {
                const core_lo = @ceil(box.l + 0.5);
                const core_hi = @floor(box.r - 0.5);
                ringCells(row_px, box.l - 0.5, core_lo - 1, vy, 0, box, null, paint);
                self.fillSpan(core_lo, core_hi, y, paint);
                ringCells(row_px, core_hi + 1, box.r + 0.5, vy, 0, box, null, paint);
            } else {
                ringCells(row_px, box.l - 0.5, box.r + 0.5, vy, 0, box, null, paint);
            }
        }

        /// Antialiased ring between `outer` and a non-empty `inner`: a pixel's coverage is its
        /// overlap with the outer rectangle less its overlap with the inner one, each the
        /// product of two 1-D overlaps, so the ring is exact without seams. Rows outside the
        /// inner rectangle are box rows.
        fn fillRingSoft(self: Self, outer: Rectangle(f32), inner: Rectangle(f32), paint: Paint) void {
            assert(!inner.isEmpty());
            const frows: f32 = @floatFromInt(self.image.rows);
            // Down a wall the horizontal coverage profile repeats, so it is computed once.
            var left_buf: [max_ring_profile]f32 = undefined;
            var right_buf: [max_ring_profile]f32 = undefined;
            const left = self.ringProfile(&left_buf, outer.l - 0.5, inner.l + 0.5, outer, inner);
            const right = self.ringProfile(&right_buf, inner.r - 0.5, outer.r + 0.5, outer, inner);
            var y = @max(@ceil(outer.t - 0.5), 0);
            const y_end = @min(@floor(outer.b + 0.5), frows - 1);
            while (y <= y_end) : (y += 1) {
                const vy_o = pixelOverlap(outer.t, outer.b, y);
                if (vy_o <= 0) continue;
                const vy_i = pixelOverlap(inner.t, inner.b, y);
                if (vy_i <= 0) {
                    self.boxRow(outer, y, vy_o, paint);
                    continue;
                }
                const row: u32 = @trunc(y);
                const row_px = self.image.data[row * self.image.stride ..][0..self.image.cols];
                if (vy_i < 1) {
                    // The inner rectangle's own edge row: every cell by formula.
                    ringCells(row_px, outer.l - 0.5, outer.r + 0.5, vy_o, vy_i, outer, inner, paint);
                } else if (left != null and right != null) {
                    // A full wall row: paint the profiles.
                    inline for (.{ left.?, right.? }) |wall| {
                        for (wall.coverage, row_px[wall.first..][0..wall.coverage.len]) |coverage, *px| {
                            if (coverage > 0) paint.cover(px, coverage);
                        }
                    }
                } else {
                    ringCells(row_px, outer.l - 0.5, inner.l + 0.5, vy_o, vy_i, outer, inner, paint);
                    ringCells(row_px, inner.r - 0.5, outer.r + 0.5, vy_o, vy_i, outer, inner, paint);
                }
            }
        }

        const RingWall = struct { first: usize, coverage: []const f32 };

        /// The horizontal coverage of the cells whose centers lie in `[lo, hi]` for a row
        /// inside `outer` (and `inner`, when there is one), written to `buf`; null when the
        /// range is empty or wider than `buf`.
        fn ringProfile(self: Self, buf: []f32, lo: f32, hi: f32, outer: Rectangle(f32), inner: ?Rectangle(f32)) ?RingWall {
            const fcols: f32 = @floatFromInt(self.image.cols);
            const c0 = @max(@ceil(lo), 0);
            const c1 = @min(@floor(hi), fcols - 1);
            if (c1 < c0) return null;
            const len: usize = @trunc(c1 - c0 + 1);
            if (len > buf.len) return null;
            for (buf[0..len], 0..) |*coverage, i| {
                const c = c0 + as(f32, i);
                coverage.* = pixelOverlap(outer.l, outer.r, c);
                if (inner) |hole| coverage.* -= pixelOverlap(hole.l, hole.r, c);
            }
            return .{ .first = @trunc(c0), .coverage = buf[0..len] };
        }

        /// Coverage-paints the cells of `row_px` whose centers lie in `[lo, hi]` for a row with
        /// vertical overlap `vy_o` with `outer` and `vy_i` with `inner`, when there is one.
        /// Inlined so a call site's missing `inner` costs nothing per cell.
        inline fn ringCells(row_px: []T, lo: f32, hi: f32, vy_o: f32, vy_i: f32, outer: Rectangle(f32), inner: ?Rectangle(f32), paint: Paint) void {
            const fcols: f32 = @floatFromInt(row_px.len);
            var c = @max(@ceil(lo), 0);
            const c_end = @min(@floor(hi), fcols - 1);
            while (c <= c_end) : (c += 1) {
                var coverage = vy_o * pixelOverlap(outer.l, outer.r, c);
                if (inner) |hole| coverage -= vy_i * pixelOverlap(hole.l, hole.r, c);
                if (coverage > 0) paint.cover(&row_px[@trunc(c)], coverage);
            }
        }

        /// Fills a rectangle on the given image.
        /// The rectangle is defined using standard conventions where l,t are inclusive and r,b are exclusive.
        /// This means a rectangle from (0,0) to (10,10) will fill pixels at positions 0-9 in both dimensions.
        /// Fractional edges are truncated to whole pixels, so `opts.mode` has no effect here.
        pub fn fillRectangle(self: Self, rect: Rectangle(f32), color: anytype, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));

            const bounds = self.clampRectToImage(rect) orelse return;
            const paint: Paint = .init(color, opts.blending);

            for (bounds.t..bounds.b) |row| {
                self.fillSpan(as(f32, bounds.l), as(f32, bounds.r - 1), as(f32, row), paint);
            }
        }

        /// Draws the outline of a circle on the given image.
        /// Use DrawMode.soft for anti-aliased edges or DrawMode.fast for fast aliased edges.
        pub fn drawCircle(self: Self, center: Point(2, f32), radius: f32, color: anytype, width: u32, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0 or width == 0) return;
            self.strokeRing(center, radius, .full, width, color, opts);
        }

        /// Draws an arc outline. Angles are in radians from the positive X-axis, counter-clockwise,
        /// and may exceed 2π (auto-normalized). Arcs spanning ≥ 2π render as a full circle.
        ///
        /// Example:
        /// ```zig
        /// // Red quarter arc from 0 to π/2
        /// try canvas.drawArc(center, 50, 0, std.math.pi / 2.0, Rgb.red, 2, .soft);
        /// ```
        pub fn drawArc(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype, width: u32, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0 or width == 0) return;
            const arc = ArcRange.fromAngles(start_angle, end_angle) orelse return;
            if (opts.mode == .soft and !arc.isFull()) return self.drawArcSoft(center, radius, arc, width, color, opts);
            self.strokeRing(center, radius, arc, width, color, opts);
        }

        /// A ring `width` pixels wide centered on `radius`, restricted to `arc`; a pixel-wide
        /// `.fast` ring is a Bresenham circle.
        inline fn strokeRing(self: Self, center: Point(2, f32), radius: f32, arc: ArcRange, width: u32, color: anytype, opts: DrawOptions) void {
            if (opts.mode == .fast and width == 1) return self.drawBresenhamCircle(center, radius, color, arc, opts.blending);
            const half = as(f32, width) / 2;
            self.renderRing(center, radius - half, radius + half, color, arc, opts);
        }

        /// Screen-space bounding box of the outer circle (padded by one pixel for the AA ramp),
        /// clamped to image bounds. Returns null when the box has zero area.
        inline fn ringBoundingBox(self: Self, center: Point(2, f32), outer_radius: f32) ?Rectangle(u32) {
            const pad = outer_radius + 1;
            return self.clampRectToImage(.{
                .l = @floor(center.x() - pad),
                .t = @floor(center.y() - pad),
                .r = @ceil(center.x() + pad),
                .b = @ceil(center.y() + pad),
            });
        }

        /// Renders a thick ring outline (or arc segment); `.fast` uses binary coverage. Each
        /// row visits only the columns within reach of the outer edge and outside the inner
        /// one (a pixel of margin either side), then tests every pixel as before.
        inline fn renderRing(self: Self, center: Point(2, f32), inner_radius: f32, outer_radius: f32, color: anytype, arc: ArcRange, opts: DrawOptions) void {
            const paint: Paint = .init(color, opts.blending);
            switch (opts.mode) {
                inline else => |mode| self.renderRingMode(center, inner_radius, outer_radius, paint, arc, mode),
            }
        }

        /// `renderRing` with the mode resolved at compile time. Inlined so a caller's constant
        /// radius or arc folds into the pixel loop.
        inline fn renderRingMode(self: Self, center: Point(2, f32), inner_radius: f32, outer_radius: f32, paint: Paint, arc: ArcRange, comptime mode: DrawMode) void {
            const bbox = self.ringBoundingBox(center, outer_radius) orelse return;
            const edge: f32 = if (mode == .soft) antialias_edge_offset else 0;
            const full = arc.isFull();
            const reach_outer = outer_radius + edge;
            const reach_inner = if (inner_radius > 0) inner_radius - edge else 0;
            const col_min = as(f32, bbox.l);
            const col_max = as(f32, bbox.r - 1);
            for (bbox.t..bbox.b) |r| {
                const y = as(f32, r) - center.y();
                const half_outer_sq = reach_outer * reach_outer - y * y;
                if (half_outer_sq < 0) continue;
                const half_outer = @sqrt(half_outer_sq) + 1;
                const half_inner_sq = reach_inner * reach_inner - y * y;
                const half_inner = if (reach_inner > 0 and half_inner_sq > 0) @max(@sqrt(half_inner_sq) - 1, 0) else 0;
                // Two runs around the inner gap, or one across when the gap closes.
                const runs = [2][2]f32{
                    .{ center.x() - half_outer, if (half_inner > 0) center.x() - half_inner else center.x() + half_outer },
                    .{ center.x() + half_inner, center.x() + half_outer },
                };
                for (runs, 0..) |run, i| {
                    if (i == 1 and half_inner == 0) break;
                    const lo = @max(@ceil(run[0]), col_min);
                    const hi = @min(@floor(run[1]), col_max);
                    if (hi < lo) continue;
                    const c_lo: usize = @trunc(lo);
                    const c_hi: usize = @trunc(hi);
                    for (c_lo..c_hi + 1) |c| {
                        const x = as(f32, c) - center.x();
                        const coverage = ringCoverage(x, y, inner_radius, outer_radius, mode);
                        if (coverage <= 0) continue;
                        if (!full and !arc.contains(std.math.atan2(y, x))) continue;
                        paint.cover(&self.image.data[r * self.image.stride + c], coverage);
                    }
                }
            }
        }

        /// Rasterizes a 1-pixel-thick Bresenham circle around `center`.
        /// Each of the 8 octant-symmetric pixels is gated by an angle check.
        fn drawBresenhamCircle(self: Self, center: Point(2, f32), radius: f32, color: anytype, arc: ArcRange, blending: Blending) void {
            const paint: Paint = .init(color, blending);
            const cx: i32 = @round(center.x());
            const cy: i32 = @round(center.y());
            const r = @round(radius);
            var x: f32 = r;
            var y: f32 = 0;
            var err: f32 = 0;
            const full = arc.isFull();
            while (x >= y) {
                const offsets = [_][2]f32{
                    .{ x, y }, .{ -x, y }, .{ x, -y }, .{ -x, -y },
                    .{ y, x }, .{ -y, x }, .{ y, -x }, .{ -y, -x },
                };
                for (offsets) |o| {
                    if (!full and !arc.contains(std.math.atan2(o[1], o[0]))) continue;
                    const col: i32 = @trunc(o[0]);
                    const row: i32 = @trunc(o[1]);
                    if (self.atOrNull(cy + row, cx + col)) |dest| paint.put(dest);
                }
                if (err <= 0) {
                    y += 1;
                    err += 2 * y + 1;
                }
                if (err > 0) {
                    x -= 1;
                    err -= 2 * x + 1;
                }
            }
        }

        /// Antialiased partial arc outline: a polyline along the arc for width 1, otherwise
        /// the polygon between the outer and inner arcs.
        fn drawArcSoft(self: Self, center: Point(2, f32), radius: f32, arc: ArcRange, width: u32, color: anytype, opts: DrawOptions) !void {
            const segments: u32 = @max(8, @min(bezier_max_segments_count, @as(u32, @ceil(arc.length(radius) / 5.0))));
            const angle_step = arc.span() / as(f32, segments);
            var points: [2 * (bezier_max_segments_count + 1)]Point(2, f32) = undefined;
            const outer = points[0 .. segments + 1];
            if (width == 1) {
                // The arc itself.
                fillArcRing(outer, center, radius, arc.start, angle_step);
                return self.strokePath(&.{outer}, false, 1, color, opts);
            }
            const half = as(f32, width) / 2;
            fillArcRing(outer, center, radius + half, arc.start, angle_step);
            fillArcRing(points[segments + 1 .. 2 * (segments + 1)], center, radius - half, arc.end, -angle_step);
            try self.fillPolygon(points[0 .. 2 * (segments + 1)], color, opts);
        }

        /// Fills the given polygon on an image using an even-odd scanline algorithm.
        /// The polygon is defined by an array of points (vertices).
        ///
        /// **Rendering Modes:**
        /// - **DrawMode.fast**: hard edges, one sample per row.
        /// - **DrawMode.soft**: antialiased edges in every direction — each row is sampled at
        ///   `polygon_subscanlines` heights with exact horizontal coverage.
        pub fn fillPolygon(self: Self, polygon: []const Point(2, f32), color: anytype, opts: DrawOptions) !void {
            return self.fillPolygons(&.{polygon}, color, .even_odd, opts);
        }

        /// Fills several closed contours as one shape under `fill_rule`. Glyph outlines need
        /// `.nonzero`: their holes are reverse-winding contours and their strokes may overlap.
        pub fn fillPolygons(self: Self, contours: []const []const Point(2, f32), color: anytype, fill_rule: FillRule, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            const paint: Paint = .init(color, opts.blending);
            switch (opts.mode) {
                inline else => |mode| return self.fillContours(contours, fill_rule, mode, paint),
            }
        }

        /// Accumulates antialiased coverage of `contours` into an 8-bit mask:
        /// `dest = max(dest, 255 * coverage)`. Pixels outside the shape are never written, so
        /// clear the mask first. Only for `Canvas(u8)`.
        pub fn rasterizePolygons(self: Self, contours: []const []const Point(2, f32), fill_rule: FillRule) !void {
            comptime assert(T == u8);
            return self.fillContours(contours, fill_rule, .soft, CoverageMax{});
        }

        /// Fills a glyph outline with the nonzero rule; `transform` maps font units to pixels.
        pub fn fillGlyph(self: Self, outline: Outline, transform: Outline.Transform, color: anytype, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            const paint: Paint = .init(color, opts.blending);
            switch (opts.mode) {
                inline else => |mode| return self.renderOutline(outline, transform, .fill, mode, paint),
            }
        }

        /// Antialiased coverage of a glyph outline into an 8-bit mask, as `rasterizePolygons`.
        pub fn rasterizeGlyph(self: Self, outline: Outline, transform: Outline.Transform) !void {
            comptime assert(T == u8);
            return self.renderOutline(outline, transform, .fill, .soft, CoverageMax{});
        }

        /// A glyph outline flattened into device-space polygons, in scratch memory.
        const FlatGlyph = struct {
            points: []Point(2, f32),
            contours: [][]const Point(2, f32),
            polys: [][]const Point(2, f32),

            fn init(scratch: std.mem.Allocator, outline: Outline, transform: Outline.Transform) !FlatGlyph {
                const points = try scratch.alloc(Point(2, f32), outline.flattenedPointCount(transform));
                errdefer scratch.free(points);
                const contours = try scratch.alloc([]const Point(2, f32), outline.contourCount());
                return .{ .points = points, .contours = contours, .polys = outline.flatten(transform, points, contours) };
            }

            fn deinit(self: FlatGlyph, scratch: std.mem.Allocator) void {
                scratch.free(self.contours);
                scratch.free(self.points);
            }
        };

        /// Flattens `outline` in scratch memory, then fills it (nonzero) or strokes it as
        /// `style` says.
        fn renderOutline(self: Self, outline: Outline, transform: Outline.Transform, style: GlyphStyle, comptime mode: DrawMode, sink: anytype) !void {
            var stack: [glyph_scratch_size]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const scratch = buffer_first.allocator();
            const flat = try FlatGlyph.init(scratch, outline, transform);
            defer flat.deinit(scratch);
            return switch (style) {
                .fill => self.fillContours(flat.polys, .nonzero, mode, sink),
                .outline => |width| self.strokePolylines(scratch, flat.polys, true, width, mode, sink),
            };
        }

        /// Strokes polylines `width` pixels wide with round joins and caps. Each polyline
        /// becomes one outline: its left offsets forward, a round end cap, its right offsets
        /// backward and a round start cap; outer joins are arcs, inner joins detour through
        /// the vertex so the overlap winds like the rest and a single nonzero pass fills it.
        /// Closed polylines repeat their first point, which turns the two caps into a join.
        fn strokePolylines(self: Self, scratch: std.mem.Allocator, polys: []const []const Point(2, f32), closed: bool, width: f32, comptime mode: DrawMode, sink: anytype) !void {
            const radius = @max(width, 0.5) / 2;
            // Chords of an arc stay within the flatness tolerance at this angular step.
            const step: f32 = if (radius <= Outline.flatness_tolerance) std.math.pi else 2 * std.math.acos(1 - Outline.flatness_tolerance / radius);
            const arc_points: usize = @as(usize, @ceil(std.math.pi / step)) + 1;

            var total: usize = 0;
            for (polys) |poly| total += (poly.len + 1) * 2 * (arc_points + 3) + 2 * arc_points + 4;
            const points = try scratch.alloc(Point(2, f32), total);
            defer scratch.free(points);
            const contours = try scratch.alloc([]const Point(2, f32), polys.len);
            defer scratch.free(contours);

            var n: usize = 0;
            var k: usize = 0;
            for (polys) |poly| {
                if (poly.len == 0) continue;
                var builder: StrokeBuilder = .{ .out = points[n..], .radius = radius, .step = step };
                builder.polyline(poly, closed);
                const contour = points[n..][0..builder.len];
                // Every stroke winds the same way, so overlapping strokes add up instead of cancelling.
                if (signedArea(contour) < 0) std.mem.reverse(Point(2, f32), contour);
                contours[k] = contour;
                k += 1;
                n += builder.len;
            }
            return self.fillContours(contours[0..k], .nonzero, mode, sink);
        }

        /// Outlines for the shape API: width 1 draws each polyline's segments as pixel lines,
        /// wider ones go through `strokePolylines`.
        fn strokePath(self: Self, polys: []const []const Point(2, f32), closed: bool, width: u32, color: anytype, opts: DrawOptions) void {
            if (width == 0) return;
            if (width == 1) {
                for (polys) |points| {
                    if (points.len == 0) continue;
                    for (1..points.len) |i| self.drawLine(points[i - 1], points[i], color, 1, opts);
                    if (closed) self.drawLine(points[points.len - 1], points[0], color, 1, opts);
                }
                return;
            }
            var stack: [glyph_scratch_size]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const paint: Paint = .init(color, opts.blending);
            switch (opts.mode) {
                inline else => |mode| self.strokePolylines(buffer_first.allocator(), polys, closed, as(f32, width), mode, paint) catch return,
            }
        }

        /// Scanline fill shared by every polygon entry point. `sink` is a `Paint` or, for
        /// `.soft` masks, a `CoverageMax`.
        fn fillContours(self: Self, contours: []const []const Point(2, f32), fill_rule: FillRule, comptime mode: DrawMode, sink: anytype) !void {
            var vertices: usize = 0;
            var bounds: ?Rectangle(f32) = null;
            for (contours) |polygon| {
                if (polygon.len < 3) continue;
                vertices += polygon.len;
                const box: Rectangle(f32) = .fromPoints(polygon);
                bounds = if (bounds) |acc| acc.merge(box) else box;
            }
            const shape_bounds = bounds orelse return;

            var stack: [polygon_scratch_size]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const scratch = buffer_first.allocator();
            const edges_buf = try scratch.alloc(Edge, vertices);
            defer scratch.free(edges_buf);
            const crossings_buf = try scratch.alloc(Crossing, vertices);
            defer scratch.free(crossings_buf);

            const edges = polygonEdges(contours, edges_buf);
            switch (mode) {
                .fast => {
                    comptime assert(@TypeOf(sink) == Paint);
                    const frows: f32 = @floatFromInt(self.image.rows);
                    const first_row = @max(0, @floor(shape_bounds.t));
                    const end_y = @min(frows - 1, @ceil(shape_bounds.b));
                    if (first_row > end_y) return;
                    const row_count: usize = @trunc(end_y - first_row + 1);
                    var sweep: ?EdgeSweep = if (edges.len < few_edges) null else try EdgeSweep.init(scratch, edges, first_row, row_count, 0);
                    defer if (sweep) |s| s.deinit(scratch);
                    for (0..row_count) |row| {
                        const y = first_row + as(f32, row);
                        const crossings = if (sweep) |*s| s.crossingsAt(row, y, crossings_buf) else scanlineCrossings(edges, y, crossings_buf);
                        var spans: SpanIter = .{ .crossings = crossings, .rule = fill_rule };
                        while (spans.next()) |span| self.fillSpan(span[0], span[1], y, sink);
                    }
                },
                .soft => switch (fill_rule) {
                    .nonzero => try self.fillPolygonArea(edges, shape_bounds, sink, scratch),
                    .even_odd => try self.fillPolygonSoft(edges, shape_bounds, crossings_buf, fill_rule, sink, scratch),
                },
            }
        }

        /// Antialiased nonzero fill by signed-area accumulation: every edge deposits its exact
        /// area and cover contribution into the cells of the rows it crosses, and a running
        /// sum along each row turns them into coverage (the winding's magnitude, clamped).
        /// Work follows the edges and the cells they touch rather than the rows of the shape,
        /// so tall hollow outlines cost their perimeter, not their area.
        fn fillPolygonArea(self: Self, edges: []const Edge, bounds: Rectangle(f32), sink: anytype, scratch: std.mem.Allocator) !void {
            // Pixel (r, c) spans [c - 0.5, c + 0.5) x [r - 0.5, r + 0.5); in accumulation space
            // it is the unit cell at (r - row_start, c - col_start).
            const frows: f32 = @floatFromInt(self.image.rows);
            const row_start: usize = @floor(@max(0, bounds.t + 0.5));
            const row_end: usize = @ceil(@min(frows, bounds.b + 0.5));
            if (row_start >= row_end) return;
            const height = row_end - row_start;
            const col_start: i64 = @floor(bounds.l + 0.5);
            const col_end: i64 = @ceil(bounds.r + 0.5);
            // One spare cell on the right for the last contribution.
            const width: usize = @intCast(col_end - col_start + 2);

            // Cells are zeroed block by block as edges first touch them; the resolve leaps
            // over untouched blocks while the running sum is zero.
            const blocks = (width + area_block - 1) / area_block;
            const acc = try scratch.alloc(f32, blocks * area_block * height);
            defer scratch.free(acc);
            const touched = try scratch.alloc(u8, blocks * height);
            defer scratch.free(touched);
            @memset(touched, 0);
            const shift_x = 0.5 - as(f32, col_start);
            const shift_y = 0.5 - as(f32, row_start);
            for (edges) |e| {
                accumulateEdge(acc, touched, width, height, .init(.{ e.p1.x() + shift_x, e.p1.y() + shift_y }), .init(.{ e.p2.x() + shift_x, e.p2.y() + shift_y }));
            }

            // Cells left of the image still feed the running sum; only the visible ones paint.
            const image_cols: i64 = @intCast(self.image.cols);
            const visible_lo: usize = @intCast(clamp(-col_start, 0, @as(i64, @intCast(width))));
            const visible_hi: usize = @intCast(clamp(image_cols - col_start, 0, @as(i64, @intCast(width))));
            if (visible_lo >= visible_hi) return;
            const threshold = 1.0 / 512.0;
            for (0..height) |r| {
                const row_acc = acc[r * blocks * area_block ..][0 .. blocks * area_block];
                const row_touched = touched[r * blocks ..][0..blocks];
                const row_px = self.image.data[(row_start + r) * self.image.stride ..];
                var sum: f32 = 0;
                for (row_touched, 0..) |flag, b| {
                    if (flag == 0 and @abs(sum) <= threshold) continue;
                    // Cells before the visible range only feed the sum.
                    const lo = b * area_block;
                    const hi = @min(lo + area_block, visible_hi);
                    const paint_from = @max(lo, visible_lo);
                    for (row_acc[lo..paint_from]) |a| sum += a;
                    if (paint_from >= hi) continue;
                    const px_base: usize = @intCast(col_start + @as(i64, @intCast(paint_from)));
                    for (row_acc[paint_from..hi], row_px[px_base..][0 .. hi - paint_from]) |a, *px| {
                        sum += a;
                        // Snap accumulation error to full or no coverage so the paint takes its
                        // overwrite path on the interior.
                        const alpha = @abs(sum);
                        if (alpha >= 1 - threshold) sink.cover(px, 1) else if (alpha > threshold) sink.cover(px, alpha);
                    }
                }
            }
        }

        /// Antialiased fill. Each pixel row is sampled at `polygon_subscanlines` heights;
        /// span ends contribute their exact horizontal overlap and fully covered interiors go
        /// through a difference array, so per-row cost does not scale with the sample count.
        /// Polygons have no closed-form distance field, hence supersampling rather than the
        /// analytic coverage used for rings and lines.
        fn fillPolygonSoft(self: Self, edges: []const Edge, bounds: Rectangle(f32), crossings_buf: []Crossing, rule: FillRule, sink: anytype, scratch: std.mem.Allocator) !void {
            const frows: f32 = @floatFromInt(self.image.rows);
            const end_y = @min(frows - 1, @ceil(bounds.b));
            const col_start = clampToImageBounds(@floor(bounds.l - 0.5), self.image.cols);
            const col_end = clampToImageBounds(@ceil(bounds.r + 0.5) + 1, self.image.cols);
            if (col_start >= col_end) return;
            const width = col_end - col_start;

            const cells = try scratch.alloc(CoverageCell, width);
            defer scratch.free(cells);
            @memset(cells, .{});

            // Buffer-relative x: cell j covers [j, j+1), i.e. image column col_start + j.
            const x_lo = as(f32, col_start) - 0.5;
            const x_hi = as(f32, col_end) - 0.5;
            const weight = 1.0 / as(f32, polygon_subscanlines);
            const first_row = @max(0, @floor(bounds.t));
            if (first_row > end_y) return;
            const row_count: usize = @trunc(end_y - first_row + 1);
            var sweep: EdgeSweep = try .init(scratch, edges, first_row, row_count, 0.5);
            defer sweep.deinit(scratch);

            for (0..row_count) |row_index| {
                const y = first_row + as(f32, row_index);
                var touched_lo: usize = width;
                var touched_hi: usize = 0;
                for (0..polygon_subscanlines) |k| {
                    const sy = y - 0.5 + (as(f32, k) + 0.5) * weight;
                    var spans: SpanIter = .{ .crossings = sweep.crossingsAt(row_index, sy, crossings_buf), .rule = rule };
                    while (spans.next()) |span| {
                        const left = @max(span[0], x_lo) - x_lo;
                        const right = @min(span[1], x_hi) - x_lo;
                        if (right <= left) continue;

                        const first: usize = @floor(left);
                        const last_raw: usize = @floor(right);
                        const last = @min(last_raw, width - 1);
                        if (first == last) {
                            cells[first].area += (right - left) * weight;
                        } else {
                            cells[first].area += (as(f32, first + 1) - left) * weight;
                            cells[last].area += (right - as(f32, last)) * weight;
                            cells[first + 1].run += weight;
                            cells[last].run -= weight;
                        }
                        touched_lo = @min(touched_lo, first);
                        touched_hi = @max(touched_hi, last);
                    }
                }
                if (touched_lo > touched_hi) continue;

                const row: u32 = @trunc(y);
                const row_px = self.image.data[row * self.image.stride + col_start ..][0..width];
                var run: f32 = 0;
                for (cells[touched_lo .. touched_hi + 1], row_px[touched_lo .. touched_hi + 1]) |*cell, *px| {
                    run += cell.run;
                    const alpha = @min(cell.area + run, 1);
                    cell.* = .{};
                    sink.cover(px, alpha);
                }
            }
        }

        /// Fills a circle on the given image.
        /// Use DrawMode.soft for anti-aliased edges or DrawMode.fast for hard edges.
        pub fn fillCircle(self: Self, center: Point(2, f32), radius: f32, color: anytype, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;

            switch (opts.mode) {
                .fast => self.fillCircleFast(center, radius, color, opts.blending),
                .soft => self.renderRing(center, 0, radius, color, .full, opts),
            }
        }

        /// Fills a pie slice (arc including the center point). Angles are in radians from the
        /// positive X-axis, counter-clockwise. Arcs spanning ≥ 2π render as a full circle.
        ///
        /// Example:
        /// ```zig
        /// // Green pie slice from π/4 to 3π/4
        /// try canvas.fillArc(center, 60, std.math.pi / 4.0, 3.0 * std.math.pi / 4.0, Rgb.green, .soft);
        /// ```
        pub fn fillArc(self: Self, center: Point(2, f32), radius: f32, start_angle: f32, end_angle: f32, color: anytype, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (radius <= 0) return;
            const arc = ArcRange.fromAngles(start_angle, end_angle) orelse return;
            if (arc.isFull()) return self.fillCircle(center, radius, color, opts);
            switch (opts.mode) {
                .fast => self.fillArcFast(center, radius, arc, color, opts.blending),
                .soft => try self.fillArcSoft(center, radius, arc, color, opts.blending),
            }
        }

        /// Internal function for filling solid (non-anti-aliased) circles.
        fn fillCircleFast(self: Self, center: Point(2, f32), radius: f32, color: anytype, blending: Blending) void {
            const paint: Paint = .init(color, blending);
            const frows: f32 = @floatFromInt(self.image.rows);
            const top = @max(0, center.y() - radius);
            const bottom = @min(frows - 1, center.y() + radius);

            var y = top;
            while (y <= bottom) : (y += 1) {
                const dy = y - center.y();
                const dx = @sqrt(@max(0, radius * radius - dy * dy));

                if (dx > 0) {
                    const x1 = center.x() - dx;
                    const x2 = center.x() + dx;
                    self.fillSpan(x1, x2, y, paint);
                }
            }
        }

        /// Internal function for filling solid (non-anti-aliased) arcs.
        /// Fills a pie slice (arc + lines to center).
        fn fillArcFast(self: Self, center: Point(2, f32), radius: f32, arc: ArcRange, color: anytype, blending: Blending) void {
            if (arc.span() <= 0) return;

            const paint: Paint = .init(color, blending);
            const frows: f32 = @floatFromInt(self.image.rows);
            const fcols: f32 = @floatFromInt(self.image.cols);

            // Half-plane edges: a point (dx, dy) is "after" start_angle CCW iff
            // dx*sin(start) - dy*cos(start) <= 0, and "before" end_angle CCW iff
            // dx*sin(end) - dy*cos(end) >= 0. For span <= π the wedge is their intersection;
            // for span > π it's their union (everything except the smaller complementary wedge).
            const sv = arc.startVector();
            const ev = arc.endVector();
            const sin_s = sv.y();
            const cos_s = sv.x();
            const sin_e = ev.y();
            const cos_e = ev.x();
            const cx_sin_s = center.x() * sin_s;
            const cx_sin_e = center.x() * sin_e;

            const top = @max(0, center.y() - radius);
            const bottom = @min(frows - 1, center.y() + radius);
            const left = @max(0, center.x() - radius);
            const right = @min(fcols - 1, center.x() + radius);

            const radius_sq = radius * radius;

            var y = top;
            while (y <= bottom) : (y += 1) {
                const dy = y - center.y();
                const dx_max_sq = radius_sq - dy * dy;
                if (dx_max_sq <= 0) continue;
                const dx_max = @sqrt(dx_max_sq);
                const scan_left = @max(left, center.x() - dx_max);
                const scan_right = @min(right, center.x() + dx_max);

                // Per-scanline x-independent components of the cross products.
                const start_const = -cx_sin_s - dy * cos_s;
                const end_const = -cx_sin_e - dy * cos_e;

                var x = scan_left;
                while (x <= scan_right) : (x += 1) {
                    if (!arc.containsCross(x * sin_s + start_const, x * sin_e + end_const)) continue;

                    var span_end = x;
                    while (span_end < scan_right) : (span_end += 1) {
                        const nx = span_end + 1;
                        const dnx = nx - center.x();
                        // Guard against scan_right's float imprecision pushing nx past the circle —
                        // without this, fillSpan's @ceil can include an extra pixel.
                        if (dnx * dnx + dy * dy > radius_sq) break;
                        if (!arc.containsCross(nx * sin_s + start_const, nx * sin_e + end_const)) break;
                    }

                    self.fillSpan(x, span_end, y, paint);
                    x = span_end;
                }
            }
        }

        /// Internal function for filling smooth (anti-aliased) arcs.
        fn fillArcSoft(self: Self, center: Point(2, f32), radius: f32, arc: ArcRange, color: anytype, blending: Blending) !void {
            // Precompute edge vectors
            const start_edge = arc.startVector();
            const end_edge = arc.endVector();

            const bounds = self.ringBoundingBox(center, radius) orelse return;
            const paint: Paint = .init(color, blending);

            for (bounds.t..bounds.b) |r| {
                const y = as(f32, r) - center.y();
                for (bounds.l..bounds.r) |c| {
                    const x = as(f32, c) - center.x();

                    const dist_sq = x * x + y * y;
                    if (dist_sq > (radius + 1) * (radius + 1)) continue;

                    const angle = std.math.atan2(y, x);
                    const in_arc = arc.contains(angle);

                    const p: Point(2, f32) = .init(.{ x, y });

                    const start_cross_product = p.cross(start_edge);
                    const end_cross_product = p.cross(end_edge);

                    if (!in_arc) {
                        const eps = 1e-5;
                        const near_start = @abs(start_cross_product) < 1.0 and start_cross_product < eps;
                        const near_end = @abs(end_cross_product) < 1.0 and end_cross_product > -eps;
                        if (!near_start and !near_end) continue;
                    }

                    const dist = @sqrt(dist_sq);
                    const coverage = calculateArcCoverage(dist, radius, in_arc, start_cross_product, end_cross_product);
                    paint.cover(&self.image.data[r * self.image.stride + c], coverage);
                }
            }
        }

        /// Draws a quadratic Bézier curve with specified width and fill mode.
        pub fn drawQuadraticBezier(
            self: Self,
            p0: Point(2, f32),
            p1: Point(2, f32),
            p2: Point(2, f32),
            color: anytype,
            width: u32,
            opts: DrawOptions,
        ) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            const estimated_length = estimateQuadraticBezierLength(p0, p1, p2);

            self.drawBezierTessellated(
                estimated_length,
                pixels_per_segment_quadratic,
                quadratic_min_segments_count,
                evalQuadraticBezier,
                .{ p0, p1, p2 },
                color,
                width,
                opts,
            );
        }

        /// Draws a cubic Bézier curve with specified width and fill mode.
        /// The curve is adaptively subdivided for optimal quality and performance.
        pub fn drawCubicBezier(
            self: Self,
            p0: Point(2, f32),
            p1: Point(2, f32),
            p2: Point(2, f32),
            p3: Point(2, f32),
            color: anytype,
            width: u32,
            opts: DrawOptions,
        ) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0) return;

            const estimated_length = estimateCubicBezierLength(p0, p1, p2, p3);
            const pixels_per_segment: f32 = if (opts.mode == .soft or width > 2) pixels_per_segment_soft else pixels_per_segment_fast;

            self.drawBezierTessellated(
                estimated_length,
                pixels_per_segment,
                spline_min_segments_count,
                evalCubicBezier,
                .{ p0, p1, p2, p3 },
                color,
                width,
                opts,
            );
        }

        /// Draws a spline polygon outline with Bézier curves connecting vertices.
        /// The polygon's edges are rendered as cubic Bézier splines for smooth, curved appearance.
        /// Use tension to control curve smoothness: 0=sharp corners, 1=maximum smoothness.
        pub fn drawSplinePolygon(self: Self, polygon: []const Point(2, f32), color: anytype, width: u32, tension: f32, opts: DrawOptions) void {
            comptime assert(isColor(@TypeOf(color)));
            if (width == 0 or polygon.len < 3) return;
            // The whole closed curve is tessellated first so thick strokes join the segments.
            var stack: [spline_polygon_stack_buffer_size * @sizeOf(Point(2, f32))]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const scratch = buffer_first.allocator();
            const pixels_per_segment: f32 = if (opts.mode == .soft or width > 2) pixels_per_segment_soft else pixels_per_segment_fast;
            const points = tessellateSpline(scratch, polygon, tension, pixels_per_segment, bezier_max_segments_count, true) catch return;
            defer scratch.free(points);
            self.strokePath(&.{points[0 .. points.len - 1]}, true, width, color, opts);
        }

        /// Fills a spline polygon with Bézier curves connecting vertices.
        /// The polygon's outline is defined by Bézier splines for smooth, curved edges.
        /// Use tension to control curve smoothness: 0=sharp corners, 1=maximum smoothness.
        pub fn fillSplinePolygon(self: Self, polygon: []const Point(2, f32), color: anytype, tension: f32, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            if (polygon.len < 3) return;
            var stack: [spline_polygon_stack_buffer_size * @sizeOf(Point(2, f32))]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const scratch = buffer_first.allocator();
            const points = try tessellateSpline(scratch, polygon, tension, pixels_per_segment_fast, spline_max_segments_count, false);
            defer scratch.free(points);
            try self.fillPolygon(points, color, opts);
        }

        /// Draws a Bézier curve by tessellating it into line segments.
        fn drawBezierTessellated(
            self: Self,
            estimated_length: f32,
            pixels_per_segment: f32,
            min_segments: u32,
            comptime evalFn: anytype,
            evalArgs: anytype,
            color: anytype,
            width: u32,
            opts: DrawOptions,
        ) void {
            var stack_buffer: [bezier_max_segments_count]Point(2, f32) = undefined;
            const points = stack_buffer[0..bezierSegments(estimated_length, pixels_per_segment, min_segments, bezier_max_segments_count)];
            tessellateBezier(evalFn, evalArgs, points);
            self.strokePath(&.{points}, false, width, color, opts);
        }

        /// Draws `text` with its top-left corner at `position`, at `font_size` pixels: the em height
        /// for vector fonts, the character height for bitmap fonts. `null` draws at
        /// `font.defaultSize()`, a bitmap font's native size. `\n` starts a new line.
        pub fn drawText(self: Self, text: []const u8, position: Point(2, f32), color: anytype, font: Font, font_size: ?f32, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            return self.renderText(text, unboundedBox(position), .init(color, opts.blending), font, font_size, .default, .fill, opts.mode);
        }

        /// Lays `text` out inside `box` as `layout` says: wrapped to the box width when asked,
        /// each line aligned horizontally and the block vertically. Text that does not fit
        /// is clipped by the image only, not the box.
        pub fn drawTextBox(self: Self, text: []const u8, box: Rectangle(f32), color: anytype, font: Font, font_size: ?f32, layout: TextLayout, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            return self.renderText(text, box, .init(color, opts.blending), font, font_size, layout, .fill, opts.mode);
        }

        /// `drawText` stroking each glyph's outline `width` pixels wide (round joins and caps)
        /// instead of filling it. Bitmap fonts have no outlines and get a halo of that
        /// diameter; draw the text over it for a readable label.
        pub fn drawTextOutline(self: Self, text: []const u8, position: Point(2, f32), color: anytype, font: Font, font_size: ?f32, width: f32, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            return self.renderText(text, unboundedBox(position), .init(color, opts.blending), font, font_size, .default, .{ .outline = width }, opts.mode);
        }

        /// `drawTextBox` stroking the glyphs as `drawTextOutline` does.
        pub fn drawTextBoxOutline(self: Self, text: []const u8, box: Rectangle(f32), color: anytype, font: Font, font_size: ?f32, width: f32, layout: TextLayout, opts: DrawOptions) !void {
            comptime assert(isColor(@TypeOf(color)));
            return self.renderText(text, box, .init(color, opts.blending), font, font_size, layout, .{ .outline = width }, opts.mode);
        }

        /// The box `drawText` uses: anchored at `position`, open to the right and below, so
        /// only left/top alignment without wrapping makes sense in it.
        fn unboundedBox(position: Point(2, f32)) Rectangle(f32) {
            return .{ .l = position.x(), .t = position.y(), .r = std.math.inf(f32), .b = std.math.inf(f32) };
        }

        /// The canvas area as a float rectangle, for clipping tests.
        fn imageRect(self: Self) Rectangle(f32) {
            return self.image.getRectangle().as(f32);
        }

        /// Shared by every text entry point: breaks `text` into lines, places the block and
        /// each line inside `box` per `layout`, and draws the lines. The lines are kept from
        /// the one wrapping pass, since placing the block needs their count first.
        fn renderText(self: Self, text: []const u8, box: Rectangle(f32), paint: Paint, font: Font, font_size: ?f32, layout: TextLayout, style: GlyphStyle, mode: DrawMode) !void {
            const px = font_size orelse font.defaultSize();
            if (px <= 0) return;
            const max_width: ?f32 = if (layout.wrap) box.width() else null;
            var lines: text_layout.Lines = .init(font, text, px, max_width, layout.letter_spacing);
            var stack: [lines_scratch_size]u8 align(16) = undefined;
            var buffer_first: std.heap.BufferFirstAllocator = .init(&stack, self.allocator);
            const scratch = buffer_first.allocator();
            var kept: std.ArrayList(text_layout.Lines.Line) = .empty;
            defer kept.deinit(scratch);
            while (lines.next()) |line| try kept.append(scratch, line);

            const advance = text_layout.lineAdvance(font, px, layout);
            const block = as(f32, kept.items.len) * advance;
            const top: f32 = switch (layout.valign) {
                .top => box.t,
                .middle => box.t + (box.height() - block) / 2,
                .bottom => box.b - block,
            };
            for (kept.items, 0..) |line, i| {
                const y = top + as(f32, i) * advance;
                const x: f32 = switch (layout.halign) {
                    .left => box.l,
                    .center => box.l + (box.width() - lines.width(line)) / 2,
                    .right => box.r - lines.width(line),
                };
                const position: Point(2, f32) = .init(.{ x, y });
                switch (font) {
                    .bitmap => |bitmap| try self.drawTextBitmap(line.text, position, paint, bitmap, bitmap.scaleFor(px), layout.letter_spacing, style, mode),
                    .vector => |vector| try self.drawTextVector(line.text, position, paint, vector, px, layout.letter_spacing, style, mode),
                }
            }
        }

        /// Lays out one line with `VectorFont.Layout`, then fills or strokes each glyph's
        /// outline. Glyphs that fail to parse are skipped; only allocation errors propagate.
        /// With a glyph cache, antialiased fills come from cached coverage masks.
        fn drawTextVector(self: Self, text: []const u8, position: Point(2, f32), paint: Paint, font: VectorFont, font_size: f32, letter_spacing: f32, style: GlyphStyle, mode: DrawMode) !void {
            const canvas_rect = self.imageRect();
            const margin = style.reach() + 1;
            var layout: VectorFont.Layout = .init(font, text, font_size);
            layout.letter_spacing = letter_spacing;
            const use_masks = font.cache != null and mode == .soft and style == .fill;
            while (layout.next()) |item| {
                const ink = layout.inkBounds(item) orelse continue;
                if (ink.translate(position.x(), position.y()).grow(margin).intersect(canvas_rect) == null) continue;
                const transform = layout.transform(item, position);
                if (use_masks and try self.drawCachedGlyph(font, item, ink, transform, paint)) continue;
                var ref = (try self.glyphOutline(font, item.gid)) orelse continue;
                defer ref.deinit(self.allocator);
                switch (mode) {
                    inline else => |m| try self.renderOutline(ref.outline, transform, style, m, paint),
                }
            }
        }

        /// The outline of glyph `gid`, or null when the font cannot provide one; only
        /// allocation failures propagate.
        fn glyphOutline(self: Self, font: VectorFont, gid: u16) !?VectorFont.OutlineRef {
            return font.outlineRef(self.allocator, gid) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => null,
            };
        }

        /// Paints `item` from its cached coverage mask, rasterizing one on a miss with the pen
        /// origin snapped to a quarter pixel. False when the mask is over the cache's size
        /// limit or cannot be allocated, so the caller draws the glyph directly.
        fn drawCachedGlyph(self: Self, font: VectorFont, item: VectorFont.Layout.Item, ink: Rectangle(f32), transform: Outline.Transform, paint: Paint) !bool {
            assert(transform.shear == 0);
            const cache = font.cache.?;
            const placed = GlyphCache.place(item.gid, transform);
            const mask = cache.getMask(placed.key) orelse blk: {
                // The snapped glyph's box relative to the integer pen position, with a pixel
                // of antialiasing margin.
                const box = ink.translate(placed.phase.x() - item.origin.x(), placed.phase.y() - item.origin.y()).grow(1);
                const left = @floor(box.l);
                const top = @floor(box.t);
                const width = @ceil(box.r) - left;
                const height = @ceil(box.b) - top;
                if (!cache.fits(width, height)) return false;
                var ref = (try self.glyphOutline(font, item.gid)) orelse return true;
                defer ref.deinit(self.allocator);
                const mask = cache.reserve(placed.key, @trunc(left), @trunc(top), @trunc(width), @trunc(height)) catch return false;
                errdefer cache.drop(placed.key);
                const mask_canvas: Canvas(u8) = .init(self.allocator, .initFromSlice(mask.height, mask.width, mask.data));
                const origin: Point(2, f32) = .init(.{ placed.phase.x() - left, placed.phase.y() - top });
                try mask_canvas.rasterizeGlyph(ref.outline, .{ .scale = transform.scale, .origin = origin });
                break :blk mask;
            };
            self.blitMask(.initFromSlice(mask.height, mask.width, mask.data), placed.x + mask.left, placed.y + mask.top, paint);
            return true;
        }

        /// Paints an 8-bit coverage mask with its top-left corner at (`left`, `top`), clipped
        /// to the image.
        fn blitMask(self: Self, mask: Image(u8), left: i32, top: i32, paint: Paint) void {
            const box: Rectangle(i32) = .{ .l = left, .t = top, .r = left + @as(i32, @intCast(mask.cols)), .b = top + @as(i32, @intCast(mask.rows)) };
            const clip = box.intersect(self.image.getRectangle().as(i32)) orelse return;
            // Whether every pixel takes the integer blend, decided once rather than per pixel.
            const opaque_normal = T == Rgb and paint.opaque_normal;
            var y = clip.t;
            while (y < clip.b) : (y += 1) {
                const src = mask.data[@as(usize, @intCast(y - top)) * mask.stride ..][@intCast(clip.l - left)..@intCast(clip.r - left)];
                const dst = self.image.data[@as(usize, @intCast(y)) * self.image.stride ..][@intCast(clip.l)..@intCast(clip.r)];
                if (opaque_normal) {
                    for (src, dst) |value, *px| {
                        if (value == 255) px.* = paint.solid else if (value != 0) paint.mixOpaque(px, value);
                    }
                } else for (src, dst) |value, *px| paint.coverByte(px, value);
            }
        }

        /// Draws one line of bitmap glyphs. An `.outline` style renders the line into a
        /// coverage mask, dilates it by the stroke radius and blits it once, so the halo
        /// composites like any other shape.
        fn drawTextBitmap(self: Self, text: []const u8, position: Point(2, f32), paint: Paint, font: BitmapFont, scale: f32, letter_spacing: f32, style: GlyphStyle, mode: DrawMode) !void {
            const radius: u32 = @ceil(style.reach());
            if (radius == 0) return self.blitBitmapLine(text, position, paint, font, scale, letter_spacing, mode);
            // A few overwriting stamps of the unscaled (hard-edged) glyphs cannot double-blend
            // and beat the mask. Stamping per glyph, not per line, keeps the text walk to one.
            if (paint.overwrite and scale == 1 and radius <= 2) {
                const r: i32 = @intCast(radius);
                var glyphs: BitmapFont.Layout = .init(font, text, 1);
                glyphs.x = position.x();
                glyphs.letter_spacing = letter_spacing;
                while (glyphs.next()) |placed| {
                    var dy: i32 = -r;
                    while (dy <= r) : (dy += 1) {
                        var dx: i32 = -r;
                        while (dx <= r) : (dx += 1) {
                            if (dx * dx + dy * dy > r * r) continue;
                            self.renderGlyphUnscaled(placed.glyph, placed.x + as(f32, dx), position.y() + as(f32, dy), paint);
                        }
                    }
                }
                return;
            }

            // The line's box grown by the radius, clipped to the image, in whole pixels.
            const extent = bitmapLineExtent(text, font, scale, letter_spacing);
            const line_rect: Rectangle(f32) = .{ .l = position.x(), .t = position.y(), .r = position.x() + extent.r, .b = position.y() + extent.b };
            const area = line_rect.grow(as(f32, radius)).intersect(self.imageRect()) orelse return;
            const left: u32 = @floor(area.l);
            const top: u32 = @floor(area.t);
            const width = @as(u32, @ceil(area.r)) - left;
            const height = @as(u32, @ceil(area.b)) - top;
            if (height == 0 or width == 0) return;

            // Source coverage plus the dilated result, then per-row scratch for the running max.
            const radius_px: usize = radius;
            const padded = width + 2 * radius_px;
            const buffers = try self.allocator.alloc(u8, 2 * height * width + 4 * padded);
            defer self.allocator.free(buffers);
            @memset(buffers, 0);
            const coverage = buffers[0 .. height * width];
            const dilated = buffers[height * width .. 2 * height * width];
            const mask: Canvas(u8) = .init(self.allocator, .initFromSlice(height, width, coverage));
            const origin: Point(2, f32) = .init(.{ position.x() - as(f32, left), position.y() - as(f32, top) });
            mask.blitBitmapLine(text, origin, .init(@as(u8, 255), .none), font, scale, letter_spacing, mode);

            // Dilate by a disc: every source row, widened by the disc's half-width at each
            // vertical offset, is max-merged into the rows it reaches.
            const row_max = buffers[2 * height * width ..][0..padded];
            const prefix = buffers[2 * height * width + padded ..][0..padded];
            const suffix = buffers[2 * height * width + 2 * padded ..][0..padded];
            const widened = buffers[2 * height * width + 3 * padded ..][0..padded];
            for (0..height) |sy| {
                const source = coverage[sy * width ..][0..width];
                if (std.mem.allEqual(u8, source, 0)) continue;
                var dy: usize = 0;
                while (dy <= radius_px) : (dy += 1) {
                    const half: usize = @trunc(@sqrt(as(f32, radius_px * radius_px - dy * dy)));
                    runningMax(source, half, row_max, prefix, suffix, widened[0..width]);
                    for ([_]bool{ false, true }) |up| {
                        if (up and dy == 0) continue;
                        const ty = if (up) std.math.sub(usize, sy, dy) catch continue else sy + dy;
                        if (ty >= height) continue;
                        const target = dilated[ty * width ..][0..width];
                        for (target, widened[0..width]) |*t, w| t.* = @max(t.*, w);
                    }
                }
            }
            self.blitMask(.initFromSlice(height, width, dilated), @intCast(left), @intCast(top), paint);
        }

        /// `out[i] = max(src[i - half ..= i + half])`, clipped to the row, in O(n) (van Herk):
        /// block prefix/suffix maxima over the zero-padded row.
        fn runningMax(src: []const u8, half: usize, padded: []u8, prefix: []u8, suffix: []u8, out: []u8) void {
            const n = src.len;
            if (half == 0) return @memcpy(out, src);
            const window = 2 * half + 1;
            const m = n + 2 * half;
            @memset(padded[0..half], 0);
            @memcpy(padded[half..][0..n], src);
            @memset(padded[half + n .. m], 0);
            for (0..m) |j| prefix[j] = if (j % window == 0) padded[j] else @max(prefix[j - 1], padded[j]);
            var j = m;
            while (j > 0) {
                j -= 1;
                suffix[j] = if (j % window == window - 1 or j == m - 1) padded[j] else @max(suffix[j + 1], padded[j]);
            }
            for (0..n) |i| out[i] = @max(suffix[i], prefix[i + window - 1]);
        }

        /// The box of one line of bitmap text relative to its pen origin: the font's bounds
        /// plus the letter spacing after each glyph.
        fn bitmapLineExtent(text: []const u8, font: BitmapFont, scale: f32, letter_spacing: f32) Rectangle(f32) {
            const glyphs = std.unicode.utf8CountCodepoints(text) catch text.len;
            var extent = font.getTextBounds(text, scale);
            extent.r += @max(letter_spacing, 0) * as(f32, glyphs);
            return extent;
        }

        /// Blits one line of bitmap glyphs at `position`.
        fn blitBitmapLine(self: Self, text: []const u8, position: Point(2, f32), paint: Paint, font: BitmapFont, scale: f32, letter_spacing: f32, mode: DrawMode) void {
            const text_rect = bitmapLineExtent(text, font, scale, letter_spacing).translate(position.x(), position.y());
            const clip_rect = text_rect.intersect(self.imageRect()) orelse return;

            const y = position.y();
            var glyphs: BitmapFont.Layout = .init(font, text, scale);
            glyphs.x = position.x();
            glyphs.letter_spacing = letter_spacing;
            while (glyphs.next()) |placed| {
                if (scale == 1.0) {
                    self.renderGlyphUnscaled(placed.glyph, placed.x, y, paint);
                } else switch (mode) {
                    .fast => self.renderGlyphFastScaled(placed.glyph, placed.x, y, scale, clip_rect, paint),
                    .soft => self.renderGlyphSoftScaled(placed.glyph, placed.x, y, scale, paint),
                }
            }
        }

        /// Blits a glyph 1:1. `x`/`y` are floored once, since floor commutes with adding the
        /// integer bit positions and offsets.
        fn renderGlyphUnscaled(self: Self, glyph: BitmapFont.Glyph, x: f32, y: f32, paint: Paint) void {
            const glyph_info = glyph.info;
            const fx: i32 = @floor(x);
            const fy: i32 = @floor(y);
            const base_col = fx + as(i32, glyph_info.x_offset);
            const base_row = fy + as(i32, glyph_info.y_offset);
            const rows_i32: i32 = @intCast(self.image.rows);
            const cols_i32: i32 = @intCast(self.image.cols);

            for (0..glyph_info.height) |row| {
                const py = base_row + as(i32, row);
                if (py < 0 or py >= rows_i32) continue;
                const row_offset: usize = @as(usize, @intCast(py)) * self.image.stride;
                for (0..glyph_info.width) |col| {
                    if (glyph.bit(row, col) == 0) continue;
                    const px = base_col + as(i32, col);
                    if (px < 0 or px >= cols_i32) continue;
                    paint.put(&self.image.data[row_offset + @as(usize, @intCast(px))]);
                }
            }
        }

        /// Nearest-neighbor upscale: each set glyph bit produces a `scale`-wide block of
        /// identical pixels, clipped to the precomputed text rect.
        fn renderGlyphFastScaled(self: Self, glyph: BitmapFont.Glyph, x: f32, y: f32, scale: f32, clip_rect: Rectangle(f32), paint: Paint) void {
            const glyph_info = glyph.info;
            for (0..glyph_info.height) |row| {
                for (0..glyph_info.width) |col| {
                    if (glyph.bit(row, col) == 0) continue;
                    const base_x = x + (as(f32, col) + as(f32, glyph_info.x_offset)) * scale;
                    const base_y = y + (as(f32, row) + as(f32, glyph_info.y_offset)) * scale;
                    const x_start: u32 = @trunc(@max(@floor(base_x), clip_rect.l));
                    const y_start: u32 = @trunc(@max(@floor(base_y), clip_rect.t));
                    const x_end: u32 = @trunc(@min(@ceil(base_x + scale), clip_rect.r));
                    const y_end: u32 = @trunc(@min(@ceil(base_y + scale), clip_rect.b));
                    if (x_start >= x_end or y_start >= y_end) continue;
                    // clip_rect is inside the image, so writes need no per-pixel bounds check.
                    for (y_start..y_end) |py| {
                        const row_offset = py * self.image.stride;
                        for (x_start..x_end) |px| {
                            const dest = &self.image.data[row_offset + px];
                            paint.put(dest);
                        }
                    }
                }
            }
        }

        /// Box-filter antialiased upscale: each destination pixel samples a `1/scale`-radius
        /// box of the source bitmap and writes the area-weighted coverage as alpha.
        fn renderGlyphSoftScaled(self: Self, glyph: BitmapFont.Glyph, x: f32, y: f32, scale: f32, paint: Paint) void {
            const glyph_info = glyph.info;
            const glyph_width_f = as(f32, glyph_info.width);
            const glyph_height_f = as(f32, glyph_info.height);
            const dest_width = @ceil(glyph_width_f * scale);
            const dest_height = @ceil(glyph_height_f * scale);
            const sample_radius = 0.5 / scale;

            var dy: f32 = 0;
            while (dy < dest_height) : (dy += 1) {
                var dx: f32 = 0;
                while (dx < dest_width) : (dx += 1) {
                    const dest_x = x + dx + as(f32, glyph_info.x_offset) * scale;
                    const dest_y = y + dy + as(f32, glyph_info.y_offset) * scale;
                    const dest = self.atOrNull(@floor(dest_y), @floor(dest_x)) orelse continue;

                    const src_x = dx / scale;
                    const src_y = dy / scale;
                    const x0 = src_x - sample_radius;
                    const x1 = src_x + sample_radius;
                    const y0 = src_y - sample_radius;
                    const y1 = src_y + sample_radius;

                    const row_start_f = @max(0, @floor(y0));
                    const row_end_f = @min(glyph_height_f - 1, @ceil(y1));
                    const col_start_f = @max(0, @floor(x0));
                    const col_end_f = @min(glyph_width_f - 1, @ceil(x1));

                    var total_coverage: f32 = 0;
                    var row_f = row_start_f;
                    while (row_f <= row_end_f) : (row_f += 1) {
                        var col_f = col_start_f;
                        while (col_f <= col_end_f) : (col_f += 1) {
                            const row_idx: u32 = @trunc(row_f);
                            const col_idx: u32 = @trunc(col_f);
                            if (glyph.bit(row_idx, col_idx) == 0) continue;
                            const overlap_x = clamp(@min(x1, col_f + 1) - @max(x0, col_f), 0, 1);
                            const overlap_y = clamp(@min(y1, row_f + 1) - @max(y0, row_f), 0, 1);
                            total_coverage += overlap_x * overlap_y;
                        }
                    }

                    const box_area = (x1 - x0) * (y1 - y0);
                    const normalized_coverage = total_coverage / box_area;
                    if (normalized_coverage > 0) {
                        paint.cover(dest, normalized_coverage);
                    }
                }
            }
        }
    };
}
