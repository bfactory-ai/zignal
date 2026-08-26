//! Bézier and spline tessellation into polylines, independent of the pixel type.

const std = @import("std");
const clamp = std.math.clamp;

const Point = @import("../geometry/Point.zig").Point;
const as = @import("../meta.zig").as;

/// Maximum number of line segments when tessellating Bézier curves for line drawing
pub const bezier_max_segments_count = 200;

/// Maximum number of line segments when tessellating spline polygons
pub const spline_max_segments_count = 50;

/// Minimum number of line segments for spline curves to ensure reasonable quality
pub const spline_min_segments_count = 4;

/// Minimum number of line segments for quadratic Bézier curves
pub const quadratic_min_segments_count = 3;

/// Target pixels per segment for smooth/antialiased rendering (higher quality, more segments)
pub const pixels_per_segment_soft = 1.5;

/// Target pixels per segment for solid/fast rendering (lower quality, fewer segments)
pub const pixels_per_segment_fast = 3.0;

/// Target pixels per segment specifically for quadratic Bézier curves
pub const pixels_per_segment_quadratic = 2.0;

/// The arguments of `evalCubicBezier` before `t`.
pub const CubicBezier = struct { Point(2, f32), Point(2, f32), Point(2, f32), Point(2, f32) };

/// Evaluates a quadratic Bézier curve at parameter t.
/// Uses the standard quadratic Bézier formula: (1-t)²P₀ + 2t(1-t)P₁ + t²P₂
/// Parameter t is in range [0, 1] where 0=start point, 1=end point.
pub fn evalQuadraticBezier(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), t: f32) Point(2, f32) {
    const u = 1 - t;
    const uu = u * u;
    const tt = t * t;
    return .init(.{
        uu * p0.x() + 2 * u * t * p1.x() + tt * p2.x(),
        uu * p0.y() + 2 * u * t * p1.y() + tt * p2.y(),
    });
}

/// Evaluates a cubic Bézier curve at parameter t.
/// Uses the standard cubic Bézier formula: (1-t)³P₀ + 3t(1-t)²P₁ + 3t²(1-t)P₂ + t³P₃
/// Parameter t is in range [0, 1] where 0=start point, 1=end point.
pub fn evalCubicBezier(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), p3: Point(2, f32), t: f32) Point(2, f32) {
    const u = 1 - t;
    const uu = u * u;
    const uuu = uu * u;
    const tt = t * t;
    const ttt = tt * t;
    return .init(.{
        uuu * p0.x() + 3 * uu * t * p1.x() + 3 * u * tt * p2.x() + ttt * p3.x(),
        uuu * p0.y() + 3 * uu * t * p1.y() + 3 * u * tt * p2.y() + ttt * p3.y(),
    });
}

/// Estimates the length of a quadratic Bézier curve segment.
/// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
/// The estimate is (chord_length + control_polygon_length) / 2.
pub fn estimateQuadraticBezierLength(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32)) f32 {
    // Use chord + control polygon approximation
    const chord = p0.distance(p2);
    const control_net = p0.distance(p1) + p1.distance(p2);
    return (chord + control_net) / 2.0;
}

/// Estimates the length of a cubic Bézier curve segment.
/// Uses chord + control polygon approximation for fast, reasonably accurate estimation.
/// The estimate is (chord_length + control_polygon_length) / 2.
pub fn estimateCubicBezierLength(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), p3: Point(2, f32)) f32 {
    // Use chord + control polygon approximation
    const chord = p0.distance(p3);
    const control_net = p0.distance(p1) + p1.distance(p2) + p2.distance(p3);
    return (chord + control_net) / 2.0;
}

/// Points that render a curve `estimated_length` pixels long at `pixels_per_segment`.
pub fn bezierSegments(estimated_length: f32, pixels_per_segment: f32, min_segments: u32, max_segments: u32) u32 {
    return @max(min_segments, @min(max_segments, @as(u32, @trunc(estimated_length / pixels_per_segment))));
}

/// Evaluates a curve at parameters evenly spaced over [0, 1], one per point of `buffer`.
pub fn tessellateBezier(comptime evalFn: anytype, evalArgs: anytype, buffer: []Point(2, f32)) void {
    for (buffer, 0..) |*p, i| {
        const t = as(f32, i) / as(f32, buffer.len - 1);
        p.* = @call(.auto, evalFn, evalArgs ++ .{t});
    }
}

/// Calculates cubic Bézier control points (`cp1` outgoing from p0, `cp2` incoming to p1)
/// for a smooth curve through `p1` influenced by neighbors `p0`/`p2`. `tension` ranges
/// from 0 (sharp corners) to 1 (maximum smoothness).
pub fn calculateSmoothControlPoints(p0: Point(2, f32), p1: Point(2, f32), p2: Point(2, f32), tension: f32) struct { cp1: Point(2, f32), cp2: Point(2, f32) } {
    const tension_factor = 1 - clamp(tension, 0, 1);
    return .{
        .cp1 = .init(.{
            p0.x() + (p1.x() - p0.x()) * tension_factor,
            p0.y() + (p1.y() - p0.y()) * tension_factor,
        }),
        .cp2 = .init(.{
            p1.x() - (p2.x() - p1.x()) * tension_factor,
            p1.y() - (p2.y() - p1.y()) * tension_factor,
        }),
    };
}

/// Edge `i` of the closed spline through `polygon`: its cubic Bézier and how many
/// points render it.
pub fn splineEdge(polygon: []const Point(2, f32), i: usize, tension: f32, pixels_per_segment: f32, max_segments: u32) struct { curve: CubicBezier, segments: u32 } {
    const p0 = polygon[i];
    const p1 = polygon[(i + 1) % polygon.len];
    const p2 = polygon[(i + 2) % polygon.len];
    const cps = calculateSmoothControlPoints(p0, p1, p2, tension);
    const length = estimateCubicBezierLength(p0, cps.cp1, cps.cp2, p1);
    return .{
        .curve = .{ p0, cps.cp1, cps.cp2, p1 },
        .segments = bezierSegments(length, pixels_per_segment, spline_min_segments_count, max_segments),
    };
}

/// The closed spline through `polygon` as a polyline in `scratch` memory, one cubic
/// Bézier per edge. With `overlap` each edge starts on the previous one's end point
/// instead of repeating it, and the last point closes the loop back onto the first.
pub fn tessellateSpline(scratch: std.mem.Allocator, polygon: []const Point(2, f32), tension: f32, pixels_per_segment: f32, max_segments: u32, overlap: bool) ![]Point(2, f32) {
    const shared: usize = @intFromBool(overlap);
    var total: usize = shared;
    for (0..polygon.len) |i| total += splineEdge(polygon, i, tension, pixels_per_segment, max_segments).segments - shared;
    const points = try scratch.alloc(Point(2, f32), total);
    var n: usize = 0;
    for (0..polygon.len) |i| {
        const edge = splineEdge(polygon, i, tension, pixels_per_segment, max_segments);
        tessellateBezier(evalCubicBezier, edge.curve, points[n..][0..edge.segments]);
        n += edge.segments - shared;
    }
    return points;
}
