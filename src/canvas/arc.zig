//! Circles and arcs, independent of the pixel type: angular ranges and the per-pixel
//! coverage of rings and pie slices, plus the antialiasing convention they share.

const std = @import("std");
const clamp = std.math.clamp;

const Point = @import("../geometry/Point.zig").Point;
const as = @import("../meta.zig").as;
const DrawMode = @import("Canvas.zig").DrawMode;

/// Offset for antialiasing edge calculations (0.5 = pixel center alignment). Soft paths
/// treat pixel (r, c) as centered at (c, r); fast span writes are top-left inclusive.
pub const antialias_edge_offset = 0.5;

/// Angular range for arc filtering. `start`/`end` are
/// normalized to [0, 2π] with `end` shifted by +2π when the arc wraps past 0,
/// so that per-pixel `contains` checks need no `@mod`.
pub const ArcRange = struct {
    start: f32,
    end: f32,

    pub const full: ArcRange = .{ .start = 0, .end = 2 * std.math.pi };

    /// The arc swept from `start` to `end`: null for non-finite angles, `.full` when
    /// the sweep covers the whole circle.
    pub fn fromAngles(start: f32, end: f32) ?ArcRange {
        if (!std.math.isFinite(start) or !std.math.isFinite(end)) return null;
        if (@abs(end - start) >= 2 * std.math.pi) return .full;
        const ns = normalize(start);
        var ne = normalize(end);
        if (ne < ns) ne += 2 * std.math.pi;
        return .{ .start = ns, .end = ne };
    }

    /// Normalizes an angle to the [0, 2π] range. Use only when the input range is unknown;
    /// for atan2 outputs (already in [-π, π]) prefer the cheaper inline form in `contains`.
    pub fn normalize(angle: f32) f32 {
        var normalized = @mod(angle, 2 * std.math.pi);
        if (normalized < 0) normalized += 2 * std.math.pi;
        return normalized;
    }

    /// Tests whether an `atan2`-produced angle lies within the precomputed arc range.
    /// Caller must pass a value in [-π, π] (i.e., the output of `std.math.atan2`); other
    /// inputs require a prior `normalize` call.
    pub inline fn contains(self: ArcRange, angle: f32) bool {
        // atan2 ∈ [-π, π] — one conditional add suffices to reach [0, 2π].
        const norm_angle = if (angle < 0) angle + 2 * std.math.pi else angle;
        if (norm_angle >= self.start and norm_angle <= self.end) return true;
        const shifted = norm_angle + 2 * std.math.pi;
        return shifted >= self.start and shifted <= self.end;
    }

    /// Returns the absolute angular span of the arc.
    pub inline fn span(self: ArcRange) f32 {
        return self.end - self.start;
    }

    /// Returns the absolute geometric length of the arc along the specified radius.
    pub inline fn length(self: ArcRange, radius: f32) f32 {
        return self.span() * radius;
    }

    /// Returns true if the arc spans more than half a circle (π radians).
    pub inline fn isLong(self: ArcRange) bool {
        return self.span() > std.math.pi;
    }

    /// Returns true if the arc spans a full circle (≥ 2π radians).
    pub inline fn isFull(self: ArcRange) bool {
        return self.span() >= 2 * std.math.pi;
    }

    /// Returns the directional vector for the start of the arc.
    pub inline fn startVector(self: ArcRange) Point(2, f32) {
        return .init(.{ @cos(self.start), @sin(self.start) });
    }

    /// Returns the directional vector for the end of the arc.
    pub inline fn endVector(self: ArcRange) Point(2, f32) {
        return .init(.{ @cos(self.end), @sin(self.end) });
    }

    /// Half-plane test for "angle in arc" using precomputed cross-product components.
    pub inline fn containsCross(self: ArcRange, start_cross: f32, end_cross: f32) bool {
        const a = start_cross <= 0;
        const b = end_cross >= 0;
        return if (self.isLong()) (a or b) else (a and b);
    }
};

/// Coverage in the annulus [inner_r, outer_r] for a pixel at offset (x,y) from the
/// center. `aa=true` returns boundary-centered coverage in [0,1] (~0.5 at the
/// geometric edge); `aa=false` returns 1.0 strictly inside the ring, 0.0 otherwise.
/// `inner_r <= 0` disables the inner edge — pass 0 to fill a disk.
pub inline fn ringCoverage(x: f32, y: f32, inner_r: f32, outer_r: f32, comptime mode: DrawMode) f32 {
    const dist_sq = x * x + y * y;
    if (mode == .soft) {
        const dist = @sqrt(dist_sq);
        if (dist > outer_r + antialias_edge_offset) return 0;
        if (inner_r > 0 and dist < inner_r - antialias_edge_offset) return 0;
        var alpha: f32 = 1.0;
        if (dist > outer_r - antialias_edge_offset) alpha = @min(alpha, outer_r + antialias_edge_offset - dist);
        if (inner_r > 0 and dist < inner_r + antialias_edge_offset) alpha = @min(alpha, dist - (inner_r - antialias_edge_offset));
        return clamp(alpha, 0, 1);
    } else {
        const inside_outer = dist_sq <= outer_r * outer_r;
        const outside_inner = inner_r <= 0 or dist_sq >= inner_r * inner_r;
        return if (inside_outer and outside_inner) 1.0 else 0.0;
    }
}

/// Helper: Calculate antialiased coverage for arc boundaries
pub inline fn calculateArcCoverage(dist: f32, radius: f32, in_arc: bool, start_cross_product: f32, end_cross_product: f32) f32 {
    const start_cross = @abs(start_cross_product);
    const end_cross = @abs(end_cross_product);

    // Circular boundary coverage
    const circ_coverage = if (dist <= radius - 1.0)
        1.0
    else if (dist < radius + 1.0)
        clamp(radius - dist + 0.5, 0, 1)
    else
        0.0;

    const eps = 1e-5;

    if (!in_arc) {
        // Outside arc - apply edge antialiasing
        var edge_coverage: f32 = 0;
        if (start_cross < 1.0 and start_cross_product < eps) edge_coverage = @max(edge_coverage, 1.0 - start_cross);
        if (end_cross < 1.0 and end_cross_product > -eps) edge_coverage = @max(edge_coverage, 1.0 - end_cross);
        return circ_coverage * edge_coverage;
    } else {
        // Inside arc - reduce coverage near edges
        var coverage = circ_coverage;
        if (start_cross < 1.0 and start_cross_product >= -eps) coverage = @min(coverage, start_cross);
        if (end_cross < 1.0 and end_cross_product <= eps) coverage = @min(coverage, end_cross);
        return coverage;
    }
}

/// Populates `buf` with points along a circular arc, starting at `start_angle` and
/// stepping by `angle_step` for each successive index.
pub fn fillArcRing(buf: []Point(2, f32), center: Point(2, f32), radius: f32, start_angle: f32, angle_step: f32) void {
    for (buf, 0..) |*p, i| {
        const angle = start_angle + as(f32, i) * angle_step;
        p.* = .init(.{
            center.x() + radius * @cos(angle),
            center.y() + radius * @sin(angle),
        });
    }
}
