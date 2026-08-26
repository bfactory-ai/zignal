//! The offset-outline stroker: turns a polyline into the closed outline of its stroke,
//! with round joins and caps, for a nonzero fill.

const std = @import("std");

const Point = @import("../geometry/Point.zig").Point;
const as = @import("../meta.zig").as;
const Outline = @import("../font.zig").Outline;

/// Writes the outline of one stroked polyline.
pub const StrokeBuilder = struct {
    out: []Point(2, f32),
    len: usize = 0,
    radius: f32,
    step: f32,

    pub fn emit(b: *StrokeBuilder, p: Point(2, f32)) void {
        b.out[b.len] = p;
        b.len += 1;
    }

    pub fn offset(b: StrokeBuilder, p: Point(2, f32), dir: Point(2, f32), side: f32) Point(2, f32) {
        return p.add(perpendicular(dir).scale(side * b.radius));
    }

    /// Points on the circle around `center` from the unit direction `start`, sweeping
    /// `sweep` radians: the end included, the start not. The radius vector is rotated
    /// step by step, so an arc costs one sine and cosine.
    pub fn arc(b: *StrokeBuilder, center: Point(2, f32), start: Point(2, f32), sweep: f32) void {
        const steps: usize = @ceil(@max(1, @abs(sweep) / b.step));
        const angle = sweep / as(f32, steps);
        const c = @cos(angle);
        const sn = @sin(angle);
        var v = start.scale(b.radius);
        for (0..steps) |_| {
            v = .init(.{ v.x() * c - v.y() * sn, v.x() * sn + v.y() * c });
            b.emit(center.add(v));
        }
    }

    /// The join at `v` on `side` (+1 left, -1 right) between the incoming direction
    /// `in_dir` and the outgoing `out_dir`, both in traversal order.
    pub fn join(b: *StrokeBuilder, v: Point(2, f32), in_dir: Point(2, f32), out_dir: Point(2, f32), side: f32) void {
        const cross = in_dir.cross(out_dir);
        const dot = in_dir.dot(out_dir);
        // The left side lies outside a turn with negative cross (and a reversal).
        const outer = if (side > 0) cross < 0 or (cross == 0 and dot < 0) else cross > 0;
        b.emit(b.offset(v, in_dir, side));
        if (outer) {
            b.arc(v, perpendicular(in_dir).scale(side), std.math.atan2(cross, dot));
        } else if (cross != 0 or dot < 0) {
            // The inner offsets cross each other; the loop they close is invisible
            // below the flatness tolerance, otherwise detouring through the vertex
            // makes it wind like the stroke.
            const turn = std.math.atan2(@abs(cross), dot);
            if (b.radius * @tan(turn / 2) >= Outline.flatness_tolerance) b.emit(v);
            b.emit(b.offset(v, out_dir, side));
        }
    }

    pub fn polyline(b: *StrokeBuilder, input: []const Point(2, f32), closed: bool) void {
        const m = input.len + @intFromBool(closed and input.len > 1);
        const first_dir = for (0..m -| 1) |i| {
            if (unitDirection(input[i], input[(i + 1) % input.len])) |d| break d;
        } else {
            // Every point coincides: a dot.
            b.emit(input[0].add(.init(.{ b.radius, 0 })));
            b.arc(input[0], .init(.{ 1, 0 }), 2 * std.math.pi);
            return;
        };
        // Direction of each segment, degenerate ones borrowing their predecessor's.
        var dir = first_dir;
        b.emit(b.offset(input[0], dir, 1));
        for (1..m - 1) |i| {
            const next = unitDirection(input[i % input.len], input[(i + 1) % input.len]) orelse dir;
            b.join(input[i % input.len], dir, next, 1);
            dir = next;
        }
        const last = input[(m - 1) % input.len];
        b.emit(b.offset(last, dir, 1));
        // End cap: from the left offset around the tip to the right one.
        b.arc(last, perpendicular(dir), -std.math.pi);
        // Back along the right side; the incoming direction is now the later segment's.
        var i = m - 1;
        while (i > 1) : (i -= 1) {
            const prev = unitDirection(input[(i - 2) % input.len], input[(i - 1) % input.len]) orelse dir;
            b.join(input[(i - 1) % input.len], dir, prev, -1);
            dir = prev;
        }
        b.emit(b.offset(input[0], dir, -1));
        b.arc(input[0], perpendicular(dir).scale(-1), -std.math.pi);
    }
};

pub fn perpendicular(d: Point(2, f32)) Point(2, f32) {
    return .init(.{ -d.y(), d.x() });
}

pub fn signedArea(polygon: []const Point(2, f32)) f32 {
    var area: f32 = 0;
    for (polygon, 0..) |p, i| area += p.cross(polygon[(i + 1) % polygon.len]);
    return area / 2;
}

/// Unit vector from `p` to `q`, or null when they coincide.
pub fn unitDirection(p: Point(2, f32), q: Point(2, f32)) ?Point(2, f32) {
    const d = q.sub(p);
    const len = d.norm();
    return if (len > 0) d.scale(1 / len) else null;
}
