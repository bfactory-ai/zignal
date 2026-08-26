//! Scanline and signed-area polygon rasterization, independent of the pixel type: edges
//! and their crossings with scanlines, the edge sweep, the span iterator that applies a
//! fill rule, and the area accumulator of the antialiased nonzero fill.

const std = @import("std");
const clamp = std.math.clamp;

const Point = @import("../geometry/Point.zig").Point;
const as = @import("../meta.zig").as;
const FillRule = @import("Canvas.zig").FillRule;

/// Cells per touched-block flag in the area rasterizer.
pub const area_block = 8;

/// Polygon edge with its y-extent precomputed for scanline crossing tests.
pub const Edge = struct {
    p1: Point(2, f32),
    p2: Point(2, f32),
    y_min: f32,
    y_max: f32,
    /// +1 when the edge runs down the screen, -1 up; the winding contribution.
    dir: i8,

    /// x where the edge crosses the horizontal line at `y`, for y in [y_min, y_max).
    pub inline fn xAt(e: Edge, y: f32) f32 {
        return e.p1.x() + (y - e.p1.y()) * (e.p2.x() - e.p1.x()) / (e.p2.y() - e.p1.y());
    }
};

/// Where an edge crosses a scanline, with its winding direction. `dir` is a full
/// word so that sorting's whole-struct copies never read a byte-wide store.
pub const Crossing = struct {
    x: f32,
    dir: i32,

    pub fn lessThan(_: void, a: Crossing, b: Crossing) bool {
        return a.x < b.x;
    }
};

/// Fills `buf` with the non-horizontal edges of every contour (horizontal ones never
/// cross a scanline).
pub fn polygonEdges(contours: []const []const Point(2, f32), buf: []Edge) []Edge {
    var count: usize = 0;
    for (contours) |polygon| {
        if (polygon.len < 3) continue;
        for (polygon, 0..) |p1, i| {
            const p2 = polygon[(i + 1) % polygon.len];
            if (p1.y() == p2.y()) continue;
            buf[count] = .{
                .p1 = p1,
                .p2 = p2,
                .y_min = @min(p1.y(), p2.y()),
                .y_max = @max(p1.y(), p2.y()),
                .dir = if (p1.y() < p2.y()) 1 else -1,
            };
            count += 1;
        }
    }
    return buf[0..count];
}

/// The edges bucketed by the row they start on (a counting sort) and swept downward
/// once: `crossingsAt` activates the edges of each row as it comes and forgets those
/// passed, so a scanline only tests the edges spanning it. Scanlines must not move
/// back up.
pub const EdgeSweep = struct {
    edges: []const Edge,
    /// Edge indices grouped by starting row; row `r`'s group ends at `ends[r]`.
    order: []u32,
    ends: []u32,
    /// Edges started but not yet passed.
    active: []u32,
    slab: []u32,
    next: usize = 0,
    count: usize = 0,

    /// Sweeps `row_count` rows from `first_row`. An edge starts on the row of
    /// `y_min + shift`: 0.5 when rows are the bands around pixel centers.
    pub fn init(scratch: std.mem.Allocator, edges: []const Edge, first_row: f32, row_count: usize, shift: f32) !EdgeSweep {
        const slab = try scratch.alloc(u32, 2 * edges.len + row_count + 1);
        const order = slab[0..edges.len];
        const active = slab[edges.len..][0..edges.len];
        const ends = slab[2 * edges.len ..];
        @memset(ends, 0);
        for (edges) |e| ends[rowOf(e, first_row, row_count, shift) + 1] += 1;
        for (1..row_count + 1) |r| ends[r] += ends[r - 1];
        // Placing each edge at its row's cursor leaves the cursor at the row's end.
        for (edges, 0..) |e, i| {
            const row = rowOf(e, first_row, row_count, shift);
            order[ends[row]] = @intCast(i);
            ends[row] += 1;
        }
        return .{ .edges = edges, .order = order, .ends = ends, .active = active, .slab = slab };
    }

    pub fn rowOf(e: Edge, first_row: f32, row_count: usize, shift: f32) usize {
        return @floor(clamp(e.y_min + shift - first_row, 0, as(f32, row_count - 1)));
    }

    pub fn deinit(self: EdgeSweep, scratch: std.mem.Allocator) void {
        scratch.free(self.slab);
    }

    /// Crossings with the scanline at `y`, on row `row` (counted from `first_row`),
    /// sorted by x. Edges ending at or above `y` are dropped on the way.
    pub fn crossingsAt(self: *EdgeSweep, row: usize, y: f32, buf: []Crossing) []Crossing {
        while (self.next < self.ends[row]) : (self.next += 1) {
            self.active[self.count] = self.order[self.next];
            self.count += 1;
        }
        var count: usize = 0;
        var i: usize = 0;
        while (i < self.count) {
            const e = self.edges[self.active[i]];
            if (e.y_max <= y) {
                self.count -= 1;
                self.active[i] = self.active[self.count];
                continue;
            }
            if (y >= e.y_min) {
                buf[count] = .{ .x = e.xAt(y), .dir = e.dir };
                count += 1;
            }
            i += 1;
        }
        return sortCrossings(buf[0..count]);
    }
};

/// Crossings of all `edges` with the horizontal line at `y`, sorted by x.
pub fn scanlineCrossings(edges: []const Edge, y: f32, buf: []Crossing) []Crossing {
    var count: usize = 0;
    for (edges) |e| {
        if (y >= e.y_min and y < e.y_max) {
            buf[count] = .{ .x = e.xAt(y), .dir = e.dir };
            count += 1;
        }
    }
    return sortCrossings(buf[0..count]);
}

/// Sorts crossings by x. Rows rarely have more than a handful, where a plain
/// insertion sort beats the generic sorts' setup many times over.
pub fn sortCrossings(crossings: []Crossing) []Crossing {
    if (crossings.len <= 32) {
        for (1..@max(crossings.len, 1)) |i| {
            const c = crossings[i];
            var j = i;
            while (j > 0 and crossings[j - 1].x > c.x) : (j -= 1) crossings[j] = crossings[j - 1];
            crossings[j] = c;
        }
    } else {
        std.sort.pdq(Crossing, crossings, {}, Crossing.lessThan);
    }
    return crossings;
}

/// The spans of one scanline inside the shape under `rule`, from x-sorted crossings.
/// Even-odd toggles on every crossing, pairing them as a pairwise walk would; nonzero
/// sums the edge directions.
pub const SpanIter = struct {
    crossings: []const Crossing,
    rule: FillRule,
    i: usize = 0,
    winding: i32 = 0,
    left: f32 = 0,

    pub fn next(it: *SpanIter) ?[2]f32 {
        while (it.i < it.crossings.len) {
            const c = it.crossings[it.i];
            it.i += 1;
            const was_inside = it.winding != 0;
            it.winding = switch (it.rule) {
                .even_odd => it.winding ^ 1,
                .nonzero => it.winding + c.dir,
            };
            if (it.winding != 0) {
                if (!was_inside) it.left = c.x;
            } else if (was_inside) {
                return .{ it.left, c.x };
            }
        }
        return null;
    }
};

/// One pixel of a `fillPolygonSoft` row: `area` accumulates partial coverage at span
/// ends, `run` is a difference array for fully covered interiors.
pub const CoverageCell = struct { area: f32 = 0, run: f32 = 0 };

/// Marks the blocks holding cells `first..=last` of a row as touched, zeroing each
/// block the first time.
pub inline fn touchBlocks(row: []f32, row_touched: []u8, first: usize, last: usize) void {
    for (first / area_block..last / area_block + 1) |b| {
        if (row_touched[b] == 0) {
            row_touched[b] = 1;
            row[b * area_block ..][0..area_block].* = @splat(0);
        }
    }
}

/// Adds one edge's contributions to the accumulation buffer: `d` per row is the
/// signed height crossed, split between the cells the edge passes through by area.
/// Inlined into each `Canvas(T)` fill: shared between them out of line, wrapped vector
/// text measured 6% slower.
pub inline fn accumulateEdge(acc: []f32, touched: []u8, width: usize, height: usize, p0: Point(2, f32), p1: Point(2, f32)) void {
    if (p0.y() == p1.y()) return;
    const dir: f32 = if (p0.y() < p1.y()) 1 else -1;
    const top = if (dir > 0) p0 else p1;
    const bottom = if (dir > 0) p1 else p0;
    if (bottom.y() <= 0 or top.y() >= as(f32, height)) return;
    const dxdy = (bottom.x() - top.x()) / (bottom.y() - top.y());
    const x_max: f32 = as(f32, width - 2);
    const blocks = (width + area_block - 1) / area_block;
    const row_len = blocks * area_block;
    var x = top.x();
    var y: usize = 0;
    if (top.y() >= 0) {
        y = @floor(top.y());
    } else {
        x -= top.y() * dxdy;
    }
    const y_end: usize = @min(height, @as(usize, @ceil(bottom.y())));
    if (dxdy == 0) {
        // Vertical: the same two cells in every row, fully crossed except at the ends.
        const xc = @max(0, @min(x, x_max));
        const x_floor = @floor(xc);
        const xi: usize = @trunc(x_floor);
        const xmf = xc - x_floor;
        const full_start: usize = @ceil(clamp(top.y(), 0, as(f32, height)));
        const full_end: usize = @floor(clamp(bottom.y(), 0, as(f32, height)));
        const full_lo = dir - dir * xmf;
        const full_hi = dir * xmf;
        while (y < y_end) : (y += 1) {
            const row = acc[y * row_len ..][0..row_len];
            touchBlocks(row, touched[y * blocks ..][0..blocks], xi, xi + 1);
            if (y >= full_start and y < full_end) {
                row[xi] += full_lo;
                row[xi + 1] += full_hi;
            } else {
                const fy = as(f32, y);
                const d = (@min(fy + 1, bottom.y()) - @max(fy, top.y())) * dir;
                row[xi] += d - d * xmf;
                row[xi + 1] += d * xmf;
            }
        }
        return;
    }
    while (y < y_end) : (y += 1) {
        const row = acc[y * row_len ..][0..row_len];
        const fy = as(f32, y);
        const dy = @min(fy + 1, bottom.y()) - @max(fy, top.y());
        const x_next = x + dxdy * dy;
        const d = dy * dir;
        const x0 = @max(0, @min(@min(x, x_next), x_max));
        const x1 = @max(0, @min(@max(x, x_next), x_max));
        const x0_floor = @floor(x0);
        const x0i: usize = @trunc(x0_floor);
        const x1_ceil = @ceil(x1);
        const x1i: usize = @trunc(x1_ceil);
        touchBlocks(row, touched[y * blocks ..][0..blocks], x0i, x1i);
        if (x1i <= x0i + 1) {
            // Within one cell: split by the midpoint.
            const xmf = 0.5 * (x0 + x1) - x0_floor;
            row[x0i] += d - d * xmf;
            row[x0i + 1] += d * xmf;
        } else {
            const s = 1 / (x1 - x0);
            const x0f = x0 - x0_floor;
            const a0 = 0.5 * s * (1 - x0f) * (1 - x0f);
            const x1f = x1 - x1_ceil + 1;
            const am = 0.5 * s * x1f * x1f;
            row[x0i] += d * a0;
            if (x1i == x0i + 2) {
                row[x0i + 1] += d * (1 - a0 - am);
            } else {
                const a1 = s * (1.5 - x0f);
                row[x0i + 1] += d * (a1 - a0);
                for (x0i + 2..x1i - 1) |xi| row[xi] += d * s;
                const a2 = a1 + as(f32, x1i - x0i - 3) * s;
                row[x1i - 1] += d * (1 - a2 - am);
            }
            row[x1i] += d * am;
        }
        x = x_next;
    }
}
