const std = @import("std");
const builtin = @import("builtin");

const Image = @import("zignal").Image;
const Rgba = @import("zignal").Rgba;
const Rgb = @import("zignal").Rgb;
const Xyz = @import("zignal").Xyz;

pub const std_options: std.Options = .{
    .logFn = if (builtin.cpu.arch.isWasm()) @import("js.zig").logFn else std.log.defaultLog,
    .log_level = if (builtin.mode == .Debug) .debug else .info,
};

pub fn panic(msg: []const u8, st: ?*std.builtin.StackTrace, addr: ?usize) noreturn {
    _ = st;
    _ = addr;
    std.log.err("panic: {s}", .{msg});
    @trap();
}

const RgbGains = struct {
    r: f64,
    g: f64,
    b: f64,
};

fn estimateIlluminant(image: Image(Rgba), color: Rgb, fraction: f64) RgbGains {
    var sum_r: f64 = 0;
    var sum_g: f64 = 0;
    var sum_b: f64 = 0;
    // Compute the average color per channel
    const sep: usize = @intFromFloat(@as(f32, @floatFromInt(image.rows)) * fraction);
    const size: f64 = @floatFromInt(image.cols * image.rows);
    for (0..sep) |r| {
        for (0..image.cols) |c| {
            const p = image.at(r, c);
            sum_r += @floatFromInt(p.r);
            sum_g += @floatFromInt(p.g);
            sum_b += @floatFromInt(p.b);
        }
    }
    for (sep..image.rows) |r| {
        for (0..image.cols) |c| {
            const p = image.at(r, c);
            p.r = color.r;
            p.g = color.g;
            p.b = color.b;
            sum_r += @floatFromInt(p.r);
            sum_g += @floatFromInt(p.g);
            sum_b += @floatFromInt(p.b);
        }
    }
    sum_r /= size;
    sum_g /= size;
    sum_b /= size;
    const avg = (sum_r + sum_g + sum_b) / 3.0;
    return .{ .r = sum_r / avg, .g = sum_g / avg, .b = sum_b / avg };
}

fn chromaticAdaptation(xyz: Xyz, w: RgbGains) Xyz {
    // Target illuminant (D65): LMS = (0.9642, 1.0000, 0.8252) (approx.)
    var lms = xyz.toLms();
    lms.l *= 0.9642 / w.r;
    lms.m *= 1.0000 / w.g;
    lms.s *= 0.8252 / w.b;
    return lms.toXyz();
}

pub export fn whitebalance(rgba_ptr: [*]Rgba, rows: usize, cols: usize, r: u8, g: u8, b: u8) void {
    const color: Rgb = .{ .r = r, .g = g, .b = b };
    std.log.info("color: {}, {}, {}\n", color);
    const image: Image(Rgba) = .init(rows, cols, rgba_ptr[0 .. rows * cols]);
    const w = estimateIlluminant(image, color, 0.7);
    for (image.data) |*p| {
        p.* = chromaticAdaptation(p.toXyz(), w).toRgba(p.a);
    }
}
