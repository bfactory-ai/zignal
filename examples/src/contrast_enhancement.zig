const std = @import("std");
const builtin = @import("builtin");
const zignal = @import("zignal");

const Image = zignal.Image;
const Rgba = zignal.Rgba;
const js = @import("js.zig");

pub const std_options: std.Options = .{
    .logFn = if (builtin.cpu.arch.isWasm()) js.logFn else std.log.defaultLog,
    .log_level = .info,
};

pub const alloc = js.alloc;
pub const free = js.free;

pub fn panic(msg: []const u8, st: ?*std.builtin.StackTrace, addr: ?usize) noreturn {
    _ = st;
    _ = addr;
    std.log.err("panic: {s}", .{msg});
    @trap();
}

fn imageFromPtr(rgba_ptr: [*]Rgba, rows: usize, cols: usize) Image(Rgba) {
    const total_pixels = std.math.mul(usize, rows, cols) catch @panic("contrast_enhancement: image too large");
    return Image(Rgba).initFromSlice(rows, cols, rgba_ptr[0..total_pixels]);
}

pub export fn autocontrast_inplace(rgba_ptr: [*]Rgba, rows: usize, cols: usize, cutoff: f32) void {
    const sanitized_cutoff = std.math.clamp(cutoff, 0.0, 0.49);
    const image = imageFromPtr(rgba_ptr, rows, cols);
    image.autocontrast(sanitized_cutoff) catch |err| {
        std.log.err("autocontrast failed: {s}", .{@errorName(err)});
    };
}

pub export fn equalize_inplace(rgba_ptr: [*]Rgba, rows: usize, cols: usize) void {
    const image = imageFromPtr(rgba_ptr, rows, cols);
    image.equalize();
}

pub export fn blend_autocontrast_equalize(
    rgba_ptr: [*]Rgba,
    rows: usize,
    cols: usize,
    cutoff: f32,
    blend: f32,
) void {
    const sanitized_cutoff = std.math.clamp(cutoff, 0.0, 0.49);
    const t = std.math.clamp(blend, 0.0, 1.0);
    const total_pixels = std.math.mul(usize, rows, cols) catch @panic("contrast_enhancement: image too large");

    var temp_allocator_state = std.heap.ArenaAllocator.init(std.heap.wasm_allocator);
    defer temp_allocator_state.deinit();
    const allocator = temp_allocator_state.allocator();

    var dest_image = imageFromPtr(rgba_ptr, rows, cols);
    var original = dest_image.dupe(allocator) catch |err| {
        std.log.err("dupe failed: {s}", .{@errorName(err)});
        return;
    };
    defer original.deinit(allocator);

    dest_image.autocontrast(sanitized_cutoff) catch |err| {
        std.log.err("autocontrast failed: {s}", .{@errorName(err)});
        return;
    };

    var equalized = original.dupe(allocator) catch |err| {
        std.log.err("dupe failed: {s}", .{@errorName(err)});
        return;
    };
    defer equalized.deinit(allocator);
    equalized.equalize();

    // Blend between autocontrast (already in dest) and equalized version
    const auto_slice = dest_image.data[0..total_pixels];
    const eq_slice = equalized.data[0..total_pixels];
    for (auto_slice, eq_slice) |*auto_px, eq_px| {
        auto_px.r = blendChannel(auto_px.r, eq_px.r, t);
        auto_px.g = blendChannel(auto_px.g, eq_px.g, t);
        auto_px.b = blendChannel(auto_px.b, eq_px.b, t);
        auto_px.a = blendChannel(auto_px.a, eq_px.a, t);
    }
}

inline fn blendChannel(a: u8, b: u8, t: f32) u8 {
    return @intFromFloat(@round(@as(f32, @floatFromInt(a)) * (1.0 - t) + @as(f32, @floatFromInt(b)) * t));
}
