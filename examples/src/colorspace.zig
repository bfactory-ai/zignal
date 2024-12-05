const std = @import("std");
const builtin = @import("builtin");

const convert = @import("zignal").colorspace.convert;
const Hsl = @import("zignal").Hsl;
const Hsv = @import("zignal").Hsv;
const Lab = @import("zignal").Lab;
const Lms = @import("zignal").Lms;
const Oklab = @import("zignal").Oklab;
const Rgb = @import("zignal").Rgb;
const Xyb = @import("zignal").Xyb;
const Xyz = @import("zignal").Xyz;

pub const std_options: std.Options = .{
    .logFn = if (builtin.cpu.arch.isWasm()) @import("js.zig").logFn else std.log.defaultLog,
    .log_level = std.log.default_level,
};

// --- RGB ---

export fn rgb2hsl(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Hsl, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn rgb2hsv(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Hsv, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn rgb2xyz(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Xyz, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn rgb2lab(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Lab, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("Lab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn rgb2lms(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Lms, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn rgb2oklab(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Oklab, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn rgb2xyb(red: u8, green: u8, blue: u8, out: [*]f64) void {
    const res = convert(Xyb, Rgb{ .r = red, .g = green, .b = blue });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}
// --- HSL ---

export fn hsl2rgb(h: f64, s: f64, l: f64, out: [*]u8) void {
    const res = convert(Rgb, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn hsl2hsv(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Hsv, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn hsl2xyz(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Xyz, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn hsl2lab(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Lab, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("LAB: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn hsl2lms(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Lms, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn hsl2oklab(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Oklab, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn hsl2xyb(h: f64, s: f64, l: f64, out: [*]f64) void {
    const res = convert(Xyb, Hsl{ .h = h, .s = s, .l = l });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- HSV ---

export fn hsv2rgb(h: f64, s: f64, v: f64, out: [*]u8) void {
    const res = convert(Rgb, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn hsv2hsl(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Hsl, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn hsv2xyz(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Xyz, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn hsv2lab(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Lab, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("LAB: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn hsv2lms(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Lms, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn hsv2oklab(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Oklab, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn hsv2xyb(h: f64, s: f64, v: f64, out: [*]f64) void {
    const res = convert(Xyb, Hsv{ .h = h, .s = s, .v = v });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- XYZ ---

export fn xyz2rgb(x: f64, y: f64, z: f64, out: [*]u8) void {
    const res = convert(Rgb, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn xyz2hsl(x: f64, y: f64, z: f64, out: [*]f64) void {
    const res = convert(Hsl, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn xyz2hsv(x: f64, y: f64, z: f64, out: [*]f64) void {
    const res = convert(Hsv, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn xyz2lab(x: f64, y: f64, z: f64, out: [*]f64) void {
    const res = convert(Lab, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("LAB: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn xyz2lms(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Lms, Xyz{ .x = l, .y = m, .z = s });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn xyz2oklab(x: f64, y: f64, z: f64, out: [*]f64) void {
    const res = convert(Oklab, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn xyz2xyb(x: f64, y: f64, z: f64, out: [*]f64) void {
    const res = convert(Xyb, Xyz{ .x = x, .y = y, .z = z });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- LAB ---

export fn lab2rgb(l: f64, a: f64, b: f64, out: [*]u8) void {
    const res = convert(Rgb, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn lab2hsl(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsl, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn lab2hsv(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsv, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn lab2xyz(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Xyz, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn lab2lms(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Lms, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn lab2oklab(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Oklab, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn lab2xyb(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Xyb, Lab{ .l = l, .a = a, .b = b });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- LMS ---

export fn lms2rgb(l: f64, m: f64, s: f64, out: [*]u8) void {
    const res = convert(Rgb, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn lms2hsl(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Hsl, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn lms2hsv(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Hsv, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn lms2xyz(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Xyz, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn lms2lab(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Lms, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn lms2oklab(h: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Oklab, Lms{ .l = h, .m = m, .s = s });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn lms2xyb(l: f64, m: f64, s: f64, out: [*]f64) void {
    const res = convert(Xyb, Lms{ .l = l, .m = m, .s = s });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- Oklab ---

export fn oklab2rgb(l: f64, a: f64, b: f64, out: [*]u8) void {
    const res = convert(Rgb, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn oklab2hsl(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsl, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn oklab2hsv(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsv, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn oklab2xyz(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Xyz, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn oklab2lms(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Lms, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn oklab2lab(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Lab, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn oklab2xyb(l: f64, a: f64, b: f64, out: [*]f64) void {
    const res = convert(Xyb, Oklab{ .l = l, .a = a, .b = b });
    std.log.debug("Xyb: {[x]d} {[y]d} {[b]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.b;
}

// --- XYB ---

export fn xyb2rgb(x: f64, y: f64, b: f64, out: [*]u8) void {
    const res = convert(Rgb, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("RGB: {[r]d} {[g]d} {[b]d}", res);
    out[0] = res.r;
    out[1] = res.g;
    out[2] = res.b;
}

export fn xyb2hsl(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsl, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("HSL: {[h]d} {[s]d} {[l]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.l;
}

export fn xyb2hsv(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Hsv, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("HSV: {[h]d} {[s]d} {[v]d}", res);
    out[0] = res.h;
    out[1] = res.s;
    out[2] = res.v;
}

export fn xyb2xyz(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Xyz, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("XYZ: {[x]d} {[y]d} {[z]d}", res);
    out[0] = res.x;
    out[1] = res.y;
    out[2] = res.z;
}

export fn xyb2lms(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Lms, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("LMS: {[l]d} {[m]d} {[s]d}", res);
    out[0] = res.l;
    out[1] = res.m;
    out[2] = res.s;
}

export fn xyb2oklab(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Oklab, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("Oklab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}

export fn xyb2lab(x: f64, y: f64, b: f64, out: [*]f64) void {
    const res = convert(Lab, Xyb{ .x = x, .y = y, .b = b });
    std.log.debug("Lab: {[l]d} {[a]d} {[b]d}", res);
    out[0] = res.l;
    out[1] = res.a;
    out[2] = res.b;
}
