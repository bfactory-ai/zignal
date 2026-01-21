const std = @import("std");
const zignal = @import("zignal");
const terminal = zignal.terminal;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const help_text =
    \\Usage: zignal view <image> [options]
    \\
    \\Display an image in the terminal using supported graphics protocols.
    \\
    \\Options:
    \\  --width <N>      Target width in pixels
    \\  --height <N>     Target height in pixels
    \\  --protocol <p>   Force protocol: kitty, sixel, sgr, braille, auto
    \\  --filter <f>     Scaling filter: nearest, bilinear, bicubic, lanczos, catmull-rom
    \\
;

pub fn run(io: Io, gpa: Allocator, args: *std.process.Args.Iterator) !void {
    var image_path: ?[]const u8 = null;
    var width: ?u32 = null;
    var height: ?u32 = null;
    var protocol: ?[]const u8 = null;
    var filter: zignal.Interpolation = .bilinear;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--width")) {
            if (args.next()) |w| {
                width = std.fmt.parseInt(u32, w, 10) catch {
                    std.log.err("Invalid width: {s}", .{w});
                    return error.InvalidArguments;
                };
            }
        } else if (std.mem.eql(u8, arg, "--height")) {
            if (args.next()) |h| {
                height = std.fmt.parseInt(u32, h, 10) catch {
                    std.log.err("Invalid height: {s}", .{h});
                    return error.InvalidArguments;
                };
            }
        } else if (std.mem.eql(u8, arg, "--protocol")) {
            protocol = args.next();
        } else if (std.mem.eql(u8, arg, "--filter")) {
            if (args.next()) |f| {
                const filter_map = std.StaticStringMap(zignal.Interpolation).initComptime(.{
                    .{ "nearest", .nearest_neighbor },
                    .{ "bilinear", .bilinear },
                    .{ "bicubic", .bicubic },
                    .{ "lanczos", .lanczos },
                    .{ "catmull-rom", .catmull_rom },
                });
                if (filter_map.get(f)) |f_enum| {
                    filter = f_enum;
                } else {
                    std.log.err("Unknown filter type: {s}", .{f});
                    return error.InvalidArguments;
                }
            }
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            image_path = arg;
        }
    }

    const path = image_path orelse {
        std.log.err("Missing image path for 'view' command", .{});
        return error.InvalidArguments;
    };

    // Load image as RGBA to support transparency
    var image = try zignal.Image(zignal.Rgba(u8)).load(io, gpa, path);
    defer image.deinit(gpa);

    var display_fmt: zignal.DisplayFormat = undefined;

    if (protocol) |p| {
        if (std.mem.eql(u8, p, "kitty")) {
            display_fmt = .{ .kitty = .{ .width = width, .height = height, .interpolation = filter } };
        } else if (std.mem.eql(u8, p, "sixel")) {
            display_fmt = .{ .sixel = .{ .palette = .{ .adaptive = .{ .max_colors = 256 } }, .dither = .auto, .width = width, .height = height, .interpolation = filter } };
        } else if (std.mem.eql(u8, p, "sgr")) {
            display_fmt = .{ .sgr = .{ .width = width, .height = height } };
        } else if (std.mem.eql(u8, p, "braille")) {
            display_fmt = .{ .braille = .{ .width = width, .height = height } };
        } else {
            // auto
            display_fmt = detectBestFormat(io, width, height, filter);
        }
    } else {
        display_fmt = detectBestFormat(io, width, height, filter);
    }

    var buffer: [65536]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);
    try stdout.interface.print("{f}\n", .{image.display(io, display_fmt)});
    try stdout.interface.flush();
}

fn detectBestFormat(io: Io, width: ?u32, height: ?u32, filter: zignal.Interpolation) zignal.DisplayFormat {
    if (terminal.isKittySupported(io) catch false) {
        return .{ .kitty = .{ .width = width, .height = height, .interpolation = filter } };
    }
    if (terminal.isSixelSupported(io) catch false) {
        return .{ .sixel = .{ .palette = .{ .adaptive = .{ .max_colors = 256 } }, .dither = .auto, .width = width, .height = height, .interpolation = filter } };
    }
    return .{ .sgr = .{ .width = width, .height = height } };
}
