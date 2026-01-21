const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const zignal = @import("zignal");
const terminal = zignal.terminal;

pub const help_text =
    \\Usage: zignal display <image> [options]
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
            const w_arg = args.next() orelse {
                std.log.err("Missing value for --width", .{});
                return error.InvalidArguments;
            };
            width = std.fmt.parseInt(u32, w_arg, 10) catch {
                std.log.err("Invalid width: {s}", .{w_arg});
                return error.InvalidArguments;
            };
        } else if (std.mem.eql(u8, arg, "--height")) {
            const h_arg = args.next() orelse {
                std.log.err("Missing value for --height", .{});
                return error.InvalidArguments;
            };
            height = std.fmt.parseInt(u32, h_arg, 10) catch {
                std.log.err("Invalid height: {s}", .{h_arg});
                return error.InvalidArguments;
            };
        } else if (std.mem.eql(u8, arg, "--protocol")) {
            protocol = args.next() orelse {
                std.log.err("Missing value for --protocol", .{});
                return error.InvalidArguments;
            };
        } else if (std.mem.eql(u8, arg, "--filter")) {
            const f = args.next() orelse {
                std.log.err("Missing value for --filter", .{});
                return error.InvalidArguments;
            };
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
        } else if (!std.mem.startsWith(u8, arg, "-")) {
            image_path = arg;
        }
    }

    const path = image_path orelse {
        std.log.err("Missing image path for 'display' command", .{});
        return error.InvalidArguments;
    };

    var image: zignal.Image(zignal.Rgba(u8)) = try .load(io, gpa, path);
    defer image.deinit(gpa);

    var display_fmt: zignal.DisplayFormat = .{ .auto = .{
        .width = width,
        .height = height,
        .interpolation = filter,
    } };

    if (protocol) |p| {
        if (std.mem.eql(u8, p, "kitty")) {
            display_fmt = .{ .kitty = .{
                .width = width,
                .height = height,
                .interpolation = filter,
            } };
        } else if (std.mem.eql(u8, p, "sixel")) {
            display_fmt = .{ .sixel = .{
                .palette = .{ .adaptive = .{ .max_colors = 256 } },
                .dither = .auto,
                .width = width,
                .height = height,
                .interpolation = filter,
            } };
        } else if (std.mem.eql(u8, p, "sgr")) {
            display_fmt = .{ .sgr = .{ .width = width, .height = height } };
        } else if (std.mem.eql(u8, p, "braille")) {
            display_fmt = .{ .braille = .{ .width = width, .height = height } };
        } else if (std.mem.eql(u8, p, "auto")) {
            // Already set to default
        } else {
            std.log.err("Unknown protocol: {s}", .{p});
            return error.InvalidArguments;
        }
    }

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);
    try stdout.interface.print("{f}\n", .{image.display(io, display_fmt)});
    try stdout.interface.flush();
}
