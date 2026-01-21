const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const zignal = @import("zignal");

const info = @import("cli/info.zig");
const version = @import("cli/version.zig");
const view = @import("cli/view.zig");

const general_help =
    \\Usage: zignal <command> [options]
    \\
    \\Commands:
    \\  view     Display an image in the terminal
    \\  info     Display image information
    \\  version  Display version information
    \\  help     Display this help message
    \\
    \\Run 'zignal help <command>' for more information on a specific command.
    \\
;

pub fn main(init: std.process.Init) !void {
    var args = try init.minimal.args.iterateAllocator(init.gpa);
    defer args.deinit();
    _ = args.skip();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "view")) {
            view.run(init.io, init.gpa, &args) catch |err| {
                std.log.err("view command failed: {t}", .{err});
                return;
            };
            return;
        }
        if (std.mem.eql(u8, arg, "version")) {
            try version.run(init.io);
            return;
        }
        if (std.mem.eql(u8, arg, "info")) {
            if (args.next()) |image_path| {
                info.run(init.io, init.gpa, image_path) catch |err| {
                    std.log.err("failed to get info for '{s}': {t}", .{ image_path, err });
                    return;
                };
            } else {
                std.log.err("Missing image path for 'info' command", .{});
            }
            return;
        }
        if (std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try help(init.io, &args);
            return;
        }
    }
    try help(init.io, null);
}

fn help(io: Io, args: ?*std.process.Args.Iterator) !void {
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);

    if (args) |commands| {
        if (commands.next()) |subcmd| {
            if (std.mem.eql(u8, subcmd, "view")) {
                try stdout.interface.print("{s}", .{view.help_text});
            } else if (std.mem.eql(u8, subcmd, "info")) {
                try stdout.interface.print("{s}", .{info.help_text});
            } else if (std.mem.eql(u8, subcmd, "version")) {
                try stdout.interface.print("{s}", .{version.help_text});
            } else if (std.mem.eql(u8, subcmd, "help")) {
                try stdout.interface.print("{s}", .{general_help});
            } else {
                try stdout.interface.print("Unknown command: {s}\n\n{s}", .{ subcmd, general_help });
            }
            try stdout.interface.flush();
            return;
        }
    }
    try stdout.interface.print("{s}", .{general_help});
    try stdout.interface.flush();
}
