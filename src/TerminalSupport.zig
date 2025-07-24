//! Terminal capability detection and support utilities
//!
//! Provides cross-platform terminal detection for graphics protocols
//! (sixel, kitty) and other terminal features.

const std = @import("std");
const builtin = @import("builtin");

// Fields (state for terminal detection)
stdin: std.fs.File,
stdout: std.fs.File,
stderr: std.fs.File,
original_state: TerminalState,

const TerminalSupport = @This();

// Buffer size for terminal responses
const response_buffer_size: usize = 256;

// Windows API declarations and constants (conditionally compiled)
const win_api = if (builtin.os.tag == .windows) struct {
    // Console mode constants
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const ENABLE_LINE_INPUT: u32 = 0x0002;
    const ENABLE_ECHO_INPUT: u32 = 0x0004;

    // Standard handle constants
    const STD_INPUT_HANDLE: i32 = -10;
    const STD_OUTPUT_HANDLE: i32 = -11;

    // API functions
    extern "kernel32" fn GetStdHandle(nStdHandle: i32) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetConsoleMode(hConsoleHandle: ?*anyopaque, lpMode: *u32) callconv(.c) i32;
    extern "kernel32" fn SetConsoleMode(hConsoleHandle: ?*anyopaque, dwMode: u32) callconv(.c) i32;
    extern "msvcrt" fn _kbhit() callconv(.c) c_int;
    extern "msvcrt" fn _getch() callconv(.c) c_int;
} else void;

/// Terminal state for restoration
const TerminalState = union(enum) {
    windows: struct {
        output_mode: u32,
        input_mode: u32,
    },
    posix: std.posix.termios,
};

/// Configuration options for terminal detection
pub const DetectionOptions = struct {
    /// Timeout for terminal responses in milliseconds
    timeout_ms: u64 = 100,
    /// Enable functional test (may cause visible output)
    enable_functional_test: bool = false,
};

/// Initialize terminal support for capability detection
pub fn init() !TerminalSupport {
    const stdin = std.fs.File.stdin();
    const stdout = std.fs.File.stdout();
    const stderr = std.fs.File.stderr();

    if (builtin.os.tag == .windows) {
        // Windows-specific initialization
        const stdin_handle = win_api.GetStdHandle(win_api.STD_INPUT_HANDLE);
        const stdout_handle = win_api.GetStdHandle(win_api.STD_OUTPUT_HANDLE);

        // Save original console modes
        var original_output_mode: u32 = 0;
        var original_input_mode: u32 = 0;

        if (win_api.GetConsoleMode(stdout_handle, &original_output_mode) == 0) {
            return error.ConsoleError;
        }
        if (win_api.GetConsoleMode(stdin_handle, &original_input_mode) == 0) {
            return error.ConsoleError;
        }

        // Enable Virtual Terminal Processing for ANSI sequences
        const new_output_mode = original_output_mode | win_api.ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (win_api.SetConsoleMode(stdout_handle, new_output_mode) == 0) {
            return error.ConsoleError;
        }

        // Set input mode for raw reading
        const raw_input_mode = original_input_mode & ~(win_api.ENABLE_LINE_INPUT | win_api.ENABLE_ECHO_INPUT);
        _ = win_api.SetConsoleMode(stdin_handle, raw_input_mode);

        return TerminalSupport{
            .stdin = stdin,
            .stdout = stdout,
            .stderr = stderr,
            .original_state = .{ .windows = .{
                .output_mode = original_output_mode,
                .input_mode = original_input_mode,
            } },
        };
    } else {
        // POSIX: Get current terminal settings
        const original = try std.posix.tcgetattr(stdin.handle);

        return TerminalSupport{
            .stdin = stdin,
            .stdout = stdout,
            .stderr = stderr,
            .original_state = .{ .posix = original },
        };
    }
}

/// Restore terminal to original state
pub fn deinit(self: *TerminalSupport) void {
    switch (self.original_state) {
        .windows => |win_state| {
            if (builtin.os.tag == .windows) {
                // Restore original console modes
                const stdin_handle = win_api.GetStdHandle(win_api.STD_INPUT_HANDLE);
                const stdout_handle = win_api.GetStdHandle(win_api.STD_OUTPUT_HANDLE);
                _ = win_api.SetConsoleMode(stdout_handle, win_state.output_mode);
                _ = win_api.SetConsoleMode(stdin_handle, win_state.input_mode);
            }
        },
        .posix => |termios| {
            if (builtin.os.tag != .windows) {
                // Restore original terminal settings
                std.posix.tcsetattr(self.stdin.handle, .FLUSH, termios) catch {};
            }
        },
    }
}

/// Check if stdout is connected to a TTY
pub fn isStdoutTty() bool {
    return std.fs.File.stdout().isTty();
}

/// Detect if the terminal supports sixel graphics protocol
pub fn detectSixelSupport(self: *TerminalSupport) !bool {
    // Try DECRQSS - Request Status String (no visible output)
    if (try self.checkSixelSupport(.param_query)) return true;

    // Try Device Attributes (no visible output)
    if (try self.checkSixelSupport(.device_attributes)) return true;

    return false;
}

/// Detect if the terminal supports Kitty graphics protocol
pub fn detectKittySupport(self: *TerminalSupport) !bool {
    var response_buf: [response_buffer_size]u8 = undefined;

    // Send Kitty graphics query followed by device attributes
    // This allows us to detect Kitty support by checking which response we get
    const query_seq = "\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\\x1b[c";

    const response = try self.query(query_seq, &response_buf, 100);

    // If we get a graphics query response, Kitty is supported
    // The response will contain "\x1b_G" if Kitty processed the graphics query
    return std.mem.indexOf(u8, response, "\x1b_G") != null;
}

// Private implementation methods

fn enterRawMode(self: *TerminalSupport) !void {
    switch (self.original_state) {
        .windows => {
            // Already in raw mode from init
        },
        .posix => |original| {
            if (builtin.os.tag != .windows) {
                var raw = original;

                // Disable canonical mode and echo
                raw.lflag.ICANON = false;
                raw.lflag.ECHO = false;

                // Set minimum characters and timeout
                raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
                raw.cc[@intFromEnum(std.posix.V.TIME)] = 1; // 0.1 second timeout

                try std.posix.tcsetattr(self.stdin.handle, .FLUSH, raw);
            }
        },
    }
}

fn readWithTimeout(self: *TerminalSupport, buffer: []u8, timeout_ms: u64) !usize {
    if (builtin.os.tag == .windows) {
        const start_time = std.time.milliTimestamp();
        var total_read: usize = 0;

        while (std.time.milliTimestamp() - start_time < timeout_ms) {
            // Check if console has input available
            if (win_api._kbhit() != 0) {
                // Read one character
                const ch = win_api._getch();
                if (ch >= 0 and ch <= 255) {
                    buffer[total_read] = @intCast(ch);
                    total_read += 1;

                    if (total_read >= buffer.len) break;

                    // Check for response terminators
                    const char: u8 = @intCast(ch);
                    if ((char == 'c' or char == 'R' or char == '\\' or char == ';') and total_read > 3) {
                        break;
                    }
                }
            }

            // Small delay to prevent busy waiting
            std.Thread.sleep(1_000_000); // 1ms
        }

        return total_read;
    } else {
        // POSIX: Use the existing termios timeout mechanism
        return try self.stdin.read(buffer);
    }
}

fn query(self: *TerminalSupport, sequence: []const u8, buffer: []u8, timeout_ms: u64) ![]const u8 {
    // Enter raw mode
    try self.enterRawMode();
    defer {
        // Restore terminal state
        switch (self.original_state) {
            .windows => {},
            .posix => |termios| {
                if (builtin.os.tag != .windows) {
                    std.posix.tcsetattr(self.stdin.handle, .FLUSH, termios) catch {};
                }
            },
        }
    }

    // Clear any pending input
    if (builtin.os.tag == .windows) {
        // Consume any pending input
        while (win_api._kbhit() != 0) {
            _ = win_api._getch();
        }
    } else {
        var discard_buf: [response_buffer_size]u8 = undefined;
        _ = self.stdin.read(&discard_buf) catch 0;
    }

    // Send query sequence
    _ = try self.stdout.write(sequence);

    // Read response with timeout
    const n = try self.readWithTimeout(buffer, timeout_ms);

    if (n == 0) return error.NoResponse;

    return buffer[0..n];
}

/// Check sixel support using a specific query method
fn checkSixelSupport(self: *TerminalSupport, method: enum { param_query, device_attributes }) !bool {
    var response_buf: [response_buffer_size]u8 = undefined;

    switch (method) {
        .param_query => {
            // Query sixel graphics parameter
            const response = self.query("\x1b[?2;1;0S", &response_buf, 100) catch {
                return false;
            };

            // Look for positive response indicating sixel support
            // Expected format: ESC P 1 $ r <params> ESC \
            return response.len >= 4 and std.mem.indexOf(u8, response, "\x1bP") != null;
        },
        .device_attributes => {
            // Send Primary Device Attributes query
            const response = self.query("\x1b[c", &response_buf, 100) catch {
                return false;
            };

            // Parse response looking for attribute 4 (sixel graphics)
            // Format: ESC [ ? <attributes> c
            if (response.len >= 4 and response[0] == '\x1b' and response[1] == '[' and response[2] == '?') {
                // Look for '4' in the attribute list
                var i: usize = 3;
                while (i < response.len and response[i] != 'c') : (i += 1) {
                    if (response[i] == '4') {
                        // Check it's a standalone 4, not part of another number
                        const prev_is_separator = (i == 3 or response[i - 1] == ';');
                        const next_is_separator = (i + 1 >= response.len or response[i + 1] == ';' or response[i + 1] == 'c');
                        if (prev_is_separator and next_is_separator) {
                            return true;
                        }
                    }
                }
            }
            return false;
        },
    }
}
