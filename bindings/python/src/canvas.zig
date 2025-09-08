const std = @import("std");

const zignal = @import("zignal");
pub const Canvas = zignal.Canvas;
const DrawMode = zignal.DrawMode;
const Rgba = zignal.Rgba;
const Rgb = zignal.Rgb;
const BitmapFont = zignal.BitmapFont;

const py_utils = @import("py_utils.zig");
const allocator = py_utils.allocator;
pub const registerType = py_utils.registerType;
const c = py_utils.c;
const color_utils = @import("color_utils.zig");
const stub_metadata = @import("stub_metadata.zig");
const enum_utils = @import("enum_utils.zig");
const PyImageMod = @import("PyImage.zig");
const PyImage = PyImageMod.PyImage;
const Kind = enum(u8) { gray, rgb, rgba };

// Documentation for the DrawMode enum (used at runtime and for stub generation)
pub const draw_mode_doc =
    \\Rendering quality mode for drawing operations.
    \\
    \\## Attributes
    \\- `FAST` (int): Fast rendering without antialiasing (value: 0)
    \\- `SOFT` (int): High-quality rendering with antialiasing (value: 1)
    \\
    \\## Notes
    \\- FAST mode provides pixel-perfect rendering with sharp edges
    \\- SOFT mode provides smooth, antialiased edges for better visual quality
    \\- Default mode is FAST for performance
;

// Per-value documentation for stub generation
pub const draw_mode_values = [_]stub_metadata.EnumValueDoc{
    .{ .name = "FAST", .doc = "Fast rendering with hard edges" },
    .{ .name = "SOFT", .doc = "Antialiased rendering with smooth edges" },
};

// Static default font with all characters, initialized once
var font8x8: ?BitmapFont = null;

fn getFont8x8() !BitmapFont {
    if (font8x8 == null) {
        font8x8 = try zignal.font.font8x8.create(allocator, .all);
    }
    return font8x8.?;
}

pub const CanvasObject = extern struct {
    ob_base: c.PyObject,
    // Keep reference to parent Image to prevent garbage collection
    image_ref: ?*c.PyObject,
    // Variant canvas pointers and kind
    canvas_gray: ?*Canvas(u8),
    canvas_rgb: ?*Canvas(Rgb),
    canvas_rgba: ?*Canvas(Rgba),
    kind: Kind,
};

fn canvas_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*CanvasObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.image_ref = null;
        obj.canvas_gray = null;
        obj.canvas_rgb = null;
        obj.canvas_rgba = null;
        obj.kind = .rgba;
    }
    return @as(?*c.PyObject, @ptrCast(self));
}

fn canvas_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = kwds;
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments - expect an Image object
    var image_obj: ?*c.PyObject = undefined;
    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &image_obj) == 0) {
        return -1;
    }

    // Import the image module to get the ImageType
    const image_module = @import("image.zig");

    // Check if it's an Image instance
    if (c.PyObject_IsInstance(image_obj, @ptrCast(&image_module.ImageType)) <= 0) {
        c.PyErr_SetString(c.PyExc_TypeError, "Argument must be an Image instance");
        return -1;
    }

    const image = @as(*image_module.ImageObject, @ptrCast(image_obj.?));

    // Keep reference to parent Image to prevent garbage collection
    c.Py_INCREF(image_obj.?);
    self.image_ref = image_obj;

    // Initialize based on image format
    if (image.py_image) |pimg| {
        switch (pimg.data) {
            .grayscale => |imgv| {
                const cptr = allocator.create(Canvas(u8)) catch {
                    c.Py_DECREF(image_obj.?);
                    c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate Canvas");
                    return -1;
                };
                cptr.* = Canvas(u8).init(allocator, imgv);
                self.canvas_gray = cptr;
                self.canvas_rgb = null;
                self.canvas_rgba = null;
                self.kind = .gray;
            },
            .rgb => |imgv| {
                const cptr = allocator.create(Canvas(Rgb)) catch {
                    c.Py_DECREF(image_obj.?);
                    c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate Canvas");
                    return -1;
                };
                cptr.* = Canvas(Rgb).init(allocator, imgv);
                self.canvas_gray = null;
                self.canvas_rgb = cptr;
                self.canvas_rgba = null;
                self.kind = .rgb;
            },
            .rgba => |imgv| {
                const cptr = allocator.create(Canvas(Rgba)) catch {
                    c.Py_DECREF(image_obj.?);
                    c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate Canvas");
                    return -1;
                };
                cptr.* = Canvas(Rgba).init(allocator, imgv);
                self.canvas_gray = null;
                self.canvas_rgb = null;
                self.canvas_rgba = cptr;
                self.kind = .rgba;
            },
        }
    } else {
        c.Py_DECREF(image_obj.?);
        c.PyErr_SetString(c.PyExc_ValueError, "Image not initialized");
        return -1;
    }

    return 0;
}

fn canvas_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Free the Canvas struct
    switch (self.kind) {
        .gray => if (self.canvas_gray) |ptr| allocator.destroy(ptr),
        .rgb => if (self.canvas_rgb) |ptr| allocator.destroy(ptr),
        .rgba => if (self.canvas_rgba) |ptr| allocator.destroy(ptr),
    }

    // Release reference to parent Image
    if (self.image_ref) |ref| {
        c.Py_XDECREF(ref);
    }

    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn canvas_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    if (self.kind == .gray and self.canvas_gray != null) {
        var buffer: [64]u8 = undefined;
        const formatted = std.fmt.bufPrintZ(&buffer, "Canvas({d}x{d}, fmt=Grayscale)", .{ self.canvas_gray.?.image.rows, self.canvas_gray.?.image.cols }) catch return null;
        return c.PyUnicode_FromString(formatted.ptr);
    } else if (self.kind == .rgb and self.canvas_rgb != null) {
        var buffer: [64]u8 = undefined;
        const formatted = std.fmt.bufPrintZ(&buffer, "Canvas({d}x{d}, fmt=Rgb)", .{ self.canvas_rgb.?.image.rows, self.canvas_rgb.?.image.cols }) catch return null;
        return c.PyUnicode_FromString(formatted.ptr);
    } else if (self.kind == .rgba and self.canvas_rgba != null) {
        var buffer: [64]u8 = undefined;
        const formatted = std.fmt.bufPrintZ(&buffer, "Canvas({d}x{d}, fmt=Rgba)", .{ self.canvas_rgba.?.image.rows, self.canvas_rgba.?.image.cols }) catch return null;
        return c.PyUnicode_FromString(formatted.ptr);
    } else {
        return c.PyUnicode_FromString("Canvas(uninitialized)");
    }
}

// Common parsing structure for draw methods
const DrawArgs = struct {
    kind: Kind,
    canvas_gray: ?*Canvas(u8) = null,
    canvas_rgb: ?*Canvas(Rgb) = null,
    canvas_rgba: ?*Canvas(Rgba) = null,
    color_gray: u8 = 0,
    color_rgb: Rgb = .{ .r = 0, .g = 0, .b = 0 },
    color_rgba: Rgba = .{ .r = 0, .g = 0, .b = 0, .a = 255 },
    width: u32,
    mode: DrawMode,
};

// Common parsing structure for fill methods
const FillArgs = struct {
    kind: Kind,
    canvas_gray: ?*Canvas(u8) = null,
    canvas_rgb: ?*Canvas(Rgb) = null,
    canvas_rgba: ?*Canvas(Rgba) = null,
    color_gray: u8 = 0,
    color_rgb: Rgb = .{ .r = 0, .g = 0, .b = 0 },
    color_rgba: Rgba = .{ .r = 0, .g = 0, .b = 0, .a = 255 },
    mode: DrawMode,
};

// Helper to parse common draw arguments
fn parseDrawArgs(self: *CanvasObject, color_obj: ?*c.PyObject, width: c_long, mode: c_long) !DrawArgs {
    const rgba = try color_utils.parseColorTo(Rgba, @ptrCast(color_obj));
    var args = DrawArgs{
        .kind = self.kind,
        .width = try py_utils.validateNonNegative(u32, width, "Width"),
        .mode = if (try py_utils.validateRange(u32, mode, 0, 1, "Mode") == 0) .fast else .soft,
    };
    switch (self.kind) {
        .gray => {
            if (self.canvas_gray) |cptr| {
                args.canvas_gray = cptr;
                args.color_gray = rgba.toGray();
            } else return error.CanvasNotInitialized;
        },
        .rgb => {
            if (self.canvas_rgb) |cptr| {
                args.canvas_rgb = cptr;
                args.color_rgb = rgba.toRgb();
            } else return error.CanvasNotInitialized;
        },
        .rgba => {
            if (self.canvas_rgba) |cptr| {
                args.canvas_rgba = cptr;
                args.color_rgba = rgba;
            } else return error.CanvasNotInitialized;
        },
    }
    return args;
}

// Helper to parse common fill arguments
fn parseFillArgs(self: *CanvasObject, color_obj: ?*c.PyObject, mode: c_long) !FillArgs {
    const rgba = try color_utils.parseColorTo(Rgba, @ptrCast(color_obj));
    var args = FillArgs{
        .kind = self.kind,
        .mode = if (try py_utils.validateRange(u32, mode, 0, 1, "Mode") == 0) .fast else .soft,
    };
    switch (self.kind) {
        .gray => {
            if (self.canvas_gray) |cptr| {
                args.canvas_gray = cptr;
                args.color_gray = rgba.toGray();
            } else return error.CanvasNotInitialized;
        },
        .rgb => {
            if (self.canvas_rgb) |cptr| {
                args.canvas_rgb = cptr;
                args.color_rgb = rgba.toRgb();
            } else return error.CanvasNotInitialized;
        },
        .rgba => {
            if (self.canvas_rgba) |cptr| {
                args.canvas_rgba = cptr;
                args.color_rgba = rgba;
            } else return error.CanvasNotInitialized;
        },
    }
    return args;
}

// Special case: fill method
const canvas_fill_doc =
    \\Fill the entire canvas with a color.
    \\
    \\## Parameters
    \\- `color` (int, tuple or color object): Color to fill the canvas with. Can be:
    \\  - Integer: grayscale value 0-255 (0=black, 255=white)
    \\  - RGB tuple: `(r, g, b)` with values 0-255
    \\  - RGBA tuple: `(r, g, b, a)` with values 0-255
    \\  - Any color object: `Rgb`, `Rgba`, `Hsl`, `Hsv`, `Lab`, `Lch`, `Lms`, `Oklab`, `Oklch`, `Xyb`, `Xyz`, `Ycbcr`
    \\
    \\## Examples
    \\```python
    \\img = Image.load("photo.png")
    \\canvas = img.canvas()
    \\canvas.fill(128)  # Fill with gray
    \\canvas.fill((255, 0, 0))  # Fill with red
    \\canvas.fill(Rgb(0, 255, 0))  # Fill with green using Rgb object
    \\```
;

fn canvas_fill(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse color argument
    var color_obj: ?*c.PyObject = undefined;
    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &color_obj) == 0) {
        return null;
    }

    // Parse and fill according to canvas kind
    const rgba = color_utils.parseColorTo(Rgba, @ptrCast(color_obj)) catch return null;
    switch (self.kind) {
        .gray => if (self.canvas_gray) |cptr| cptr.fill(rgba.toGray()) else return null,
        .rgb => if (self.canvas_rgb) |cptr| cptr.fill(rgba.toRgb()) else return null,
        .rgba => if (self.canvas_rgba) |cptr| cptr.fill(rgba) else return null,
    }

    return py_utils.getPyNone();
}

// Macro-like function to generate draw methods with width parameter
fn makeDrawMethodWithWidth(
    comptime name: []const u8,
    comptime canvas_method: []const u8,
    comptime param_count: usize,
    comptime param_names: [param_count][]const u8,
    comptime param_types: [param_count][]const u8,
    comptime parse_format: []const u8,
    comptime doc: []const u8,
) type {
    _ = canvas_method;
    _ = param_types;
    return struct {
        fn method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
            const self = @as(*CanvasObject, @ptrCast(self_obj.?));

            // Build kwlist at comptime
            const kwlist = comptime blk: {
                var list: [param_count + 4][*c]const u8 = undefined;
                for (param_names, 0..) |pname, i| {
                    list[i] = pname.ptr;
                }
                list[param_count] = "color";
                list[param_count + 1] = "width";
                list[param_count + 2] = "mode";
                list[param_count + 3] = null;
                break :blk list;
            };

            // Parse arguments - use generic vars array
            var param_objs: [param_count]?*c.PyObject = undefined;
            var color_obj: ?*c.PyObject = undefined;
            var width: c_long = 1;
            var mode: c_long = 0;

            const format = std.fmt.comptimePrint(parse_format ++ "O|ll", .{});

            // Build argument list for parsing
            var parse_args: [param_count + 3]?*anyopaque = undefined;
            inline for (0..param_count) |i| {
                parse_args[i] = @ptrCast(&param_objs[i]);
            }
            parse_args[param_count] = @ptrCast(&color_obj);
            parse_args[param_count + 1] = @ptrCast(&width);
            parse_args[param_count + 2] = @ptrCast(&mode);

            // Call PyArg_ParseTupleAndKeywords with variable arguments
            switch (param_count) {
                1 => {
                    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), parse_args[0], parse_args[1], parse_args[2], parse_args[3]) == 0) {
                        return null;
                    }
                },
                2 => {
                    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), parse_args[0], parse_args[1], parse_args[2], parse_args[3], parse_args[4]) == 0) {
                        return null;
                    }
                },
                3 => {
                    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), parse_args[0], parse_args[1], parse_args[2], parse_args[3], parse_args[4], parse_args[5]) == 0) {
                        return null;
                    }
                },
                else => @compileError("Unsupported param count"),
            }

            // Parse common arguments
            const common = parseDrawArgs(self, color_obj, width, mode) catch return null;

            // Call the appropriate method based on parameter types
            if (comptime std.mem.eql(u8, name, "draw_line")) {
                const p1 = py_utils.parsePointTuple(f32, @ptrCast(param_objs[0])) catch return null;
                const p2 = py_utils.parsePointTuple(f32, @ptrCast(param_objs[1])) catch return null;
                switch (common.kind) {
                    .gray => common.canvas_gray.?.drawLine(p1, p2, common.color_gray, common.width, common.mode),
                    .rgb => common.canvas_rgb.?.drawLine(p1, p2, common.color_rgb, common.width, common.mode),
                    .rgba => common.canvas_rgba.?.drawLine(p1, p2, common.color_rgba, common.width, common.mode),
                }
            } else if (comptime std.mem.eql(u8, name, "draw_rectangle")) {
                const rect = py_utils.parseRectangle(f32, @ptrCast(param_objs[0])) catch return null;
                switch (common.kind) {
                    .gray => common.canvas_gray.?.drawRectangle(rect, common.color_gray, common.width, common.mode),
                    .rgb => common.canvas_rgb.?.drawRectangle(rect, common.color_rgb, common.width, common.mode),
                    .rgba => common.canvas_rgba.?.drawRectangle(rect, common.color_rgba, common.width, common.mode),
                }
            } else if (comptime std.mem.eql(u8, name, "draw_polygon")) {
                const points = py_utils.parsePointList(f32, @ptrCast(param_objs[0])) catch return null;
                defer allocator.free(points);
                switch (common.kind) {
                    .gray => common.canvas_gray.?.drawPolygon(points, common.color_gray, common.width, common.mode),
                    .rgb => common.canvas_rgb.?.drawPolygon(points, common.color_rgb, common.width, common.mode),
                    .rgba => common.canvas_rgba.?.drawPolygon(points, common.color_rgba, common.width, common.mode),
                }
            } else if (comptime std.mem.eql(u8, name, "draw_circle")) {
                const center = py_utils.parsePointTuple(f32, @ptrCast(param_objs[0])) catch return null;
                const radius = py_utils.validateNonNegative(f32, @as(*f64, @ptrCast(@alignCast(parse_args[1]))).*, "Radius") catch return null;
                switch (common.kind) {
                    .gray => common.canvas_gray.?.drawCircle(center, radius, common.color_gray, common.width, common.mode),
                    .rgb => common.canvas_rgb.?.drawCircle(center, radius, common.color_rgb, common.width, common.mode),
                    .rgba => common.canvas_rgba.?.drawCircle(center, radius, common.color_rgba, common.width, common.mode),
                }
            }

            return py_utils.getPyNone();
        }

        const doc_string = doc;
    };
}

// Macro-like function to generate fill methods
fn makeFillMethod(
    comptime name: []const u8,
    comptime canvas_method: []const u8,
    comptime param_count: usize,
    comptime param_names: [param_count][]const u8,
    comptime param_types: [param_count][]const u8,
    comptime parse_format: []const u8,
    comptime doc: []const u8,
    comptime has_error_handling: bool,
) type {
    _ = canvas_method;
    _ = param_types;
    return struct {
        fn method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
            const self = @as(*CanvasObject, @ptrCast(self_obj.?));

            // Build kwlist at comptime
            const kwlist = comptime blk: {
                var list: [param_count + 3][*c]const u8 = undefined;
                for (param_names, 0..) |pname, i| {
                    list[i] = pname.ptr;
                }
                list[param_count] = "color";
                list[param_count + 1] = "mode";
                list[param_count + 2] = null;
                break :blk list;
            };

            // Parse arguments
            var param_objs: [param_count]?*c.PyObject = undefined;
            var color_obj: ?*c.PyObject = undefined;
            var mode: c_long = 0;

            const format = std.fmt.comptimePrint(parse_format ++ "O|l", .{});

            // Build argument list for parsing
            var parse_args: [param_count + 2]?*anyopaque = undefined;
            inline for (0..param_count) |i| {
                parse_args[i] = @ptrCast(&param_objs[i]);
            }
            parse_args[param_count] = @ptrCast(&color_obj);
            parse_args[param_count + 1] = @ptrCast(&mode);

            // Call PyArg_ParseTupleAndKeywords with variable arguments
            switch (param_count) {
                1 => {
                    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), parse_args[0], parse_args[1], parse_args[2]) == 0) {
                        return null;
                    }
                },
                2 => {
                    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), parse_args[0], parse_args[1], parse_args[2], parse_args[3]) == 0) {
                        return null;
                    }
                },
                else => @compileError("Unsupported param count"),
            }

            // Parse common arguments
            const common = parseFillArgs(self, color_obj, mode) catch return null;

            // Call the appropriate method based on parameter types
            if (comptime std.mem.eql(u8, name, "fill_rectangle")) {
                const rect = py_utils.parseRectangle(f32, @ptrCast(param_objs[0])) catch return null;
                switch (common.kind) {
                    .gray => common.canvas_gray.?.fillRectangle(rect, common.color_gray, common.mode),
                    .rgb => common.canvas_rgb.?.fillRectangle(rect, common.color_rgb, common.mode),
                    .rgba => common.canvas_rgba.?.fillRectangle(rect, common.color_rgba, common.mode),
                }
            } else if (comptime std.mem.eql(u8, name, "fill_polygon")) {
                const points = py_utils.parsePointList(f32, @ptrCast(param_objs[0])) catch return null;
                defer allocator.free(points);
                if (has_error_handling) {
                    switch (common.kind) {
                        .gray => common.canvas_gray.?.fillPolygon(points, common.color_gray, common.mode) catch {
                            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill polygon");
                            return null;
                        },
                        .rgb => common.canvas_rgb.?.fillPolygon(points, common.color_rgb, common.mode) catch {
                            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill polygon");
                            return null;
                        },
                        .rgba => common.canvas_rgba.?.fillPolygon(points, common.color_rgba, common.mode) catch {
                            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill polygon");
                            return null;
                        },
                    }
                } else {
                    switch (common.kind) {
                        .gray => common.canvas_gray.?.fillPolygon(points, common.color_gray, common.mode) catch unreachable,
                        .rgb => common.canvas_rgb.?.fillPolygon(points, common.color_rgb, common.mode) catch unreachable,
                        .rgba => common.canvas_rgba.?.fillPolygon(points, common.color_rgba, common.mode) catch unreachable,
                    }
                }
            } else if (comptime std.mem.eql(u8, name, "fill_circle")) {
                const center = py_utils.parsePointTuple(f32, @ptrCast(param_objs[0])) catch return null;
                const radius = py_utils.validateNonNegative(f32, @as(*f64, @ptrCast(@alignCast(parse_args[1]))).*, "Radius") catch return null;
                switch (common.kind) {
                    .gray => common.canvas_gray.?.fillCircle(center, radius, common.color_gray, common.mode),
                    .rgb => common.canvas_rgb.?.fillCircle(center, radius, common.color_rgb, common.mode),
                    .rgba => common.canvas_rgba.?.fillCircle(center, radius, common.color_rgba, common.mode),
                }
            }

            return py_utils.getPyNone();
        }

        const doc_string = doc;
    };
}

// Generate method instances
const DrawLine = makeDrawMethodWithWidth(
    "draw_line",
    "drawLine",
    2,
    [_][]const u8{ "p1", "p2" },
    [_][]const u8{ "tuple", "tuple" },
    "OO",
    \\Draw a line between two points.
    \\
    \\## Parameters
    \\- `p1` (tuple[float, float]): Starting point coordinates (x, y)
    \\- `p2` (tuple[float, float]): Ending point coordinates (x, y)
    \\- `color` (int, tuple or color object): Color of the line.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    ,
);

const DrawRectangle = makeDrawMethodWithWidth(
    "draw_rectangle",
    "drawRectangle",
    1,
    [_][]const u8{"rect"},
    [_][]const u8{"Rectangle"},
    "O",
    \\Draw a rectangle outline.
    \\
    \\## Parameters
    \\- `rect` (Rectangle): Rectangle object defining the bounds
    \\- `color` (int, tuple or color object): Color of the rectangle.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    ,
);

const DrawPolygon = makeDrawMethodWithWidth(
    "draw_polygon",
    "drawPolygon",
    1,
    [_][]const u8{"points"},
    [_][]const u8{"list"},
    "O",
    \\Draw a polygon outline.
    \\
    \\## Parameters
    \\- `points` (list[tuple[float, float]]): List of (x, y) coordinates forming the polygon
    \\- `color` (int, tuple or color object): Color of the polygon.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    ,
);

const DrawCircle = makeDrawMethodWithWidth(
    "draw_circle",
    "drawCircle",
    2,
    [_][]const u8{ "center", "radius" },
    [_][]const u8{ "tuple", "float" },
    "Od",
    \\Draw a circle outline.
    \\
    \\## Parameters
    \\- `center` (tuple[float, float]): Center coordinates (x, y)
    \\- `radius` (float): Circle radius
    \\- `color` (int, tuple or color object): Color of the circle.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    ,
);

const FillRectangle = makeFillMethod(
    "fill_rectangle",
    "fillRectangle",
    1,
    [_][]const u8{"rect"},
    [_][]const u8{"Rectangle"},
    "O",
    \\Fill a rectangle area.
    \\
    \\## Parameters
    \\- `rect` (Rectangle): Rectangle object defining the bounds
    \\- `color` (int, tuple or color object): Fill color.
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
,
    false,
);

const FillPolygon = makeFillMethod(
    "fill_polygon",
    "fillPolygon",
    1,
    [_][]const u8{"points"},
    [_][]const u8{"list"},
    "O",
    \\Fill a polygon area.
    \\
    \\## Parameters
    \\- `points` (list[tuple[float, float]]): List of (x, y) coordinates forming the polygon
    \\- `color` (int, tuple or color object): Fill color.
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
,
    true,
);

const FillCircle = makeFillMethod(
    "fill_circle",
    "fillCircle",
    2,
    [_][]const u8{ "center", "radius" },
    [_][]const u8{ "tuple", "float" },
    "Od",
    \\Fill a circle area.
    \\
    \\## Parameters
    \\- `center` (tuple[float, float]): Center coordinates (x, y)
    \\- `radius` (float): Circle radius
    \\- `color` (int, tuple or color object): Fill color.
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
,
    false,
);

// Special methods that need custom handling
fn canvas_draw_quadratic_bezier(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var p0_obj: ?*c.PyObject = undefined;
    var p1_obj: ?*c.PyObject = undefined;
    var p2_obj: ?*c.PyObject = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var width: c_long = 1;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "p0", "p1", "p2", "color", "width", "mode" });
    const format = std.fmt.comptimePrint("OOOO|ll", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &p0_obj, &p1_obj, &p2_obj, &color_obj, &width, &mode) == 0) {
        return null;
    }

    const common = parseDrawArgs(self, color_obj, width, mode) catch return null;

    // Convert arguments
    const p0 = py_utils.parsePointTuple(f32, @ptrCast(p0_obj)) catch return null;
    const p1 = py_utils.parsePointTuple(f32, @ptrCast(p1_obj)) catch return null;
    const p2 = py_utils.parsePointTuple(f32, @ptrCast(p2_obj)) catch return null;

    switch (common.kind) {
        .gray => common.canvas_gray.?.drawQuadraticBezier(p0, p1, p2, common.color_gray, common.width, common.mode),
        .rgb => common.canvas_rgb.?.drawQuadraticBezier(p0, p1, p2, common.color_rgb, common.width, common.mode),
        .rgba => common.canvas_rgba.?.drawQuadraticBezier(p0, p1, p2, common.color_rgba, common.width, common.mode),
    }

    return py_utils.getPyNone();
}

fn canvas_draw_cubic_bezier(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var p0_obj: ?*c.PyObject = undefined;
    var p1_obj: ?*c.PyObject = undefined;
    var p2_obj: ?*c.PyObject = undefined;
    var p3_obj: ?*c.PyObject = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var width: c_long = 1;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "p0", "p1", "p2", "p3", "color", "width", "mode" });
    const format = std.fmt.comptimePrint("OOOOO|ll", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &p0_obj, &p1_obj, &p2_obj, &p3_obj, &color_obj, &width, &mode) == 0) {
        return null;
    }

    const common = parseDrawArgs(self, color_obj, width, mode) catch return null;

    // Convert arguments
    const p0 = py_utils.parsePointTuple(f32, @ptrCast(p0_obj)) catch return null;
    const p1 = py_utils.parsePointTuple(f32, @ptrCast(p1_obj)) catch return null;
    const p2 = py_utils.parsePointTuple(f32, @ptrCast(p2_obj)) catch return null;
    const p3 = py_utils.parsePointTuple(f32, @ptrCast(p3_obj)) catch return null;

    switch (common.kind) {
        .gray => common.canvas_gray.?.drawCubicBezier(p0, p1, p2, p3, common.color_gray, common.width, common.mode),
        .rgb => common.canvas_rgb.?.drawCubicBezier(p0, p1, p2, p3, common.color_rgb, common.width, common.mode),
        .rgba => common.canvas_rgba.?.drawCubicBezier(p0, p1, p2, p3, common.color_rgba, common.width, common.mode),
    }

    return py_utils.getPyNone();
}

fn canvas_draw_spline_polygon(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var points_obj: ?*c.PyObject = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var width: c_long = 1;
    var tension: f64 = 0.5;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "points", "color", "width", "tension", "mode" });
    const format = std.fmt.comptimePrint("OO|ldl", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &points_obj, &color_obj, &width, &tension, &mode) == 0) {
        return null;
    }

    const common = parseDrawArgs(self, color_obj, width, mode) catch return null;
    const tension_val = py_utils.validateRange(f32, tension, 0.0, 1.0, "Tension") catch return null;

    // Parse points
    const points = py_utils.parsePointList(f32, @ptrCast(points_obj)) catch return null;
    defer allocator.free(points);

    switch (common.kind) {
        .gray => common.canvas_gray.?.drawSplinePolygon(points, common.color_gray, common.width, tension_val, common.mode),
        .rgb => common.canvas_rgb.?.drawSplinePolygon(points, common.color_rgb, common.width, tension_val, common.mode),
        .rgba => common.canvas_rgba.?.drawSplinePolygon(points, common.color_rgba, common.width, tension_val, common.mode),
    }

    return py_utils.getPyNone();
}

fn canvas_fill_spline_polygon(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var points_obj: ?*c.PyObject = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var tension: f64 = 0.5;
    var mode: c_long = 0;

    const kwlist = [_][*c]const u8{ "points", "color", "tension", "mode", null };
    const format = std.fmt.comptimePrint("OO|dl", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kwlist)), &points_obj, &color_obj, &tension, &mode) == 0) {
        return null;
    }

    const common = parseFillArgs(self, color_obj, mode) catch return null;
    const tension_val = py_utils.validateRange(f32, tension, 0.0, 1.0, "Tension") catch return null;

    // Parse points
    const points = py_utils.parsePointList(f32, @ptrCast(points_obj)) catch return null;
    defer allocator.free(points);

    switch (common.kind) {
        .gray => common.canvas_gray.?.fillSplinePolygon(points, common.color_gray, tension_val, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill spline polygon");
            return null;
        },
        .rgb => common.canvas_rgb.?.fillSplinePolygon(points, common.color_rgb, tension_val, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill spline polygon");
            return null;
        },
        .rgba => common.canvas_rgba.?.fillSplinePolygon(points, common.color_rgba, tension_val, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill spline polygon");
            return null;
        },
    }

    return py_utils.getPyNone();
}

fn canvas_draw_arc(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var center_obj: ?*c.PyObject = undefined;
    var radius: f64 = undefined;
    var start_angle: f64 = undefined;
    var end_angle: f64 = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var width: c_long = 1;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "center", "radius", "start_angle", "end_angle", "color", "width", "mode" });
    const format = std.fmt.comptimePrint("OdddO|ll", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &center_obj, &radius, &start_angle, &end_angle, &color_obj, &width, &mode) == 0) {
        return null;
    }

    const common = parseDrawArgs(self, color_obj, width, mode) catch return null;

    // Convert arguments
    const center = py_utils.parsePointTuple(f32, @ptrCast(center_obj)) catch return null;
    const radius_val = @as(f32, @floatCast(radius));
    const start_angle_val = @as(f32, @floatCast(start_angle));
    const end_angle_val = @as(f32, @floatCast(end_angle));

    switch (common.kind) {
        .gray => common.canvas_gray.?.drawArc(center, radius_val, start_angle_val, end_angle_val, common.color_gray, common.width, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to draw arc");
            return null;
        },
        .rgb => common.canvas_rgb.?.drawArc(center, radius_val, start_angle_val, end_angle_val, common.color_rgb, common.width, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to draw arc");
            return null;
        },
        .rgba => common.canvas_rgba.?.drawArc(center, radius_val, start_angle_val, end_angle_val, common.color_rgba, common.width, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to draw arc");
            return null;
        },
    }

    return py_utils.getPyNone();
}

fn canvas_fill_arc(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var center_obj: ?*c.PyObject = undefined;
    var radius: f64 = undefined;
    var start_angle: f64 = undefined;
    var end_angle: f64 = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "center", "radius", "start_angle", "end_angle", "color", "mode" });
    const format = std.fmt.comptimePrint("OdddO|l", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &center_obj, &radius, &start_angle, &end_angle, &color_obj, &mode) == 0) {
        return null;
    }

    const common = parseFillArgs(self, color_obj, mode) catch return null;

    // Convert arguments
    const center = py_utils.parsePointTuple(f32, @ptrCast(center_obj)) catch return null;
    // Allow negative radius - the Zig library will handle it gracefully (no-op)
    const radius_val = @as(f32, @floatCast(radius));
    const start_angle_val = @as(f32, @floatCast(start_angle));
    const end_angle_val = @as(f32, @floatCast(end_angle));

    switch (common.kind) {
        .gray => common.canvas_gray.?.fillArc(center, radius_val, start_angle_val, end_angle_val, common.color_gray, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill arc");
            return null;
        },
        .rgb => common.canvas_rgb.?.fillArc(center, radius_val, start_angle_val, end_angle_val, common.color_rgb, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill arc");
            return null;
        },
        .rgba => common.canvas_rgba.?.fillArc(center, radius_val, start_angle_val, end_angle_val, common.color_rgba, common.mode) catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to fill arc");
            return null;
        },
    }

    return py_utils.getPyNone();
}

fn canvas_draw_text(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    // Parse arguments
    var text_obj: ?*c.PyObject = undefined;
    var position_obj: ?*c.PyObject = undefined;
    var color_obj: ?*c.PyObject = undefined;
    var font_obj: ?*c.PyObject = null;
    var scale: f64 = 1.0;
    var mode: c_long = 0;

    const kw = comptime py_utils.kw(&.{ "text", "position", "color", "font", "scale", "mode" });
    const format = std.fmt.comptimePrint("OOO|Odl:draw_text", .{});

    if (c.PyArg_ParseTupleAndKeywords(args, @constCast(kwds), format.ptr, @ptrCast(@constCast(&kw)), &text_obj, &position_obj, &color_obj, &font_obj, &scale, &mode) == 0) {
        return null;
    }

    const text_cstr = c.PyUnicode_AsUTF8(text_obj) orelse {
        c.PyErr_SetString(c.PyExc_TypeError, "text must be a string");
        return null;
    };
    const text = std.mem.span(text_cstr);

    const position = py_utils.parsePointTuple(f32, position_obj) catch return null;

    const rgba = color_utils.parseColorTo(Rgba, @ptrCast(color_obj)) catch return null;

    const mode_val = py_utils.validateRange(u32, mode, 0, 1, "Mode") catch return null;
    const draw_mode: DrawMode = @enumFromInt(mode_val);

    if (font_obj) |font| {
        const bitmap_font_module = @import("bitmap_font.zig");
        if (c.PyObject_IsInstance(font, @ptrCast(&bitmap_font_module.BitmapFontType)) <= 0) {
            if (c.PyErr_Occurred() == null) {
                c.PyErr_SetString(c.PyExc_TypeError, "font must be a BitmapFont instance or None");
            }
            return null;
        }

        const font_wrapper = @as(*bitmap_font_module.BitmapFontObject, @ptrCast(font));
        const font_ptr = py_utils.validateNonNull(*BitmapFont, font_wrapper.font, "BitmapFont") catch return null;
        switch (self.kind) {
            .gray => if (self.canvas_gray) |cptr| cptr.drawText(text, position, rgba.toGray(), font_ptr.*, @as(f32, @floatCast(scale)), draw_mode) else return null,
            .rgb => if (self.canvas_rgb) |cptr| cptr.drawText(text, position, rgba.toRgb(), font_ptr.*, @as(f32, @floatCast(scale)), draw_mode) else return null,
            .rgba => if (self.canvas_rgba) |cptr| cptr.drawText(text, position, rgba, font_ptr.*, @as(f32, @floatCast(scale)), draw_mode) else return null,
        }
    } else {
        const font = getFont8x8() catch {
            c.PyErr_SetString(c.PyExc_RuntimeError, "Failed to initialize default font");
            return null;
        };
        switch (self.kind) {
            .gray => if (self.canvas_gray) |cptr| cptr.drawText(text, position, rgba.toGray(), font, @floatCast(scale), draw_mode) else return null,
            .rgb => if (self.canvas_rgb) |cptr| cptr.drawText(text, position, rgba.toRgb(), font, @floatCast(scale), draw_mode) else return null,
            .rgba => if (self.canvas_rgba) |cptr| cptr.drawText(text, position, rgba, font, @floatCast(scale), draw_mode) else return null,
        }
    }

    return py_utils.getPyNone();
}

// Property getters
fn canvas_get_rows(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    switch (self.kind) {
        .gray => if (self.canvas_gray) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.rows)) else {},
        .rgb => if (self.canvas_rgb) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.rows)) else {},
        .rgba => if (self.canvas_rgba) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.rows)) else {},
    }
    c.PyErr_SetString(c.PyExc_ValueError, "Canvas not initialized");
    return null;
}

fn canvas_get_cols(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    switch (self.kind) {
        .gray => if (self.canvas_gray) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.cols)) else {},
        .rgb => if (self.canvas_rgb) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.cols)) else {},
        .rgba => if (self.canvas_rgba) |cptr| return c.PyLong_FromLong(@intCast(cptr.image.cols)) else {},
    }
    c.PyErr_SetString(c.PyExc_ValueError, "Canvas not initialized");
    return null;
}

fn canvas_get_image(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*CanvasObject, @ptrCast(self_obj.?));

    if (self.image_ref) |img_ref| {
        c.Py_INCREF(img_ref);
        return img_ref;
    } else {
        c.PyErr_SetString(c.PyExc_ValueError, "Canvas not initialized");
        return null;
    }
}

// Documentation strings
const canvas_draw_quadratic_bezier_doc =
    \\Draw a quadratic Bézier curve.
    \\
    \\## Parameters
    \\- `p0` (tuple[float, float]): Start point (x, y)
    \\- `p1` (tuple[float, float]): Control point (x, y)
    \\- `p2` (tuple[float, float]): End point (x, y)
    \\- `color` (int, tuple or color object): Color of the curve.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
;

const canvas_draw_cubic_bezier_doc =
    \\Draw a cubic Bézier curve.
    \\
    \\## Parameters
    \\- `p0` (tuple[float, float]): Start point (x, y)
    \\- `p1` (tuple[float, float]): First control point (x, y)
    \\- `p2` (tuple[float, float]): Second control point (x, y)
    \\- `p3` (tuple[float, float]): End point (x, y)
    \\- `color` (int, tuple or color object): Color of the curve.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
;

const canvas_draw_spline_polygon_doc =
    \\Draw a smooth spline through polygon points.
    \\
    \\## Parameters
    \\- `points` (list[tuple[float, float]]): List of (x, y) coordinates to interpolate through
    \\- `color` (int, tuple or color object): Color of the spline.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `tension` (float, optional): Spline tension (0.0 = angular, 0.5 = smooth, default: 0.5)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
;

const canvas_fill_spline_polygon_doc =
    \\Fill a smooth spline area through polygon points.
    \\
    \\## Parameters
    \\- `points` (list[tuple[float, float]]): List of (x, y) coordinates to interpolate through
    \\- `color` (int, tuple or color object): Fill color.
    \\- `tension` (float, optional): Spline tension (0.0 = angular, 0.5 = smooth, default: 0.5)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
;

const canvas_draw_arc_doc =
    \\Draw an arc outline.
    \\
    \\## Parameters
    \\- `center` (tuple[float, float]): Center coordinates (x, y)
    \\- `radius` (float): Arc radius in pixels
    \\- `start_angle` (float): Starting angle in radians (0 = right, π/2 = down, π = left, 3π/2 = up)
    \\- `end_angle` (float): Ending angle in radians
    \\- `color` (int, tuple or color object): Color of the arc.
    \\- `width` (int, optional): Line width in pixels (default: 1)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    \\
    \\## Notes
    \\- Angles are measured in radians, with 0 pointing right and increasing clockwise
    \\- For a full circle, use start_angle=0 and end_angle=2π
    \\- The arc is drawn from start_angle to end_angle in the positive angular direction
;

const canvas_fill_arc_doc =
    \\Fill an arc (pie slice) area.
    \\
    \\## Parameters
    \\- `center` (tuple[float, float]): Center coordinates (x, y)
    \\- `radius` (float): Arc radius in pixels
    \\- `start_angle` (float): Starting angle in radians (0 = right, π/2 = down, π = left, 3π/2 = up)
    \\- `end_angle` (float): Ending angle in radians
    \\- `color` (int, tuple or color object): Fill color.
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
    \\
    \\## Notes
    \\- Creates a filled pie slice from the center to the arc edge
    \\- Angles are measured in radians, with 0 pointing right and increasing clockwise
    \\- For a full circle, use start_angle=0 and end_angle=2π
;

const canvas_draw_text_doc =
    \\Draw text on the canvas.
    \\
    \\## Parameters
    \\- `text` (str): Text to draw
    \\- `position` (tuple[float, float]): Position coordinates (x, y)
    \\- `color` (int, tuple or color object): Text color.
    \\- `font` (BitmapFont, optional): Font object to use for rendering. If `None`, uses BitmapFont.font8x8()
    \\- `scale` (float, optional): Text scale factor (default: 1.0)
    \\- `mode` (`DrawMode`, optional): Drawing mode (default: `DrawMode.FAST`)
;

pub const canvas_methods_metadata = [_]stub_metadata.MethodWithMetadata{
    .{
        .name = "fill",
        .meth = @ptrCast(&canvas_fill),
        .flags = c.METH_VARARGS,
        .doc = canvas_fill_doc,
        .params = "self, color: " ++ stub_metadata.COLOR,
        .returns = "None",
    },
    .{
        .name = "draw_line",
        .meth = @ptrCast(&DrawLine.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = DrawLine.doc_string,
        .params = "self, p1: tuple[float, float], p2: tuple[float, float], color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_rectangle",
        .meth = @ptrCast(&DrawRectangle.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = DrawRectangle.doc_string,
        .params = "self, rect: Rectangle, color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "fill_rectangle",
        .meth = @ptrCast(&FillRectangle.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = FillRectangle.doc_string,
        .params = "self, rect: Rectangle, color: " ++ stub_metadata.COLOR ++ ", mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_polygon",
        .meth = @ptrCast(&DrawPolygon.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = DrawPolygon.doc_string,
        .params = "self, points: list[tuple[float, float]], color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "fill_polygon",
        .meth = @ptrCast(&FillPolygon.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = FillPolygon.doc_string,
        .params = "self, points: list[tuple[float, float]], color: " ++ stub_metadata.COLOR ++ ", mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_circle",
        .meth = @ptrCast(&DrawCircle.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = DrawCircle.doc_string,
        .params = "self, center: tuple[float, float], radius: float, color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "fill_circle",
        .meth = @ptrCast(&FillCircle.method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = FillCircle.doc_string,
        .params = "self, center: tuple[float, float], radius: float, color: " ++ stub_metadata.COLOR ++ ", mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_arc",
        .meth = @ptrCast(&canvas_draw_arc),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_draw_arc_doc,
        .params = "self, center: tuple[float, float], radius: float, start_angle: float, end_angle: float, color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "fill_arc",
        .meth = @ptrCast(&canvas_fill_arc),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_fill_arc_doc,
        .params = "self, center: tuple[float, float], radius: float, start_angle: float, end_angle: float, color: " ++ stub_metadata.COLOR ++ ", mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_quadratic_bezier",
        .meth = @ptrCast(&canvas_draw_quadratic_bezier),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_draw_quadratic_bezier_doc,
        .params = "self, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_cubic_bezier",
        .meth = @ptrCast(&canvas_draw_cubic_bezier),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_draw_cubic_bezier_doc,
        .params = "self, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], color: " ++ stub_metadata.COLOR ++ ", width: int = 1, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_spline_polygon",
        .meth = @ptrCast(&canvas_draw_spline_polygon),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_draw_spline_polygon_doc,
        .params = "self, points: list[tuple[float, float]], color: " ++ stub_metadata.COLOR ++ ", width: int = 1, tension: float = 0.5, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "fill_spline_polygon",
        .meth = @ptrCast(&canvas_fill_spline_polygon),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_fill_spline_polygon_doc,
        .params = "self, points: list[tuple[float, float]], color: " ++ stub_metadata.COLOR ++ ", tension: float = 0.5, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
    .{
        .name = "draw_text",
        .meth = @ptrCast(&canvas_draw_text),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = canvas_draw_text_doc,
        .params = "self, text: str, position: tuple[float, float], color: " ++ stub_metadata.COLOR ++ ", font: BitmapFont = BitmapFont.font8x8(), scale: float = 1.0, mode: DrawMode = DrawMode.FAST",
        .returns = "None",
    },
};

var canvas_methods = stub_metadata.toPyMethodDefArray(&canvas_methods_metadata);

pub const canvas_properties_metadata = [_]stub_metadata.PropertyWithMetadata{
    .{
        .name = "rows",
        .get = @ptrCast(&canvas_get_rows),
        .set = null,
        .doc = "Number of rows in the canvas",
        .type = "int",
    },
    .{
        .name = "cols",
        .get = @ptrCast(&canvas_get_cols),
        .set = null,
        .doc = "Number of columns in the canvas",
        .type = "int",
    },
    .{
        .name = "image",
        .get = @ptrCast(&canvas_get_image),
        .set = null,
        .doc = "Parent Image object",
        .type = "Image",
    },
};

var canvas_getset = stub_metadata.toPyGetSetDefArray(&canvas_properties_metadata);

// Canvas class documentation - keep it simple
const canvas_class_doc = "Canvas for drawing operations on images.";

// Canvas init documentation - detailed explanation
pub const canvas_init_doc =
    \\Create a Canvas for drawing operations on an Image.
    \\
    \\A Canvas provides drawing methods to modify the pixels of an Image. The Canvas
    \\maintains a reference to the parent Image to prevent it from being garbage collected
    \\while drawing operations are in progress.
    \\
    \\## Parameters
    \\- `image` (Image): The Image object to draw on. Must be initialized with dimensions.
    \\
    \\## Examples
    \\```python
    \\# Create an image and get its canvas
    \\img = Image(100, 100, Rgb(255, 255, 255))
    \\canvas = Canvas(img)
    \\
    \\# Draw on the canvas
    \\canvas.fill(Rgb(0, 0, 0))
    \\canvas.draw_circle((50, 50), 20, Rgb(255, 0, 0))
    \\```
    \\
    \\## Notes
    \\- The Canvas holds a reference to the parent Image
    \\- All drawing operations modify the original Image pixels
    \\- Use Image.canvas() method as a convenient way to create a Canvas
;

// Special methods metadata for stub generation
pub const canvas_special_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "__init__",
        .params = "self, image: Image",
        .returns = "None",
        .doc = canvas_init_doc,
    },
};

pub var CanvasType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.Canvas",
    .tp_basicsize = @sizeOf(CanvasObject),
    .tp_dealloc = canvas_dealloc,
    .tp_repr = canvas_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = canvas_class_doc,
    .tp_methods = @ptrCast(&canvas_methods),
    .tp_getset = @ptrCast(&canvas_getset),
    .tp_init = canvas_init,
    .tp_new = canvas_new,
};

// No runtime wrapper; DrawMode is registered via enum_utils.registerEnum in main
