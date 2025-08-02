const std = @import("std");

const zignal = @import("zignal");
const BitmapFont = zignal.BitmapFont;

const py_utils = @import("py_utils.zig");
const allocator = py_utils.allocator;
pub const registerType = py_utils.registerType;
const c = py_utils.c;

pub const BitmapFontObject = extern struct {
    ob_base: c.PyObject,
    font: ?*BitmapFont,
};

fn bitmap_font_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*BitmapFontObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.font = null; // Initialize to null
    }
    return @as(?*c.PyObject, @ptrCast(self));
}

fn bitmap_font_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    const self = @as(*BitmapFontObject, @ptrCast(self_obj.?));

    // The default font is static, only free dynamically loaded fonts
    if (self.font) |font| {
        if (font != &zignal.font.default_font_8x8) {
            font.deinit(allocator);
            allocator.destroy(font);
        }
    }

    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn bitmap_font_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self_obj;
    return c.PyUnicode_FromString("BitmapFont()");
}

// Class method: load
const bitmap_font_load_doc =
    \\Load a bitmap font from file.
    \\
    \\Supports BDF (Bitmap Distribution Format) and PCF (Portable Compiled Format) files.
    \\
    \\## Parameters
    \\- `path` (str): Path to the font file
    \\
    \\## Returns
    \\BitmapFont: The loaded font object
    \\
    \\## Examples
    \\```python
    \\font = BitmapFont.load("unifont.bdf")
    \\canvas.draw_text("Hello", (10, 10), font, (255, 255, 255))
    \\```
;

fn bitmap_font_load(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    var path_obj: ?*c.PyObject = undefined;

    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &path_obj) == 0) {
        return null;
    }

    // Convert Python string to Zig string
    const path_cstr = c.PyUnicode_AsUTF8(path_obj) orelse {
        c.PyErr_SetString(c.PyExc_TypeError, "path must be a string");
        return null;
    };
    const path = std.mem.span(path_cstr);

    // Create new BitmapFont instance
    const instance = c.PyObject_CallObject(@ptrCast(type_obj), null) orelse return null;
    const self = @as(*BitmapFontObject, @ptrCast(instance));

    // Allocate font on heap
    const font_ptr = allocator.create(BitmapFont) catch {
        c.Py_DECREF(instance);
        c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate font");
        return null;
    };

    // Load font from file (loading all characters)
    font_ptr.* = BitmapFont.load(allocator, path, .all) catch |err| {
        allocator.destroy(font_ptr);
        c.Py_DECREF(instance);

        // Set appropriate Python exception based on error
        switch (err) {
            error.FileNotFound => c.PyErr_SetString(c.PyExc_FileNotFoundError, "Font file not found"),
            error.UnsupportedFontFormat => c.PyErr_SetString(c.PyExc_ValueError, "Unsupported font format"),
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory loading font"),
            else => {
                var buf: [256]u8 = undefined;
                const msg = std.fmt.bufPrintZ(&buf, "Failed to load font: {s}", .{@errorName(err)}) catch "Failed to load font";
                c.PyErr_SetString(c.PyExc_RuntimeError, msg);
            },
        }
        return null;
    };

    self.font = font_ptr;
    return instance;
}

// Class method: get_default_font
const bitmap_font_get_default_font_doc =
    \\Get the built-in default 8x8 bitmap font.
    \\
    \\This font covers ASCII characters from 0x20 (space) to 0x7E (tilde).
    \\
    \\## Returns
    \\BitmapFont: The default font object
    \\
    \\## Examples
    \\```python
    \\font = BitmapFont.get_default_font()
    \\canvas.draw_text("Hello World!", (10, 10), font, (255, 255, 255))
    \\```
;

fn bitmap_font_get_default_font(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;

    // Create new BitmapFont instance
    const instance = c.PyObject_CallObject(@ptrCast(type_obj), null) orelse return null;
    const self = @as(*BitmapFontObject, @ptrCast(instance));

    // Point to the static default font
    self.font = @constCast(&zignal.font.default_font_8x8);

    return instance;
}

var bitmap_font_methods = [_]c.PyMethodDef{
    .{ .ml_name = "load", .ml_meth = @ptrCast(&bitmap_font_load), .ml_flags = c.METH_VARARGS | c.METH_CLASS, .ml_doc = bitmap_font_load_doc },
    .{ .ml_name = "get_default_font", .ml_meth = @ptrCast(&bitmap_font_get_default_font), .ml_flags = c.METH_NOARGS | c.METH_CLASS, .ml_doc = bitmap_font_get_default_font_doc },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

pub var BitmapFontType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.BitmapFont",
    .tp_basicsize = @sizeOf(BitmapFontObject),
    .tp_dealloc = bitmap_font_dealloc,
    .tp_repr = bitmap_font_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = "Bitmap font for text rendering",
    .tp_methods = @ptrCast(&bitmap_font_methods),
    .tp_new = bitmap_font_new,
};
