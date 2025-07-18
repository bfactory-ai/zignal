const std = @import("std");

const zignal = @import("zignal");

const py_utils = @import("py_utils.zig");
pub const registerType = py_utils.registerType;

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// IMAGERGB TYPE
// ============================================================================

const ImageRgbObject = extern struct {
    ob_base: c.PyObject,
    // Store pointer to heap-allocated image data (optional)
    image_ptr: ?*zignal.Image(zignal.Rgb),
};

fn imagergb_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*ImageRgbObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        // Initialize to null pointer to avoid undefined behavior
        obj.image_ptr = null;
    }
    return @ptrCast(self);
}

fn imagergb_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = self_obj;
    _ = args;
    _ = kwds;
    
    // For now, don't allow direct instantiation
    c.PyErr_SetString(c.PyExc_TypeError, "ImageRgb cannot be instantiated directly. Use ImageRgb.load() instead.");
    return -1;
}

// Global allocator for the module
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

fn imagergb_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    const self = @as(*ImageRgbObject, @ptrCast(self_obj.?));
    
    // Free the image data if it was allocated
    if (self.image_ptr) |ptr| {
        const allocator = gpa.allocator();
        ptr.deinit(allocator);
        allocator.destroy(ptr);
    }
    
    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn imagergb_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*ImageRgbObject, @ptrCast(self_obj.?));
    
    if (self.image_ptr) |ptr| {
        var buffer: [64]u8 = undefined;
        const formatted = std.fmt.bufPrintZ(&buffer, "ImageRgb({d}x{d})", .{ ptr.rows, ptr.cols }) catch return null;
        return c.PyUnicode_FromString(formatted.ptr);
    } else {
        return c.PyUnicode_FromString("ImageRgb(uninitialized)");
    }
}

// Property getters
fn imagergb_get_rows(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*ImageRgbObject, @ptrCast(self_obj.?));
    
    if (self.image_ptr) |ptr| {
        return c.PyLong_FromLong(@intCast(ptr.rows));
    } else {
        c.PyErr_SetString(c.PyExc_ValueError, "Image not initialized");
        return null;
    }
}

fn imagergb_get_cols(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*ImageRgbObject, @ptrCast(self_obj.?));
    
    if (self.image_ptr) |ptr| {
        return c.PyLong_FromLong(@intCast(ptr.cols));
    } else {
        c.PyErr_SetString(c.PyExc_ValueError, "Image not initialized");
        return null;
    }
}

// Class methods
fn imagergb_load(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    var file_path: [*c]const u8 = undefined;
    
    if (c.PyArg_ParseTuple(args, "s", &file_path) == 0) {
        return null;
    }
    
    // Use global allocator
    const allocator = gpa.allocator();
    
    // Convert C string to Zig slice
    const path_slice = std.mem.span(file_path);
    
    // Load the image
    const image = zignal.Image(zignal.Rgb).load(allocator, path_slice) catch |err| {
        switch (err) {
            error.FileNotFound => c.PyErr_SetString(c.PyExc_FileNotFoundError, "Image file not found"),
            error.UnsupportedImageFormat => c.PyErr_SetString(c.PyExc_ValueError, "Unsupported image format"),
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory"),
            else => c.PyErr_SetString(c.PyExc_IOError, "Failed to load image"),
        }
        return null;
    };
    
    // Create new Python object
    const self = @as(?*ImageRgbObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) {
        var img = image;
        img.deinit(allocator);
        return null;
    }
    
    // Allocate space for the image on heap and move it there
    const image_ptr = allocator.create(zignal.Image(zignal.Rgb)) catch {
        var img = image;
        img.deinit(allocator);
        c.Py_DECREF(@ptrCast(self));
        c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate image");
        return null;
    };
    
    image_ptr.* = image;
    self.?.image_ptr = image_ptr;
    
    return @ptrCast(self);
}

var imagergb_methods = [_]c.PyMethodDef{
    .{ .ml_name = "load", .ml_meth = imagergb_load, .ml_flags = c.METH_VARARGS | c.METH_CLASS, .ml_doc = "Load an RGB image from file" },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var imagergb_getset = [_]c.PyGetSetDef{
    .{ .name = "rows", .get = imagergb_get_rows, .set = null, .doc = "Number of rows (height) in the image", .closure = null },
    .{ .name = "cols", .get = imagergb_get_cols, .set = null, .doc = "Number of columns (width) in the image", .closure = null },
    .{ .name = null, .get = null, .set = null, .doc = null, .closure = null },
};

pub var ImageRgbType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.ImageRgb",
    .tp_basicsize = @sizeOf(ImageRgbObject),
    .tp_dealloc = imagergb_dealloc,
    .tp_repr = imagergb_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = "RGB image class",
    .tp_methods = &imagergb_methods,
    .tp_getset = &imagergb_getset,
    .tp_init = imagergb_init,
    .tp_new = imagergb_new,
};