const std = @import("std");

const zignal = @import("zignal");
const Rectangle = zignal.Rectangle;

const py_utils = @import("py_utils.zig");
const allocator = py_utils.allocator;
pub const registerType = py_utils.registerType;
const c = py_utils.c;
const stub_metadata = @import("stub_metadata.zig");

pub const RectangleObject = extern struct {
    ob_base: c.PyObject,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
};

fn rectangle_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*RectangleObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.left = 0;
        obj.top = 0;
        obj.right = 0;
        obj.bottom = 0;
    }
    return @as(?*c.PyObject, @ptrCast(self));
}

fn rectangle_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = kwds;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    // Parse arguments - expect left, top, right, bottom
    var left: f64 = undefined;
    var top: f64 = undefined;
    var right: f64 = undefined;
    var bottom: f64 = undefined;
    const format = std.fmt.comptimePrint("dddd", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &left, &top, &right, &bottom) == 0) {
        return -1;
    }

    // Validate rectangle
    if (right < left) {
        c.PyErr_SetString(c.PyExc_ValueError, "Right must be greater than or equal to left");
        return -1;
    }
    if (bottom < top) {
        c.PyErr_SetString(c.PyExc_ValueError, "Bottom must be greater than or equal to top");
        return -1;
    }

    // Store as f32
    self.left = @as(f32, @floatCast(left));
    self.top = @as(f32, @floatCast(top));
    self.right = @as(f32, @floatCast(right));
    self.bottom = @as(f32, @floatCast(bottom));

    return 0;
}

fn rectangle_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn rectangle_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var buffer: [128]u8 = undefined;
    const formatted = std.fmt.bufPrintZ(&buffer, "Rectangle({d:.2}, {d:.2}, {d:.2}, {d:.2})", .{ self.left, self.top, self.right, self.bottom }) catch return null;
    return c.PyUnicode_FromString(formatted.ptr);
}

// Property getters
fn rectangle_get_left(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    return c.PyFloat_FromDouble(@as(f64, self.left));
}

fn rectangle_get_top(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    return c.PyFloat_FromDouble(@as(f64, self.top));
}

fn rectangle_get_right(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    return c.PyFloat_FromDouble(@as(f64, self.right));
}

fn rectangle_get_bottom(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    return c.PyFloat_FromDouble(@as(f64, self.bottom));
}

fn rectangle_get_width(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    const width = self.right - self.left;
    return c.PyFloat_FromDouble(@as(f64, width));
}

fn rectangle_get_height(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    const height = self.bottom - self.top;
    return c.PyFloat_FromDouble(@as(f64, height));
}

pub const rectangle_properties_metadata = [_]stub_metadata.PropertyWithMetadata{
    .{
        .name = "left",
        .get = @ptrCast(&rectangle_get_left),
        .set = null,
        .doc = "Left coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "top",
        .get = @ptrCast(&rectangle_get_top),
        .set = null,
        .doc = "Top coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "right",
        .get = @ptrCast(&rectangle_get_right),
        .set = null,
        .doc = "Right coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "bottom",
        .get = @ptrCast(&rectangle_get_bottom),
        .set = null,
        .doc = "Bottom coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "width",
        .get = @ptrCast(&rectangle_get_width),
        .set = null,
        .doc = "Width of the rectangle (right - left)",
        .type = "float",
    },
    .{
        .name = "height",
        .get = @ptrCast(&rectangle_get_height),
        .set = null,
        .doc = "Height of the rectangle (bottom - top)",
        .type = "float",
    },
};

var rectangle_getset = stub_metadata.toPyGetSetDefArray(&rectangle_properties_metadata);

pub var RectangleType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.Rectangle",
    .tp_basicsize = @sizeOf(RectangleObject),
    .tp_dealloc = rectangle_dealloc,
    .tp_repr = rectangle_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = "A rectangle defined by its left, top, right, and bottom coordinates",
    .tp_getset = @ptrCast(&rectangle_getset),
    .tp_init = rectangle_init,
    .tp_new = rectangle_new,
};

/// Convert a Python Rectangle object to a Zignal Rectangle(f32)
pub fn toZignalRectangle(rect_obj: ?*c.PyObject) !Rectangle(f32) {
    if (rect_obj == null) {
        c.PyErr_SetString(c.PyExc_TypeError, "Rectangle object is null");
        return error.InvalidRectangle;
    }

    // Check if it's a Rectangle instance
    if (c.PyObject_IsInstance(rect_obj, @ptrCast(&RectangleType)) <= 0) {
        c.PyErr_SetString(c.PyExc_TypeError, "Object must be a Rectangle instance");
        return error.InvalidRectangle;
    }

    const rect = @as(*RectangleObject, @ptrCast(rect_obj.?));
    return Rectangle(f32).init(rect.left, rect.top, rect.right, rect.bottom);
}
