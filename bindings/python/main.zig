const std = @import("std");
const zignal = @import("zignal");
const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// RGB TYPE
// ============================================================================

const RgbObject = extern struct {
    ob_base: c.PyObject,
    r: u8,
    g: u8,
    b: u8,
};

fn rgb_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;
    
    const self = @as(?*RgbObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.r = 0;
        obj.g = 0;
        obj.b = 0;
    }
    return @ptrCast(self);
}

fn rgb_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = kwds;
    
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    var r: c_int = 0;
    var g: c_int = 0;
    var b: c_int = 0;
    
    if (c.PyArg_ParseTuple(args, "iii", &r, &g, &b) == 0) {
        return -1;
    }
    
    // Validate range
    if (r < 0 or r > 255 or g < 0 or g > 255 or b < 0 or b > 255) {
        c.PyErr_SetString(c.PyExc_ValueError, "RGB values must be in range 0-255");
        return -1;
    }
    
    self.r = @intCast(r);
    self.g = @intCast(g);
    self.b = @intCast(b);
    
    return 0;
}

fn rgb_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn rgb_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    var buffer: [64]u8 = undefined;
    const formatted = std.fmt.bufPrintZ(&buffer, "Rgb(r={d}, g={d}, b={d})", .{
        self.r, self.g, self.b
    }) catch return null;
    return c.PyUnicode_FromString(formatted.ptr);
}

// Property getters
fn rgb_get_r(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    return c.PyLong_FromLong(self.r);
}

fn rgb_get_g(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    return c.PyLong_FromLong(self.g);
}

fn rgb_get_b(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    return c.PyLong_FromLong(self.b);
}

// Methods
fn rgb_to_hex(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    const rgb = zignal.Rgb{ .r = self.r, .g = self.g, .b = self.b };
    return c.PyLong_FromLong(rgb.toHex());
}

fn rgb_luma(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    const rgb = zignal.Rgb{ .r = self.r, .g = self.g, .b = self.b };
    return c.PyFloat_FromDouble(rgb.luma());
}

fn rgb_is_gray(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*RgbObject, @ptrCast(self_obj.?));
    const rgb = zignal.Rgb{ .r = self.r, .g = self.g, .b = self.b };
    const py_true = @extern(*c.PyObject, .{ .name = "_Py_TrueStruct", .linkage = .weak });
    const py_false = @extern(*c.PyObject, .{ .name = "_Py_FalseStruct", .linkage = .weak });
    
    if (rgb.isGray()) {
        c.Py_INCREF(py_true);
        return py_true;
    }
    c.Py_INCREF(py_false);
    return py_false;
}

// Class methods
fn rgb_from_hex(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    var hex_value: c_uint = 0;
    
    if (c.PyArg_ParseTuple(args, "I", &hex_value) == 0) {
        return null;
    }
    
    if (hex_value > 0xFFFFFF) {
        c.PyErr_SetString(c.PyExc_ValueError, "Hex value must be in range 0x000000-0xFFFFFF");
        return null;
    }
    
    const rgb = zignal.Rgb.fromHex(@intCast(hex_value));
    
    // Create new instance
    const new_args = c.Py_BuildValue("(iii)", rgb.r, rgb.g, rgb.b);
    defer c.Py_DECREF(new_args);
    
    return c.PyObject_CallObject(type_obj, new_args);
}

fn rgb_from_gray(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    var gray: c_int = 0;
    
    if (c.PyArg_ParseTuple(args, "i", &gray) == 0) {
        return null;
    }
    
    if (gray < 0 or gray > 255) {
        c.PyErr_SetString(c.PyExc_ValueError, "Gray value must be in range 0-255");
        return null;
    }
    
    const rgb = zignal.Rgb.fromGray(@intCast(gray));
    
    // Create new instance
    const new_args = c.Py_BuildValue("(iii)", rgb.r, rgb.g, rgb.b);
    defer c.Py_DECREF(new_args);
    
    return c.PyObject_CallObject(type_obj, new_args);
}

var rgb_methods = [_]c.PyMethodDef{
    .{ .ml_name = "to_hex", .ml_meth = rgb_to_hex, .ml_flags = c.METH_NOARGS, 
       .ml_doc = "Convert RGB to hexadecimal representation (0xRRGGBB)" },
    .{ .ml_name = "luma", .ml_meth = rgb_luma, .ml_flags = c.METH_NOARGS, 
       .ml_doc = "Calculate perceptual luminance using ITU-R BT.709 coefficients" },
    .{ .ml_name = "is_gray", .ml_meth = rgb_is_gray, .ml_flags = c.METH_NOARGS, 
       .ml_doc = "Check if all RGB components are equal (grayscale)" },
    .{ .ml_name = "from_hex", .ml_meth = rgb_from_hex, .ml_flags = c.METH_VARARGS | c.METH_CLASS,
       .ml_doc = "Create RGB from hexadecimal value (0xRRGGBB format)" },
    .{ .ml_name = "from_gray", .ml_meth = rgb_from_gray, .ml_flags = c.METH_VARARGS | c.METH_CLASS,
       .ml_doc = "Create RGB from grayscale value" },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var rgb_getset = [_]c.PyGetSetDef{
    .{ .name = "r", .get = rgb_get_r, .set = null, .doc = "Red component (0-255)", .closure = null },
    .{ .name = "g", .get = rgb_get_g, .set = null, .doc = "Green component (0-255)", .closure = null },
    .{ .name = "b", .get = rgb_get_b, .set = null, .doc = "Blue component (0-255)", .closure = null },
    .{ .name = null, .get = null, .set = null, .doc = null, .closure = null },
};

var RgbType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.Rgb",
    .tp_basicsize = @sizeOf(RgbObject),
    .tp_itemsize = 0,
    .tp_dealloc = rgb_dealloc,
    .tp_vectorcall_offset = 0,
    .tp_getattr = null,
    .tp_setattr = null,
    .tp_as_async = null,
    .tp_repr = rgb_repr,
    .tp_as_number = null,
    .tp_as_sequence = null,
    .tp_as_mapping = null,
    .tp_hash = null,
    .tp_call = null,
    .tp_str = rgb_repr,
    .tp_getattro = null,
    .tp_setattro = null,
    .tp_as_buffer = null,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = "RGB color in sRGB colorspace with components in range 0-255",
    .tp_traverse = null,
    .tp_clear = null,
    .tp_richcompare = null,
    .tp_weaklistoffset = 0,
    .tp_iter = null,
    .tp_iternext = null,
    .tp_methods = &rgb_methods,
    .tp_members = null,
    .tp_getset = &rgb_getset,
    .tp_base = null,
    .tp_dict = null,
    .tp_descr_get = null,
    .tp_descr_set = null,
    .tp_dictoffset = 0,
    .tp_init = rgb_init,
    .tp_alloc = null,
    .tp_new = rgb_new,
    .tp_free = null,
    .tp_is_gc = null,
    .tp_bases = null,
    .tp_mro = null,
    .tp_cache = null,
    .tp_subclasses = null,
    .tp_weaklist = null,
    .tp_del = null,
    .tp_version_tag = 0,
    .tp_finalize = null,
    .tp_vectorcall = null,
};

// ============================================================================
// MODULE FUNCTIONS
// ============================================================================

fn zignal_hello(self: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self;
    _ = args;
    return c.PyUnicode_FromString("Hello from Zignal!");
}

var zignal_methods = [_]c.PyMethodDef{
    .{ .ml_name = "hello", .ml_meth = zignal_hello, .ml_flags = c.METH_NOARGS, .ml_doc = "A simple hello world function." },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var zignal_module = c.PyModuleDef{
    // .m_base = c.PyModuleDef_Base{
    //     .ob_base = c.PyObject{
    //         .ob_type = null,
    //     },
    //     .m_init = null,
    //     .m_index = 0,
    //     .m_copy = null,
    // },
    .m_name = "zignal",
    .m_doc = "zero dependency image processing library",
    .m_size = -1,
    .m_methods = &zignal_methods,
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

pub export fn PyInit_zignal() ?*c.PyObject {
    const m = c.PyModule_Create(&zignal_module);
    if (m == null) return null;
    
    // Initialize Rgb type
    if (c.PyType_Ready(&RgbType) < 0) return null;
    
    c.Py_INCREF(@ptrCast(&RgbType));
    if (c.PyModule_AddObject(m, "Rgb", @ptrCast(&RgbType)) < 0) {
        c.Py_DECREF(@ptrCast(&RgbType));
        c.Py_DECREF(m);
        return null;
    }
    
    return m;
}

pub fn main() void {}
