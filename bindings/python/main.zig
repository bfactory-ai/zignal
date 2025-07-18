const std = @import("std");

const color = @import("color.zig");
const py_utils = @import("py_utils.zig");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

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
    .m_name = "zignal",
    .m_doc = "zero dependency image processing library",
    .m_size = -1,
    .m_methods = &zignal_methods,
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

pub export fn PyInit__zignal() ?*c.PyObject {
    const m = c.PyModule_Create(&zignal_module);
    if (m == null) return null;

    // Register Rgb type
    py_utils.registerType(@ptrCast(m), "Rgb", @ptrCast(&color.RgbType)) catch {
        c.Py_DECREF(m);
        return null;
    };

    return m;
}

pub fn main() void {}
