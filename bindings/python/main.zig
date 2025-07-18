const std = @import("std");
const c = @cImport({
    @cDefine("Py_LIMITED_API", "3");
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

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

pub export fn PyInit_zignal() [*]c.PyObject {
    return c.PyModule_Create(&zignal_module);
}

pub fn main() void {}
