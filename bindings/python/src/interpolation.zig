const std = @import("std");
const py_utils = @import("py_utils.zig");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// INTERPOLATION METHOD ENUM
// ============================================================================

// Python object for InterpolationMethod (empty, it's just an enum)
pub const InterpolationMethodObject = extern struct {
    ob_base: c.PyObject,
};

// Create the enum type
pub var InterpolationMethodType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.InterpolationMethod",
    .tp_basicsize = 0, // No instance size needed for enum
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = "Interpolation methods for image resizing",
    .tp_new = null, // Cannot instantiate
};

// Register the InterpolationMethod enum
pub fn registerInterpolationMethod(module: *c.PyObject) !void {
    // Create the enum using Python's enum module
    const enum_module = c.PyImport_ImportModule("enum") orelse return error.ImportFailed;
    defer c.Py_DECREF(enum_module);

    const int_enum = c.PyObject_GetAttrString(enum_module, "IntEnum") orelse return error.AttributeFailed;
    defer c.Py_DECREF(int_enum);

    // Create the enum values dictionary
    const values = c.PyDict_New() orelse return error.DictCreationFailed;
    defer c.Py_DECREF(values);

    // Add enum values
    const enum_values = .{
        .{ "NEAREST_NEIGHBOR", 0 },
        .{ "BILINEAR", 1 },
        .{ "BICUBIC", 2 },
        .{ "CATMULL_ROM", 3 },
        .{ "MITCHELL", 4 },
        .{ "LANCZOS", 5 },
    };

    inline for (enum_values) |value| {
        const py_value = c.PyLong_FromLong(value[1]) orelse return error.ValueCreationFailed;
        defer c.Py_DECREF(py_value);
        if (c.PyDict_SetItemString(values, value[0], py_value) < 0) {
            return error.DictSetFailed;
        }
    }

    // Create the enum class: IntEnum('InterpolationMethod', values)
    const args = c.PyTuple_Pack(2, c.PyUnicode_FromString("InterpolationMethod"), values) orelse return error.TupleCreationFailed;
    defer c.Py_DECREF(args);

    const interpolation_method = c.PyObject_CallObject(int_enum, args) orelse return error.EnumCreationFailed;

    // Add to module
    if (c.PyModule_AddObject(module, "InterpolationMethod", interpolation_method) < 0) {
        c.Py_DECREF(interpolation_method);
        return error.ModuleAddFailed;
    }
}
