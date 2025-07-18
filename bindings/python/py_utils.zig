const std = @import("std");
const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// GENERAL PYTHON C API UTILITIES
// ============================================================================

/// Helper to register a type with a module
pub fn registerType(module: [*c]c.PyObject, comptime name: []const u8, type_obj: *c.PyTypeObject) !void {
    if (c.PyType_Ready(type_obj) < 0) return error.TypeInitFailed;

    c.Py_INCREF(@as(?*c.PyObject, @ptrCast(type_obj)));
    if (c.PyModule_AddObject(module, name.ptr, @ptrCast(type_obj)) < 0) {
        c.Py_DECREF(@ptrCast(type_obj));
        return error.TypeAddFailed;
    }
}

/// Get Python boolean singletons
pub fn getPyBool(value: bool) [*c]c.PyObject {
    const py_true = @extern(*c.PyObject, .{ .name = "_Py_TrueStruct", .linkage = .weak });
    const py_false = @extern(*c.PyObject, .{ .name = "_Py_FalseStruct", .linkage = .weak });

    const result = if (value) py_true else py_false;
    c.Py_INCREF(result);
    return @ptrCast(result);
}

/// Generate a property getter function for any integer field
pub fn makeFieldGetter(comptime ObjectType: type, comptime field_name: []const u8) fn (?*c.PyObject, ?*anyopaque) callconv(.c) ?*c.PyObject {
    return struct {
        fn getter(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
            _ = closure;
            const self = @as(*ObjectType, @ptrCast(self_obj.?));
            return c.PyLong_FromLong(@field(self, field_name));
        }
    }.getter;
}

/// Generate property getters for all fields in a struct type
pub fn makePropertyGetters(comptime ObjectType: type) []c.PyGetSetDef {
    const struct_info = @typeInfo(ObjectType).Struct;
    comptime var getters: [struct_info.fields.len + 1]c.PyGetSetDef = undefined;

    inline for (struct_info.fields, 0..) |field, i| {
        // Skip the ob_base field
        if (std.mem.eql(u8, field.name, "ob_base")) continue;

        // Only handle u8 fields for now
        if (field.type == u8) {
            getters[i] = c.PyGetSetDef{
                .name = field.name.ptr,
                .get = makeFieldGetter(ObjectType, field.name),
                .set = null,
                .doc = field.name.ptr ++ " component (0-255)",
                .closure = null,
            };
        }
    }

    // Null terminator
    getters[struct_info.fields.len] = c.PyGetSetDef{
        .name = null,
        .get = null,
        .set = null,
        .doc = null,
        .closure = null,
    };

    return &getters;
}

// Future utility ideas:
// - makeMethodWrapper() for wrapping Zig methods with Python C API calling convention
// - makeTypeFromStruct() to automatically generate entire Python types from Zig structs
// - Error handling utilities for consistent Python exception raising
// - Memory management helpers for complex object hierarchies
