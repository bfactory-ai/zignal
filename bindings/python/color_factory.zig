const std = @import("std");

const zignal = @import("zignal");

const color_types = @import("color_registry.zig").color_types;
const ConversionError = @import("py_utils.zig").ConversionError;
const convertFromPython = @import("py_utils.zig").convertFromPython;
const convertToPython = @import("py_utils.zig").convertToPython;
const createColorPyObject = @import("color.zig").createColorPyObject;
const getFormatString = @import("py_utils.zig").getFormatString;
const getValidationErrorMessage = @import("color_registry.zig").getValidationErrorMessage;
const isSupportedColor = @import("color_registry.zig").isSupportedColor;
const registerType = @import("py_utils.zig").registerType;
const validateColorComponent = @import("color_registry.zig").validateColorComponent;

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

/// Generate a color binding with automatic property getters and validation
pub fn createColorBinding(
    comptime name: []const u8,
    comptime ZigColorType: type,
) type {
    const fields = @typeInfo(ZigColorType).@"struct".fields;

    // Create the Python object type manually (avoiding @Type complexity)
    const ObjectType = switch (fields.len) {
        1 => extern struct {
            ob_base: c.PyObject,
            field0: fields[0].type,

            pub const field_names = [_][]const u8{fields[0].name};
        },
        2 => extern struct {
            ob_base: c.PyObject,
            field0: fields[0].type,
            field1: fields[1].type,

            pub const field_names = [_][]const u8{ fields[0].name, fields[1].name };
        },
        3 => extern struct {
            ob_base: c.PyObject,
            field0: fields[0].type,
            field1: fields[1].type,
            field2: fields[2].type,

            pub const field_names = [_][]const u8{ fields[0].name, fields[1].name, fields[2].name };
        },
        4 => extern struct {
            ob_base: c.PyObject,
            field0: fields[0].type,
            field1: fields[1].type,
            field2: fields[2].type,
            field3: fields[3].type,

            pub const field_names = [_][]const u8{ fields[0].name, fields[1].name, fields[2].name, fields[3].name };
        },
        else => @compileError("Color types with more than 4 fields not supported yet"),
    };

    return struct {
        pub const PyObjectType = ObjectType;
        pub const ZigType = ZigColorType;

        /// Generate property getters and setters
        pub fn generateGetSet() [fields.len + 1]c.PyGetSetDef {
            var getset: [fields.len + 1]c.PyGetSetDef = undefined;
            inline for (fields, 0..) |field, i| {
                getset[i] = c.PyGetSetDef{
                    .name = field.name ++ "",
                    .get = generateFieldGetter(i),
                    .set = generateFieldSetter(i),
                    .doc = field.name ++ " component",
                    .closure = null,
                };
            }
            getset[fields.len] = c.PyGetSetDef{ .name = null, .get = null, .set = null, .doc = null, .closure = null };
            return getset;
        }

        /// Generate field getter for specific field index
        fn generateFieldGetter(comptime field_index: usize) fn ([*c]c.PyObject, ?*anyopaque) callconv(.c) [*c]c.PyObject {
            return struct {
                fn getter(self_obj: [*c]c.PyObject, closure: ?*anyopaque) callconv(.c) [*c]c.PyObject {
                    _ = closure;
                    const self = @as(*ObjectType, @ptrCast(self_obj));
                    const value = switch (field_index) {
                        0 => self.field0,
                        1 => if (fields.len > 1) self.field1 else unreachable,
                        2 => if (fields.len > 2) self.field2 else unreachable,
                        3 => if (fields.len > 3) self.field3 else unreachable,
                        else => unreachable,
                    };
                    return @ptrCast(@alignCast(convertToPython(value)));
                }
            }.getter;
        }

        /// Generate field setter for specific field index
        fn generateFieldSetter(comptime field_index: usize) fn ([*c]c.PyObject, [*c]c.PyObject, ?*anyopaque) callconv(.c) c_int {
            return struct {
                fn setter(self_obj: [*c]c.PyObject, value_obj: [*c]c.PyObject, closure: ?*anyopaque) callconv(.c) c_int {
                    _ = closure;
                    const self = @as(*ObjectType, @ptrCast(self_obj));

                    // Get field info
                    const field = fields[field_index];
                    const field_name = field.name ++ "";

                    // Convert Python value using idiomatic error union
                    const new_value = convertFromPython(field.type, @ptrCast(value_obj)) catch |err| {
                        switch (err) {
                            ConversionError.not_integer => {
                                c.PyErr_SetString(c.PyExc_TypeError, "Expected integer value");
                                return -1;
                            },
                            ConversionError.not_float => {
                                c.PyErr_SetString(c.PyExc_TypeError, "Expected float value");
                                return -1;
                            },
                            ConversionError.integer_out_of_range => {
                                // Generate helpful range message without the attempted value
                                if (@typeInfo(field.type) == .int) {
                                    const min_val = std.math.minInt(field.type);
                                    const max_val = std.math.maxInt(field.type);

                                    var buffer: [256]u8 = undefined;
                                    const msg = std.fmt.bufPrintZ(&buffer, "Value is out of range for {s} (valid range: {} to {})", .{ @typeName(field.type), min_val, max_val }) catch "Value out of range";

                                    c.PyErr_SetString(c.PyExc_ValueError, msg.ptr);
                                    return -1;
                                } else {
                                    c.PyErr_SetString(c.PyExc_ValueError, "Value out of range");
                                    return -1;
                                }
                            },
                            else => {
                                c.PyErr_SetString(c.PyExc_TypeError, "Unsupported value type");
                                return -1;
                            },
                        }
                    };

                    // Validate using the color registry
                    if (!validateColorComponent(ZigColorType, field_name, new_value)) {
                        const error_msg = getValidationErrorMessage(ZigColorType);
                        c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                        return -1;
                    }

                    // Set the field value
                    switch (field_index) {
                        0 => self.field0 = new_value,
                        1 => if (fields.len > 1) {
                            self.field1 = new_value;
                        } else unreachable,
                        2 => if (fields.len > 2) {
                            self.field2 = new_value;
                        } else unreachable,
                        3 => if (fields.len > 3) {
                            self.field3 = new_value;
                        } else unreachable,
                        else => unreachable,
                    }

                    return 0; // Success
                }
            }.setter;
        }

        /// Generate methods array - automatically create conversion methods for all color types
        pub fn generateMethods() [color_types.len]c.PyMethodDef {
            var methods: [color_types.len]c.PyMethodDef = undefined;
            var index: usize = 0;

            // Generate conversion methods for each color type
            inline for (color_types) |TargetColorType| {
                const method_name = getConversionMethodName(TargetColorType);
                const zig_method_name = getZigConversionMethodName(TargetColorType);

                // Skip self-conversion (e.g., Rgb.toRgb doesn't exist)
                if (TargetColorType == ZigColorType) continue;

                // Check if the Zig type has this conversion method
                if (@hasDecl(ZigColorType, zig_method_name)) {
                    methods[index] = c.PyMethodDef{
                        .ml_name = method_name.ptr,
                        .ml_meth = generateConversionMethod(TargetColorType),
                        .ml_flags = c.METH_NOARGS,
                        .ml_doc = getConversionMethodDoc(TargetColorType).ptr,
                    };
                    index += 1;
                } else {
                    @compileError("Missing conversion method: " ++ @typeName(ZigColorType) ++ "." ++ zig_method_name ++ " - expected for color type in registry");
                }
            }
            methods[index] = c.PyMethodDef{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null };
            return methods;
        }

        /// Convert string to lowercase at comptime
        fn comptimeLowercase(comptime input: []const u8) []const u8 {
            comptime var result: [input.len]u8 = undefined;
            inline for (input, 0..) |char, i| {
                result[i] = std.ascii.toLower(char);
            }
            return result[0..];
        }

        /// Automatically generate Python method name from type name
        /// e.g., zignal.Rgb -> "to_rgb", zignal.Oklab -> "to_oklab"
        fn getConversionMethodName(comptime TargetColorType: type) []const u8 {
            const type_name = @typeName(TargetColorType);

            // Find the last dot and take everything after it
            if (std.mem.lastIndexOf(u8, type_name, ".")) |dot_index| {
                const base_name = type_name[dot_index + 1 ..];
                return "to_" ++ comptimeLowercase(base_name);
            } else {
                @compileError("Expected zignal.ColorName format, got: " ++ type_name);
            }
        }

        /// Automatically generate method name from type name
        /// e.g., zignal.Rgb -> "toRgb", zignal.Oklab -> "toOklab"
        fn getZigConversionMethodName(comptime TargetColorType: type) []const u8 {
            const type_name = @typeName(TargetColorType);

            // Find the last dot and take everything after it
            if (std.mem.lastIndexOf(u8, type_name, ".")) |dot_index| {
                const base_name = type_name[dot_index + 1 ..];
                return "to" ++ base_name;
            } else {
                @compileError("Expected zignal.ColorName format, got: " ++ type_name);
            }
        }

        /// Automatically generate documentation from type name
        fn getConversionMethodDoc(comptime TargetColorType: type) []const u8 {
            const type_name = @typeName(TargetColorType);

            // Extract the color space name (everything after the last dot)
            if (std.mem.lastIndexOf(u8, type_name, ".")) |dot_index| {
                const color_space = type_name[dot_index + 1 ..];
                return "Convert to " ++ color_space ++ " color space";
            } else {
                @compileError("Expected zignal.ColorName format, got: " ++ type_name);
            }
        }

        /// Generic conversion method generator - creates specific methods for each target type
        fn generateConversionMethod(comptime TargetColorType: type) fn ([*c]c.PyObject, [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
            return switch (TargetColorType) {
                // Special case for Rgba due to default alpha parameter
                zignal.Rgba => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toRgba(255); // Default alpha 255
                        return @ptrCast(createColorPyObject(result));
                    }
                }.method,
                // All other color types use the same pattern with automatic method name generation
                else => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const method_name = comptime getZigConversionMethodName(TargetColorType);
                        const result = @field(ZigColorType, method_name)(zig_color);
                        return @ptrCast(createColorPyObject(result));
                    }
                }.method,
            };
        }

        /// Convert Python object to Zig color
        fn objectToZigColor(obj: *ObjectType) ZigColorType {
            var zig_color: ZigColorType = undefined;

            inline for (fields, 0..) |field, i| {
                const field_value = switch (i) {
                    0 => obj.field0,
                    1 => obj.field1,
                    2 => obj.field2,
                    3 => obj.field3,
                    else => unreachable,
                };
                @field(zig_color, field.name) = field_value;
            }

            return zig_color;
        }

        /// Convert Zig color to Python object fields
        fn zigColorToObject(zig_color: ZigColorType, obj: *ObjectType) void {
            inline for (fields, 0..) |field, i| {
                const field_value = @field(zig_color, field.name);
                switch (i) {
                    0 => obj.field0 = field_value,
                    1 => obj.field1 = field_value,
                    2 => obj.field2 = field_value,
                    3 => obj.field3 = field_value,
                    else => unreachable,
                }
            }
        }

        /// Custom init function with validation
        pub fn init(self_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) c_int {
            _ = kwds;
            const self = @as(*ObjectType, @ptrCast(self_obj));

            // Parse arguments based on field count
            switch (fields.len) {
                1 => {
                    var arg0: fields[0].type = undefined;
                    const format = comptime blk: {
                        const fmt = std.fmt.comptimePrint("{s}", .{getFormatString(fields[0].type)});
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    if (!validateColorComponent(ZigColorType, fields[0].name ++ "", arg0)) {
                        const error_msg = getValidationErrorMessage(ZigColorType);
                        c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                        return -1;
                    }

                    self.field0 = arg0;
                },
                2 => {
                    var arg0: fields[0].type = undefined;
                    var arg1: fields[1].type = undefined;
                    const format = comptime blk: {
                        const fmt = std.fmt.comptimePrint("{s}{s}", .{
                            getFormatString(fields[0].type),
                            getFormatString(fields[1].type),
                        });
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    if (!validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                        !validateColorComponent(ZigColorType, fields[1].name ++ "", arg1))
                    {
                        const error_msg = getValidationErrorMessage(ZigColorType);
                        c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                        return -1;
                    }

                    self.field0 = arg0;
                    self.field1 = arg1;
                },
                3 => {
                    // For validation, we need to parse as larger types to avoid overflow
                    if (fields[0].type == u8) {
                        // RGB case: parse as int to avoid u8 wrapping, then validate
                        var arg0: c_int = undefined;
                        var arg1: c_int = undefined;
                        var arg2: c_int = undefined;
                        const format = comptime std.fmt.comptimePrint("iii", .{});
                        if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1, &arg2) == 0) {
                            return -1;
                        }

                        // Use generic validation with the actual color type
                        if (!validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                            !validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                            !validateColorComponent(ZigColorType, fields[2].name ++ "", arg2))
                        {
                            const error_msg = getValidationErrorMessage(ZigColorType);
                            c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                            return -1;
                        }

                        self.field0 = @intCast(arg0);
                        self.field1 = @intCast(arg1);
                        self.field2 = @intCast(arg2);
                    } else {
                        // Non-u8 case: use field types directly
                        var arg0: fields[0].type = undefined;
                        var arg1: fields[1].type = undefined;
                        var arg2: fields[2].type = undefined;

                        // Use std.fmt.comptimePrint for proper null termination
                        if (fields[0].type == f64 and fields[1].type == f64 and fields[2].type == f64) {
                            // HSV case: all f64 fields
                            const format = comptime std.fmt.comptimePrint("ddd", .{});
                            if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1, &arg2) == 0) {
                                return -1;
                            }
                        } else if (fields[0].type == f32 and fields[1].type == f32 and fields[2].type == f32) {
                            // f32 case: all f32 fields
                            const format = comptime std.fmt.comptimePrint("fff", .{});
                            if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1, &arg2) == 0) {
                                return -1;
                            }
                        } else {
                            // Fallback: build format string character by character
                            var format_buf: [4]u8 = undefined;
                            format_buf[0] = getFormatString(fields[0].type)[0];
                            format_buf[1] = getFormatString(fields[1].type)[0];
                            format_buf[2] = getFormatString(fields[2].type)[0];
                            format_buf[3] = 0; // null terminator

                            if (c.PyArg_ParseTuple(args, &format_buf, &arg0, &arg1, &arg2) == 0) {
                                return -1;
                            }
                        }

                        // Use generic validation with the actual color type
                        if (!validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                            !validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                            !validateColorComponent(ZigColorType, fields[2].name ++ "", arg2))
                        {
                            const error_msg = getValidationErrorMessage(ZigColorType);
                            c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                            return -1;
                        }

                        self.field0 = arg0;
                        self.field1 = arg1;
                        self.field2 = arg2;
                    }
                },
                4 => {
                    var arg0: fields[0].type = undefined;
                    var arg1: fields[1].type = undefined;
                    var arg2: fields[2].type = undefined;
                    var arg3: fields[3].type = undefined;
                    const format = comptime blk: {
                        const fmt = std.fmt.comptimePrint("{s}{s}{s}{s}", .{
                            getFormatString(fields[0].type),
                            getFormatString(fields[1].type),
                            getFormatString(fields[2].type),
                            getFormatString(fields[3].type),
                        });
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1, &arg2, &arg3) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    if (!validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                        !validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                        !validateColorComponent(ZigColorType, fields[2].name ++ "", arg2) or
                        !validateColorComponent(ZigColorType, fields[3].name ++ "", arg3))
                    {
                        const error_msg = getValidationErrorMessage(ZigColorType);
                        c.PyErr_SetString(c.PyExc_ValueError, error_msg.ptr);
                        return -1;
                    }

                    self.field0 = arg0;
                    self.field1 = arg1;
                    self.field2 = arg2;
                    self.field3 = arg3;
                },
                else => unreachable,
            }

            return 0;
        }

        /// Standard Python object methods
        pub fn dealloc(self_obj: [*c]c.PyObject) callconv(.c) void {
            c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
        }

        pub fn new(type_obj: [*c]c.PyTypeObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) ?*c.PyObject {
            _ = args;
            _ = kwds;

            const self = @as(?*ObjectType, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
            if (self) |obj| {
                // Initialize fields to zero
                obj.field0 = std.mem.zeroes(fields[0].type);
                if (fields.len > 1) obj.field1 = std.mem.zeroes(fields[1].type);
                if (fields.len > 2) obj.field2 = std.mem.zeroes(fields[2].type);
                if (fields.len > 3) obj.field3 = std.mem.zeroes(fields[3].type);
            }
            return @ptrCast(self);
        }

        pub fn repr(self_obj: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
            const self = @as(*ObjectType, @ptrCast(self_obj));

            var buffer: [128]u8 = undefined;
            const formatted = switch (fields.len) {
                1 => std.fmt.bufPrintZ(&buffer, "{s}({s}={})", .{ name, fields[0].name, self.field0 }) catch return null,
                2 => std.fmt.bufPrintZ(&buffer, "{s}({s}={}, {s}={})", .{ name, fields[0].name, self.field0, fields[1].name, self.field1 }) catch return null,
                3 => std.fmt.bufPrintZ(&buffer, "{s}({s}={}, {s}={}, {s}={})", .{ name, fields[0].name, self.field0, fields[1].name, self.field1, fields[2].name, self.field2 }) catch return null,
                4 => std.fmt.bufPrintZ(&buffer, "{s}({s}={}, {s}={}, {s}={}, {s}={})", .{ name, fields[0].name, self.field0, fields[1].name, self.field1, fields[2].name, self.field2, fields[3].name, self.field3 }) catch return null,
                else => unreachable,
            };

            return @ptrCast(c.PyUnicode_FromString(formatted.ptr));
        }
    };
}
