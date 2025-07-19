const std = @import("std");

const zignal = @import("zignal");

const py_utils = @import("py_utils.zig");
pub const registerType = py_utils.registerType;

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

        // Generate property getters
        pub fn generateGetters() [fields.len + 1]c.PyGetSetDef {
            var getters: [fields.len + 1]c.PyGetSetDef = undefined;

            // Generate getter for each field
            inline for (fields, 0..) |field, i| {
                getters[i] = c.PyGetSetDef{
                    .name = field.name ++ "",
                    .get = generateFieldGetter(i),
                    .set = null,
                    .doc = field.name ++ " component",
                    .closure = null,
                };
            }

            // Null terminator
            getters[fields.len] = c.PyGetSetDef{
                .name = null,
                .get = null,
                .set = null,
                .doc = null,
                .closure = null,
            };

            return getters;
        }

        // Generate field getter for specific field index
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

                    return @ptrCast(@alignCast(py_utils.convertToPythonCast(value)));
                }
            }.getter;
        }

        // Generate methods array - automatically create conversion methods for all color types
        pub fn generateMethods() [getMethodCount() + 1]c.PyMethodDef {
            var methods: [getMethodCount() + 1]c.PyMethodDef = undefined;
            var index: usize = 0;

            const color_registry = @import("color_registry.zig");

            // Generate conversion methods for each color type
            inline for (color_registry.color_types) |TargetColorType| {
                const method_name = getConversionMethodName(TargetColorType);
                const zig_method_name = getZigConversionMethodName(TargetColorType);

                // Only add method if the Zig type has this conversion method
                if (@hasDecl(ZigColorType, zig_method_name)) {
                    methods[index] = c.PyMethodDef{
                        .ml_name = method_name.ptr,
                        .ml_meth = generateConversionMethod(TargetColorType),
                        .ml_flags = c.METH_NOARGS,
                        .ml_doc = getConversionMethodDoc(TargetColorType).ptr,
                    };
                    index += 1;
                }
            }

            // Null terminator
            methods[index] = c.PyMethodDef{
                .ml_name = null,
                .ml_meth = null,
                .ml_flags = 0,
                .ml_doc = null,
            };

            return methods;
        }

        // Count available methods - automatically count conversion methods for all color types
        fn getMethodCount() comptime_int {
            var count: comptime_int = 0;
            const color_registry = @import("color_registry.zig");

            inline for (color_registry.color_types) |TargetColorType| {
                const zig_method_name = getZigConversionMethodName(TargetColorType);
                if (@hasDecl(ZigColorType, zig_method_name)) {
                    count += 1;
                }
            }
            return count;
        }

        // Helper functions for method name generation
        fn getConversionMethodName(comptime TargetColorType: type) []const u8 {
            return switch (TargetColorType) {
                zignal.Rgb => "to_rgb",
                zignal.Rgba => "to_rgba",
                zignal.Hsl => "to_hsl",
                zignal.Hsv => "to_hsv",
                zignal.Lab => "to_lab",
                zignal.Lch => "to_lch",
                zignal.Lms => "to_lms",
                zignal.Oklab => "to_oklab",
                zignal.Oklch => "to_oklch",
                zignal.Xyb => "to_xyb",
                zignal.Xyz => "to_xyz",
                zignal.Ycbcr => "to_ycbcr",
                else => @compileError("Unknown color type for method name generation"),
            };
        }

        fn getZigConversionMethodName(comptime TargetColorType: type) []const u8 {
            return switch (TargetColorType) {
                zignal.Rgb => "toRgb",
                zignal.Rgba => "toRgba",
                zignal.Hsl => "toHsl",
                zignal.Hsv => "toHsv",
                zignal.Lab => "toLab",
                zignal.Lch => "toLch",
                zignal.Lms => "toLms",
                zignal.Oklab => "toOklab",
                zignal.Oklch => "toOklch",
                zignal.Xyb => "toXyb",
                zignal.Xyz => "toXyz",
                zignal.Ycbcr => "toYcbcr",
                else => @compileError("Unknown color type for Zig method name generation"),
            };
        }

        fn getConversionMethodDoc(comptime TargetColorType: type) []const u8 {
            return switch (TargetColorType) {
                zignal.Rgb => "Convert to RGB color space",
                zignal.Rgba => "Convert to RGBA color space with alpha",
                zignal.Hsl => "Convert to HSL color space",
                zignal.Hsv => "Convert to HSV color space",
                zignal.Lab => "Convert to CIELAB color space",
                zignal.Lch => "Convert to CIE LCH color space",
                zignal.Lms => "Convert to LMS cone response space",
                zignal.Oklab => "Convert to Oklab perceptual color space",
                zignal.Oklch => "Convert to Oklch perceptual color space",
                zignal.Xyb => "Convert to XYB color space",
                zignal.Xyz => "Convert to CIE XYZ color space",
                zignal.Ycbcr => "Convert to YCbCr color space",
                else => @compileError("Unknown color type for documentation generation"),
            };
        }

        // Generic conversion method generator - creates specific methods for each target type
        fn generateConversionMethod(comptime TargetColorType: type) fn ([*c]c.PyObject, [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
            return switch (TargetColorType) {
                zignal.Rgb => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toRgb();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Rgba => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toRgba(255); // Default alpha 255
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Hsl => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toHsl();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Hsv => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toHsv();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Lab => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toLab();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Lch => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toLch();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Lms => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toLms();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Oklab => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toOklab();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Oklch => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toOklch();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Xyb => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toXyb();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Xyz => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toXyz();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                zignal.Ycbcr => struct {
                    fn method(self_obj: [*c]c.PyObject, _: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
                        const self = @as(*ObjectType, @ptrCast(self_obj));
                        const zig_color = objectToZigColor(self);
                        const result = zig_color.toYcbcr();
                        const color = @import("color.zig");
                        return @ptrCast(color.createPyObject(result));
                    }
                }.method,
                else => @compileError("Unknown color type for method generation"),
            };
        }

        // Convert Python object to Zig color
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

        // Convert Zig color to Python object fields
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

        // Custom init function with validation
        pub fn init(self_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) c_int {
            _ = kwds;
            const self = @as(*ObjectType, @ptrCast(self_obj));

            // Parse arguments based on field count
            switch (fields.len) {
                1 => {
                    var arg0: fields[0].type = undefined;
                    const format = comptime blk: {
                        const fmt = std.fmt.comptimePrint("{s}", .{py_utils.getFormatString(fields[0].type)});
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    const color_registry = @import("color_registry.zig");
                    if (!color_registry.validateColorComponent(ZigColorType, fields[0].name ++ "", arg0)) {
                        const error_msg = color_registry.getValidationErrorMessage(ZigColorType);
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
                            py_utils.getFormatString(fields[0].type),
                            py_utils.getFormatString(fields[1].type),
                        });
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    const color_registry = @import("color_registry.zig");
                    if (!color_registry.validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                        !color_registry.validateColorComponent(ZigColorType, fields[1].name ++ "", arg1))
                    {
                        const error_msg = color_registry.getValidationErrorMessage(ZigColorType);
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
                        const color_registry = @import("color_registry.zig");
                        if (!color_registry.validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                            !color_registry.validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                            !color_registry.validateColorComponent(ZigColorType, fields[2].name ++ "", arg2))
                        {
                            const error_msg = color_registry.getValidationErrorMessage(ZigColorType);
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
                            format_buf[0] = py_utils.getFormatString(fields[0].type)[0];
                            format_buf[1] = py_utils.getFormatString(fields[1].type)[0];
                            format_buf[2] = py_utils.getFormatString(fields[2].type)[0];
                            format_buf[3] = 0; // null terminator

                            if (c.PyArg_ParseTuple(args, &format_buf, &arg0, &arg1, &arg2) == 0) {
                                return -1;
                            }
                        }

                        // Use generic validation with the actual color type
                        const color_registry = @import("color_registry.zig");
                        if (!color_registry.validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                            !color_registry.validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                            !color_registry.validateColorComponent(ZigColorType, fields[2].name ++ "", arg2))
                        {
                            const error_msg = color_registry.getValidationErrorMessage(ZigColorType);
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
                            py_utils.getFormatString(fields[0].type),
                            py_utils.getFormatString(fields[1].type),
                            py_utils.getFormatString(fields[2].type),
                            py_utils.getFormatString(fields[3].type),
                        });
                        break :blk fmt ++ ""; // Ensure null termination
                    };
                    if (c.PyArg_ParseTuple(args, format.ptr, &arg0, &arg1, &arg2, &arg3) == 0) {
                        return -1;
                    }

                    // Use generic validation with the actual color type
                    const color_registry = @import("color_registry.zig");
                    if (!color_registry.validateColorComponent(ZigColorType, fields[0].name ++ "", arg0) or
                        !color_registry.validateColorComponent(ZigColorType, fields[1].name ++ "", arg1) or
                        !color_registry.validateColorComponent(ZigColorType, fields[2].name ++ "", arg2) or
                        !color_registry.validateColorComponent(ZigColorType, fields[3].name ++ "", arg3))
                    {
                        const error_msg = color_registry.getValidationErrorMessage(ZigColorType);
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

        // Standard Python object methods
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
