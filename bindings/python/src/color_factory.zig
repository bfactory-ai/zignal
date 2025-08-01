const std = @import("std");

const zignal = @import("zignal");
const isPacked = zignal.meta.isPacked;

const c = @import("py_utils.zig").c;
const color_types = @import("color_registry.zig").color_types;
const ConversionError = @import("py_utils.zig").ConversionError;
const convertFromPython = @import("py_utils.zig").convertFromPython;
const convertToPython = @import("py_utils.zig").convertToPython;
const convertWithValidation = @import("py_utils.zig").convertWithValidation;
const createColorPyObject = @import("color.zig").createColorPyObject;
const getFormatString = @import("py_utils.zig").getFormatString;
const getValidationErrorMessage = @import("color_registry.zig").getValidationErrorMessage;
const isSupportedColor = @import("color_registry.zig").isSupportedColor;
const registerType = @import("py_utils.zig").registerType;
const validateColorComponent = @import("color_registry.zig").validateColorComponent;

/// Convert string to lowercase at comptime
pub fn comptimeLowercase(comptime input: []const u8) []const u8 {
    comptime var result: [input.len]u8 = undefined;
    inline for (input, 0..) |char, i| {
        result[i] = std.ascii.toLower(char);
    }
    return result[0..];
}

/// Automatically generate documentation from type name for color conversion methods
pub fn getConversionMethodDoc(comptime TargetColorType: type) []const u8 {
    const type_name = @typeName(TargetColorType);

    // Extract the color space name (everything after the last dot)
    if (comptime std.mem.lastIndexOf(u8, type_name, ".")) |dot_index| {
        const color_space = comptime type_name[dot_index + 1 ..];
        return comptime "Convert to `" ++ color_space ++ "` color space.";
    } else {
        @compileError("Expected zignal.ColorName format, got: " ++ type_name);
    }
}

/// Generate a color binding with automatic property getters and validation
pub fn createColorBinding(
    comptime name: []const u8,
    comptime ZigColorType: type,
) type {
    const fields = @typeInfo(ZigColorType).@"struct".fields;
    const is_packed = isPacked(ZigColorType);

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
                    const field_name = field.name;

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

        /// Generate methods array - automatically create conversion methods for all color types + __format__
        pub fn generateMethods() [color_types.len + 1]c.PyMethodDef {
            var methods: [color_types.len + 1]c.PyMethodDef = undefined;
            var index: usize = 0;

            // Add __format__ method
            methods[index] = c.PyMethodDef{
                .ml_name = "__format__",
                .ml_meth = @ptrCast(&formatMethod),
                .ml_flags = c.METH_VARARGS,
                .ml_doc = "Format the color object with optional format specifier (e.g., 'ansi' for terminal colors)",
            };
            index += 1;

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
            if (comptime is_packed) {
                // For packed structs, create a temporary unpacked representation
                // then use @bitCast to convert to packed layout
                return switch (fields.len) {
                    4 => blk: {
                        // Create array of bytes for packed struct
                        const bytes = [4]u8{ obj.field0, obj.field1, obj.field2, obj.field3 };
                        break :blk @bitCast(bytes);
                    },
                    3 => blk: {
                        // Create array of bytes for packed struct
                        const bytes = [3]u8{ obj.field0, obj.field1, obj.field2 };
                        break :blk @bitCast(bytes);
                    },
                    2 => blk: {
                        // Create array of bytes for packed struct
                        const bytes = [2]u8{ obj.field0, obj.field1 };
                        break :blk @bitCast(bytes);
                    },
                    1 => blk: {
                        // Single field packed struct
                        break :blk @bitCast([1]u8{obj.field0});
                    },
                    else => @compileError("Unsupported field count for packed struct"),
                };
            } else {
                // Regular struct - use field-by-field assignment
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
        }

        /// Convert Zig color to Python object fields
        fn zigColorToObject(zig_color: ZigColorType, obj: *ObjectType) void {
            if (comptime is_packed) {
                // For packed structs, convert to byte array using @bitCast
                const bytes: [fields.len]u8 = @bitCast(zig_color);
                switch (fields.len) {
                    4 => {
                        obj.field0 = bytes[0];
                        obj.field1 = bytes[1];
                        obj.field2 = bytes[2];
                        obj.field3 = bytes[3];
                    },
                    3 => {
                        obj.field0 = bytes[0];
                        obj.field1 = bytes[1];
                        obj.field2 = bytes[2];
                    },
                    2 => {
                        obj.field0 = bytes[0];
                        obj.field1 = bytes[1];
                    },
                    1 => {
                        obj.field0 = bytes[0];
                    },
                    else => @compileError("Unsupported field count for packed struct"),
                }
            } else {
                // Regular struct - use field-by-field assignment
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
        }

        /// Helper function to convert Python object to field type with color validation
        fn convertArgument(comptime T: type, py_obj: ?*c.PyObject, field_name: []const u8) !T {
            const validator = struct {
                fn validate(field_name_inner: []const u8, value: anytype) bool {
                    return validateColorComponent(ZigColorType, field_name_inner, value);
                }
            }.validate;

            const error_msg = getValidationErrorMessage(ZigColorType);
            return convertWithValidation(T, @ptrCast(py_obj), field_name, validator, error_msg);
        }

        /// Custom init function with validation
        pub fn init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
            _ = kwds;

            // Handle null pointers using Zig's type system
            const self = @as(*ObjectType, @ptrCast(self_obj orelse {
                c.PyErr_SetString(c.PyExc_SystemError, "self object is null");
                return -1;
            }));

            const args_tuple = args orelse {
                c.PyErr_SetString(c.PyExc_SystemError, "args object is null");
                return -1;
            };

            // Parse arguments based on field count
            switch (fields.len) {
                1 => {
                    // Parse as Python object for consistent type safety
                    var py_arg0: ?*c.PyObject = null;
                    if (c.PyArg_ParseTuple(args_tuple, "O", &py_arg0) == 0) {
                        return -1;
                    }

                    // Convert using helper function with integrated validation
                    const arg0 = convertArgument(fields[0].type, py_arg0, fields[0].name) catch {
                        return -1;
                    };

                    self.field0 = arg0;
                },
                2 => {
                    // Parse as Python objects for consistent type safety
                    var py_arg0: ?*c.PyObject = null;
                    var py_arg1: ?*c.PyObject = null;
                    if (c.PyArg_ParseTuple(args_tuple, "OO", &py_arg0, &py_arg1) == 0) {
                        return -1;
                    }

                    // Convert using helper function with integrated validation
                    const arg0 = convertArgument(fields[0].type, py_arg0, fields[0].name) catch {
                        return -1;
                    };
                    const arg1 = convertArgument(fields[1].type, py_arg1, fields[1].name) catch {
                        return -1;
                    };

                    self.field0 = arg0;
                    self.field1 = arg1;
                },
                3 => {
                    // Parse as Python objects for consistent type safety
                    var py_arg0: ?*c.PyObject = null;
                    var py_arg1: ?*c.PyObject = null;
                    var py_arg2: ?*c.PyObject = null;
                    if (c.PyArg_ParseTuple(args_tuple, "OOO", &py_arg0, &py_arg1, &py_arg2) == 0) {
                        return -1;
                    }

                    // Convert using helper function with integrated validation
                    const arg0 = convertArgument(fields[0].type, py_arg0, fields[0].name) catch {
                        return -1;
                    };
                    const arg1 = convertArgument(fields[1].type, py_arg1, fields[1].name) catch {
                        return -1;
                    };
                    const arg2 = convertArgument(fields[2].type, py_arg2, fields[2].name) catch {
                        return -1;
                    };

                    self.field0 = arg0;
                    self.field1 = arg1;
                    self.field2 = arg2;
                },
                4 => {
                    // Parse as Python objects for consistent type safety
                    var py_arg0: ?*c.PyObject = null;
                    var py_arg1: ?*c.PyObject = null;
                    var py_arg2: ?*c.PyObject = null;
                    var py_arg3: ?*c.PyObject = null;
                    if (c.PyArg_ParseTuple(args_tuple, "OOOO", &py_arg0, &py_arg1, &py_arg2, &py_arg3) == 0) {
                        return -1;
                    }

                    // Convert using helper function with integrated validation
                    const arg0 = convertArgument(fields[0].type, py_arg0, fields[0].name) catch {
                        return -1;
                    };
                    const arg1 = convertArgument(fields[1].type, py_arg1, fields[1].name) catch {
                        return -1;
                    };
                    const arg2 = convertArgument(fields[2].type, py_arg2, fields[2].name) catch {
                        return -1;
                    };
                    const arg3 = convertArgument(fields[3].type, py_arg3, fields[3].name) catch {
                        return -1;
                    };

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

        /// __format__ method implementation
        pub fn formatMethod(self_obj: [*c]c.PyObject, args: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
            const self = @as(*ObjectType, @ptrCast(self_obj));

            // Parse format_spec argument
            var format_spec: [*c]const u8 = undefined;
            const format = std.fmt.comptimePrint("s", .{});
            if (c.PyArg_ParseTuple(args, format.ptr, &format_spec) == 0) {
                return null;
            }

            // Convert C string to Zig slice
            const format_str = std.mem.span(format_spec);
            if (std.mem.eql(u8, format_str, "ansi")) {
                // Convert to Zig color
                const zig_color = objectToZigColor(self);

                // Use zignal's public API
                const convertColor = zignal.convertColor;
                const Rgb = zignal.Rgb;
                const Oklab = zignal.Oklab;

                // Convert to RGB for ANSI display
                const rgb = convertColor(Rgb, zig_color);

                // Determine text color based on background darkness
                const fg: u8 = if (convertColor(Oklab, rgb).l < 0.5) 255 else 0;

                // Build ANSI formatted string with Python-style repr
                var buffer: [512]u8 = undefined;
                var stream = std.io.fixedBufferStream(&buffer);
                const writer = stream.writer();

                // Start with ANSI escape codes and type name
                writer.print(
                    "\x1b[1m\x1b[38;2;{d};{d};{d}m\x1b[48;2;{d};{d};{d}m{s}(",
                    .{ fg, fg, fg, rgb.r, rgb.g, rgb.b, name },
                ) catch return null;

                // Print each field in Python style (field=value)
                inline for (fields, 0..) |field, i| {
                    writer.print("{s}=", .{field.name}) catch return null;

                    // Format the field value appropriately
                    const value = switch (i) {
                        0 => self.field0,
                        1 => self.field1,
                        2 => self.field2,
                        3 => self.field3,
                        else => unreachable,
                    };

                    switch (field.type) {
                        u8 => writer.print("{d}", .{value}) catch return null,
                        f64 => writer.print("{d}", .{value}) catch return null,
                        else => writer.print("{any}", .{value}) catch return null,
                    }

                    if (i < fields.len - 1) {
                        writer.print(", ", .{}) catch return null;
                    }
                }

                // Close parenthesis and reset ANSI codes
                writer.print(")\x1b[0m", .{}) catch return null;

                const formatted = stream.getWritten();
                return @ptrCast(c.PyUnicode_FromStringAndSize(formatted.ptr, @intCast(formatted.len)));
            } else if (format_str.len == 0) {
                // Empty format spec - use repr
                return repr(self_obj);
            } else {
                // Unknown format spec
                _ = c.PyErr_Format(c.PyExc_ValueError, "Unknown format code '%s' for object of type '%s'", format_spec, name.ptr);
                return null;
            }
        }
    };
}
