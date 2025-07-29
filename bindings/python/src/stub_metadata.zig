// Metadata types for automatic Python stub generation
// This file defines structures that describe Python bindings in a way
// that can be introspected at compile time for stub generation

const std = @import("std");

/// Describes a Python method for stub generation
pub const MethodInfo = struct {
    /// Method name as it appears in Python
    name: []const u8,
    /// Method signature (parameters with type hints)
    /// Examples: "self, path: str", "cls, array: np.ndarray"
    params: []const u8,
    /// Return type annotation
    /// Examples: "None", "Image", "Tuple[int, int]"
    returns: []const u8,
    /// Method flags
    is_classmethod: bool = false,
    is_staticmethod: bool = false,
    /// Documentation string (optional)
    doc: ?[]const u8 = null,
};

/// Describes a Python property for stub generation
pub const PropertyInfo = struct {
    /// Property name
    name: []const u8,
    /// Property type annotation
    /// Examples: "int", "str", "float"
    type: []const u8,
    /// Whether the property is read-only
    readonly: bool = true,
    /// Documentation string (optional)
    doc: ?[]const u8 = null,
};

/// Describes a Python class for stub generation
pub const ClassInfo = struct {
    /// Class name
    name: []const u8,
    /// Class docstring
    doc: []const u8,
    /// List of methods
    methods: []const MethodInfo,
    /// List of properties
    properties: []const PropertyInfo,
    /// Base classes (optional)
    bases: []const []const u8 = &.{},
};

/// Describes a module-level function
pub const FunctionInfo = struct {
    /// Function name
    name: []const u8,
    /// Function signature (parameters with type hints)
    params: []const u8,
    /// Return type annotation
    returns: []const u8,
    /// Documentation string
    doc: []const u8,
};

/// Describes an enum for stub generation
pub const EnumInfo = struct {
    /// Enum name
    name: []const u8,
    /// Base class (usually IntEnum)
    base: []const u8 = "IntEnum",
    /// Documentation string
    doc: []const u8,
    /// Zig type to extract values from
    zig_type: type,
};

/// Complete module metadata
pub const ModuleInfo = struct {
    /// Module-level functions
    functions: []const FunctionInfo = &.{},
    /// Classes defined in the module
    classes: []const ClassInfo = &.{},
    /// Enums defined in the module
    enums: []const EnumInfo = &.{},
};

/// Helper to create method info for common patterns
pub fn method(
    name: []const u8,
    params: []const u8,
    returns: []const u8,
) MethodInfo {
    return .{
        .name = name,
        .params = params,
        .returns = returns,
    };
}

pub fn classmethod(
    name: []const u8,
    params: []const u8,
    returns: []const u8,
) MethodInfo {
    return .{
        .name = name,
        .params = params,
        .returns = returns,
        .is_classmethod = true,
    };
}

pub fn staticmethod(
    name: []const u8,
    params: []const u8,
    returns: []const u8,
) MethodInfo {
    return .{
        .name = name,
        .params = params,
        .returns = returns,
        .is_staticmethod = true,
    };
}

pub fn property(
    name: []const u8,
    prop_type: []const u8,
) PropertyInfo {
    return .{
        .name = name,
        .type = prop_type,
    };
}

pub fn readonly_property(
    name: []const u8,
    prop_type: []const u8,
) PropertyInfo {
    return .{
        .name = name,
        .type = prop_type,
        .readonly = true,
    };
}
