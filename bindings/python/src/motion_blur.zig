const std = @import("std");
const zignal = @import("zignal");

const py_utils = @import("py_utils.zig");
const c = py_utils.c;
const stub_metadata = @import("stub_metadata.zig");

// ============================================================================
// Motion Blur Type Enum
// ============================================================================

const MotionBlurVariant = enum(u8) {
    linear,
    radial_zoom,
    radial_spin,
};

// ============================================================================
// Unified MotionBlur Object
// ============================================================================

pub const MotionBlurObject = extern struct {
    ob_base: c.PyObject,
    blur_type: MotionBlurVariant,

    // Linear parameters
    angle: f64,
    distance: c_long,

    // Radial parameters (shared by zoom and spin)
    center_x: f64,
    center_y: f64,
    strength: f64,
};

// ============================================================================
// MotionBlur Implementation
// ============================================================================

fn motion_blur_dealloc(self: [*c]c.PyObject) callconv(.c) void {
    c.Py_TYPE(self).*.tp_free.?(self);
}

fn motion_blur_new(type_obj: [*c]c.PyTypeObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*MotionBlurObject, @ptrCast(type_obj.*.tp_alloc.?(type_obj, 0)));
    if (self) |obj| {
        // Initialize with defaults
        obj.blur_type = .linear;
        obj.angle = 0.0;
        obj.distance = 0;
        obj.center_x = 0.5;
        obj.center_y = 0.5;
        obj.strength = 0.5;
    }
    return @ptrCast(self);
}

fn motion_blur_init(self_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) c_int {
    _ = self_obj;
    _ = args;
    _ = kwds;
    // This should not be called directly - factory methods handle initialization
    c.PyErr_SetString(c.PyExc_TypeError, "MotionBlur cannot be instantiated directly. Use MotionBlur.linear(), MotionBlur.radial_zoom(), or MotionBlur.radial_spin()");
    return -1;
}

fn motion_blur_repr(self_obj: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
    const self = @as(*MotionBlurObject, @ptrCast(self_obj));
    var buf: [256]u8 = undefined;

    const str = switch (self.blur_type) {
        .linear => std.fmt.bufPrintZ(
            &buf,
            "MotionBlur.linear(angle={d:.4}, distance={d})",
            .{ self.angle, self.distance },
        ) catch return null,
        .radial_zoom => std.fmt.bufPrintZ(
            &buf,
            "MotionBlur.radial_zoom(center=({d:.3}, {d:.3}), strength={d:.3})",
            .{ self.center_x, self.center_y, self.strength },
        ) catch return null,
        .radial_spin => std.fmt.bufPrintZ(
            &buf,
            "MotionBlur.radial_spin(center=({d:.3}, {d:.3}), strength={d:.3})",
            .{ self.center_x, self.center_y, self.strength },
        ) catch return null,
    };

    return @ptrCast(c.PyUnicode_FromString(str));
}

// ============================================================================
// Static Factory Methods
// ============================================================================

fn motion_blur_linear(type_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
    _ = type_obj;

    var angle: f64 = 0.0;
    var distance: c_long = 0;

    var kwlist = [_:null]?[*:0]u8{ @constCast("angle"), @constCast("distance"), null };
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, "dl", @ptrCast(&kwlist), &angle, &distance) == 0) {
        return null;
    }

    if (distance < 0) {
        c.PyErr_SetString(c.PyExc_ValueError, "distance must be non-negative");
        return null;
    }

    // Create new instance
    const self = @as(?*MotionBlurObject, @ptrCast(MotionBlurType.tp_alloc.?(&MotionBlurType, 0)));
    if (self) |obj| {
        obj.blur_type = .linear;
        obj.angle = angle;
        obj.distance = distance;
        // Other fields keep their defaults
        obj.center_x = 0.5;
        obj.center_y = 0.5;
        obj.strength = 0.5;
    }

    return @ptrCast(self);
}

fn motion_blur_radial_zoom(type_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
    _ = type_obj;

    var center_tuple: [*c]c.PyObject = null;
    var strength: f64 = 0.5;

    var kwlist = [_:null]?[*:0]u8{ @constCast("center"), @constCast("strength"), null };
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, "|Od", @ptrCast(&kwlist), &center_tuple, &strength) == 0) {
        return null;
    }

    var center_x: f64 = 0.5;
    var center_y: f64 = 0.5;

    // Parse center tuple if provided
    if (center_tuple != null) {
        if (c.PyArg_ParseTuple(center_tuple, "dd", &center_x, &center_y) == 0) {
            c.PyErr_SetString(c.PyExc_TypeError, "center must be a tuple of two floats");
            return null;
        }

        if (center_x < 0.0 or center_x > 1.0) {
            c.PyErr_SetString(c.PyExc_ValueError, "center[0] must be between 0.0 and 1.0");
            return null;
        }
        if (center_y < 0.0 or center_y > 1.0) {
            c.PyErr_SetString(c.PyExc_ValueError, "center[1] must be between 0.0 and 1.0");
            return null;
        }
    }

    if (strength < 0.0 or strength > 1.0) {
        c.PyErr_SetString(c.PyExc_ValueError, "strength must be between 0.0 and 1.0");
        return null;
    }

    // Create new instance
    const self = @as(?*MotionBlurObject, @ptrCast(MotionBlurType.tp_alloc.?(&MotionBlurType, 0)));
    if (self) |obj| {
        obj.blur_type = .radial_zoom;
        obj.center_x = center_x;
        obj.center_y = center_y;
        obj.strength = strength;
        // Linear fields keep their defaults
        obj.angle = 0.0;
        obj.distance = 0;
    }

    return @ptrCast(self);
}

fn motion_blur_radial_spin(type_obj: [*c]c.PyObject, args: [*c]c.PyObject, kwds: [*c]c.PyObject) callconv(.c) [*c]c.PyObject {
    _ = type_obj;

    var center_tuple: [*c]c.PyObject = null;
    var strength: f64 = 0.5;

    var kwlist = [_:null]?[*:0]u8{ @constCast("center"), @constCast("strength"), null };
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, "|Od", @ptrCast(&kwlist), &center_tuple, &strength) == 0) {
        return null;
    }

    var center_x: f64 = 0.5;
    var center_y: f64 = 0.5;

    // Parse center tuple if provided
    if (center_tuple != null) {
        if (c.PyArg_ParseTuple(center_tuple, "dd", &center_x, &center_y) == 0) {
            c.PyErr_SetString(c.PyExc_TypeError, "center must be a tuple of two floats");
            return null;
        }

        if (center_x < 0.0 or center_x > 1.0) {
            c.PyErr_SetString(c.PyExc_ValueError, "center[0] must be between 0.0 and 1.0");
            return null;
        }
        if (center_y < 0.0 or center_y > 1.0) {
            c.PyErr_SetString(c.PyExc_ValueError, "center[1] must be between 0.0 and 1.0");
            return null;
        }
    }

    if (strength < 0.0 or strength > 1.0) {
        c.PyErr_SetString(c.PyExc_ValueError, "strength must be between 0.0 and 1.0");
        return null;
    }

    // Create new instance
    const self = @as(?*MotionBlurObject, @ptrCast(MotionBlurType.tp_alloc.?(&MotionBlurType, 0)));
    if (self) |obj| {
        obj.blur_type = .radial_spin;
        obj.center_x = center_x;
        obj.center_y = center_y;
        obj.strength = strength;
        // Linear fields keep their defaults
        obj.angle = 0.0;
        obj.distance = 0;
    }

    return @ptrCast(self);
}

// ============================================================================
// Property Getters
// ============================================================================
fn get_type(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*MotionBlurObject, @ptrCast(self_obj.?));
    const type_str = switch (self.blur_type) {
        .linear => "linear",
        .radial_zoom => "radial_zoom",
        .radial_spin => "radial_spin",
    };
    return c.PyUnicode_FromString(type_str);
}

// Predicates for optional properties
fn is_linear(self: *MotionBlurObject) bool {
    return self.blur_type == .linear;
}
fn is_not_linear(self: *MotionBlurObject) bool {
    return self.blur_type != .linear;
}

// ============================================================================
// Type Definition
// ============================================================================

pub const motion_blur_doc =
    \\Motion blur effect configuration.
    \\
    \\Use the static factory methods to create motion blur configurations:
    \\- `MotionBlur.linear(angle, distance)` - Linear motion blur
    \\- `MotionBlur.radial_zoom(center, strength)` - Radial zoom blur
    \\- `MotionBlur.radial_spin(center, strength)` - Radial spin blur
    \\
    \\## Examples
    \\```python
    \\import math
    \\from zignal import Image, MotionBlur
    \\
    \\img = Image.load("photo.jpg")
    \\
    \\# Linear motion blur
    \\horizontal = img.motion_blur(MotionBlur.linear(angle=0, distance=30))
    \\vertical = img.motion_blur(MotionBlur.linear(angle=math.pi/2, distance=20))
    \\
    \\# Radial zoom blur
    \\zoom = img.motion_blur(MotionBlur.radial_zoom(center=(0.5, 0.5), strength=0.7))
    \\zoom_default = img.motion_blur(MotionBlur.radial_zoom())  # Uses defaults
    \\
    \\# Radial spin blur
    \\spin = img.motion_blur(MotionBlur.radial_spin(center=(0.3, 0.7), strength=0.5))
    \\```
;

var motion_blur_methods = [_]c.PyMethodDef{
    .{
        .ml_name = "linear",
        .ml_meth = @ptrCast(&motion_blur_linear),
        .ml_flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_STATIC,
        .ml_doc = "Create linear motion blur configuration.\n\nArgs:\n    angle: Blur angle in radians (0 = horizontal)\n    distance: Blur distance in pixels",
    },
    .{
        .ml_name = "radial_zoom",
        .ml_meth = @ptrCast(&motion_blur_radial_zoom),
        .ml_flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_STATIC,
        .ml_doc = "Create radial zoom blur configuration.\n\nArgs:\n    center: Normalized center (x, y), default (0.5, 0.5)\n    strength: Blur strength (0.0-1.0), default 0.5",
    },
    .{
        .ml_name = "radial_spin",
        .ml_meth = @ptrCast(&motion_blur_radial_spin),
        .ml_flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_STATIC,
        .ml_doc = "Create radial spin blur configuration.\n\nArgs:\n    center: Normalized center (x, y), default (0.5, 0.5)\n    strength: Blur strength (0.0-1.0), default 0.5",
    },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var motion_blur_getset = [_]c.PyGetSetDef{
    .{
        .name = "type",
        .get = get_type,
        .set = null,
        .doc = "Type of motion blur: 'linear', 'radial_zoom', or 'radial_spin'",
        .closure = null,
    },
    .{
        .name = "angle",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "angle", is_linear))),
        .set = null,
        .doc = "Blur angle in radians (linear only)",
        .closure = null,
    },
    .{
        .name = "distance",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "distance", is_linear))),
        .set = null,
        .doc = "Blur distance in pixels (linear only)",
        .closure = null,
    },
    .{
        .name = "center",
        .get = @ptrCast(@alignCast(py_utils.getterTuple2FieldsWhere(MotionBlurObject, "center_x", "center_y", is_not_linear))),
        .set = null,
        .doc = "Normalized center position (zoom/spin only)",
        .closure = null,
    },
    .{
        .name = "strength",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "strength", is_not_linear))),
        .set = null,
        .doc = "Blur strength 0.0-1.0 (zoom/spin only)",
        .closure = null,
    },
    .{ .name = null, .get = null, .set = null, .doc = null, .closure = null },
};

pub var MotionBlurType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.MotionBlur",
    .tp_basicsize = @sizeOf(MotionBlurObject),
    .tp_dealloc = motion_blur_dealloc,
    .tp_repr = motion_blur_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = motion_blur_doc,
    .tp_methods = &motion_blur_methods,
    .tp_getset = &motion_blur_getset,
    .tp_init = motion_blur_init,
    .tp_new = motion_blur_new,
};

// ============================================================================
// Metadata for stub generation
// ============================================================================

pub const motion_blur_properties_metadata = [_]stub_metadata.PropertyWithMetadata{
    .{
        .name = "type",
        .get = get_type,
        .set = null,
        .doc = "Type of motion blur: 'linear', 'radial_zoom', or 'radial_spin'",
        .type = "Literal['linear', 'radial_zoom', 'radial_spin']",
    },
    .{
        .name = "angle",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "angle", is_linear))),
        .set = null,
        .doc = "Blur angle in radians (linear only)",
        .type = "float | None",
    },
    .{
        .name = "distance",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "distance", is_linear))),
        .set = null,
        .doc = "Blur distance in pixels (linear only)",
        .type = "int | None",
    },
    .{
        .name = "center",
        .get = @ptrCast(@alignCast(py_utils.getterTuple2FieldsWhere(MotionBlurObject, "center_x", "center_y", is_not_linear))),
        .set = null,
        .doc = "Normalized center position (zoom/spin only)",
        .type = "tuple[float, float] | None",
    },
    .{
        .name = "strength",
        .get = @ptrCast(@alignCast(py_utils.getterOptionalFieldWhere(MotionBlurObject, "strength", is_not_linear))),
        .set = null,
        .doc = "Blur strength 0.0-1.0 (zoom/spin only)",
        .type = "float | None",
    },
};

pub const motion_blur_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "linear",
        .params = "angle: float, distance: int",
        .returns = "MotionBlur",
        .doc = "Create linear motion blur configuration.",
    },
    .{
        .name = "radial_zoom",
        .params = "center: tuple[float, float] = (0.5, 0.5), strength: float = 0.5",
        .returns = "MotionBlur",
        .doc = "Create radial zoom blur configuration.",
    },
    .{
        .name = "radial_spin",
        .params = "center: tuple[float, float] = (0.5, 0.5), strength: float = 0.5",
        .returns = "MotionBlur",
        .doc = "Create radial spin blur configuration.",
    },
};

pub const motion_blur_special_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "__repr__",
        .params = "self",
        .returns = "str",
        .doc = null,
    },
};

// Register the motion blur type
pub fn registerMotionBlur(module: *c.PyObject) !void {
    // Initialize type
    if (c.PyType_Ready(&MotionBlurType) < 0) {
        return error.TypeInitFailed;
    }

    // Add to module
    // TODO: Remove explicit cast after Python 3.10 is dropped
    c.Py_INCREF(@as(?*c.PyObject, @ptrCast(&MotionBlurType)));
    if (c.PyModule_AddObject(module, "MotionBlur", @as(?*c.PyObject, @ptrCast(&MotionBlurType))) < 0) {
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(&MotionBlurType)));
        return error.ModuleAddFailed;
    }
}
