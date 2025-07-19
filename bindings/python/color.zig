// Color bindings using the automated registry-based approach
// This file is now much simpler - all color types are defined in color_registry.zig

const std = @import("std");
const zignal = @import("zignal");
const color_factory = @import("color_factory.zig");
const color_registry = @import("color_registry.zig");
const py_utils = @import("py_utils.zig");

pub const registerType = py_utils.registerType;

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// GENERATE ALL COLOR BINDINGS FROM REGISTRY
// ============================================================================

// For each color type in the registry, we generate:
// 1. The binding type (e.g., RgbBinding)
// 2. The getset and methods arrays
// 3. The PyTypeObject (e.g., RgbType)

// RGB
pub const RgbBinding = color_factory.createColorBinding(
    "Rgb",
    zignal.Rgb,
    .{
        .validation_fn = color_registry.ColorRegistry[0].validation_fn,
        .validation_error = color_registry.ColorRegistry[0].validation_error,
        .doc = color_registry.ColorRegistry[0].doc,
    },
);
var rgb_getset = RgbBinding.generateGetters();
var rgb_methods = RgbBinding.generateMethods();
pub var RgbType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Rgb",
    .tp_basicsize = @sizeOf(RgbBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&RgbBinding.dealloc),
    .tp_repr = @ptrCast(&RgbBinding.repr),
    .tp_str = @ptrCast(&RgbBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[0].doc.ptr,
    .tp_methods = @ptrCast(&rgb_methods),
    .tp_getset = @ptrCast(&rgb_getset),
    .tp_init = @ptrCast(&RgbBinding.init),
    .tp_new = @ptrCast(&RgbBinding.new),
};

// RGBA
pub const RgbaBinding = color_factory.createColorBinding(
    "Rgba",
    zignal.Rgba,
    .{
        .validation_fn = color_registry.ColorRegistry[1].validation_fn,
        .validation_error = color_registry.ColorRegistry[1].validation_error,
        .doc = color_registry.ColorRegistry[1].doc,
    },
);
var rgba_getset = RgbaBinding.generateGetters();
var rgba_methods = RgbaBinding.generateMethods();
pub var RgbaType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Rgba",
    .tp_basicsize = @sizeOf(RgbaBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&RgbaBinding.dealloc),
    .tp_repr = @ptrCast(&RgbaBinding.repr),
    .tp_str = @ptrCast(&RgbaBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[1].doc.ptr,
    .tp_methods = @ptrCast(&rgba_methods),
    .tp_getset = @ptrCast(&rgba_getset),
    .tp_init = @ptrCast(&RgbaBinding.init),
    .tp_new = @ptrCast(&RgbaBinding.new),
};

// HSV
pub const HsvBinding = color_factory.createColorBinding(
    "Hsv",
    zignal.Hsv,
    .{
        .validation_fn = color_registry.ColorRegistry[2].validation_fn,
        .validation_error = color_registry.ColorRegistry[2].validation_error,
        .doc = color_registry.ColorRegistry[2].doc,
    },
);
var hsv_getset = HsvBinding.generateGetters();
var hsv_methods = HsvBinding.generateMethods();
pub var HsvType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Hsv",
    .tp_basicsize = @sizeOf(HsvBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&HsvBinding.dealloc),
    .tp_repr = @ptrCast(&HsvBinding.repr),
    .tp_str = @ptrCast(&HsvBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[2].doc.ptr,
    .tp_methods = @ptrCast(&hsv_methods),
    .tp_getset = @ptrCast(&hsv_getset),
    .tp_init = @ptrCast(&HsvBinding.init),
    .tp_new = @ptrCast(&HsvBinding.new),
};

// HSL
pub const HslBinding = color_factory.createColorBinding(
    "Hsl",
    zignal.Hsl,
    .{
        .validation_fn = color_registry.ColorRegistry[3].validation_fn,
        .validation_error = color_registry.ColorRegistry[3].validation_error,
        .doc = color_registry.ColorRegistry[3].doc,
    },
);
var hsl_getset = HslBinding.generateGetters();
var hsl_methods = HslBinding.generateMethods();
pub var HslType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Hsl",
    .tp_basicsize = @sizeOf(HslBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&HslBinding.dealloc),
    .tp_repr = @ptrCast(&HslBinding.repr),
    .tp_str = @ptrCast(&HslBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[3].doc.ptr,
    .tp_methods = @ptrCast(&hsl_methods),
    .tp_getset = @ptrCast(&hsl_getset),
    .tp_init = @ptrCast(&HslBinding.init),
    .tp_new = @ptrCast(&HslBinding.new),
};

// LAB
pub const LabBinding = color_factory.createColorBinding(
    "Lab",
    zignal.Lab,
    .{
        .validation_fn = color_registry.ColorRegistry[4].validation_fn,
        .validation_error = color_registry.ColorRegistry[4].validation_error,
        .doc = color_registry.ColorRegistry[4].doc,
    },
);
var lab_getset = LabBinding.generateGetters();
var lab_methods = LabBinding.generateMethods();
pub var LabType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Lab",
    .tp_basicsize = @sizeOf(LabBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&LabBinding.dealloc),
    .tp_repr = @ptrCast(&LabBinding.repr),
    .tp_str = @ptrCast(&LabBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[4].doc.ptr,
    .tp_methods = @ptrCast(&lab_methods),
    .tp_getset = @ptrCast(&lab_getset),
    .tp_init = @ptrCast(&LabBinding.init),
    .tp_new = @ptrCast(&LabBinding.new),
};

// XYZ
pub const XyzBinding = color_factory.createColorBinding(
    "Xyz",
    zignal.Xyz,
    .{
        .validation_fn = color_registry.ColorRegistry[5].validation_fn,
        .validation_error = color_registry.ColorRegistry[5].validation_error,
        .doc = color_registry.ColorRegistry[5].doc,
    },
);
var xyz_getset = XyzBinding.generateGetters();
var xyz_methods = XyzBinding.generateMethods();
pub var XyzType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Xyz",
    .tp_basicsize = @sizeOf(XyzBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&XyzBinding.dealloc),
    .tp_repr = @ptrCast(&XyzBinding.repr),
    .tp_str = @ptrCast(&XyzBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[5].doc.ptr,
    .tp_methods = @ptrCast(&xyz_methods),
    .tp_getset = @ptrCast(&xyz_getset),
    .tp_init = @ptrCast(&XyzBinding.init),
    .tp_new = @ptrCast(&XyzBinding.new),
};

// OKLAB
pub const OklabBinding = color_factory.createColorBinding(
    "Oklab",
    zignal.Oklab,
    .{
        .validation_fn = color_registry.ColorRegistry[6].validation_fn,
        .validation_error = color_registry.ColorRegistry[6].validation_error,
        .doc = color_registry.ColorRegistry[6].doc,
    },
);
var oklab_getset = OklabBinding.generateGetters();
var oklab_methods = OklabBinding.generateMethods();
pub var OklabType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Oklab",
    .tp_basicsize = @sizeOf(OklabBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&OklabBinding.dealloc),
    .tp_repr = @ptrCast(&OklabBinding.repr),
    .tp_str = @ptrCast(&OklabBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[6].doc.ptr,
    .tp_methods = @ptrCast(&oklab_methods),
    .tp_getset = @ptrCast(&oklab_getset),
    .tp_init = @ptrCast(&OklabBinding.init),
    .tp_new = @ptrCast(&OklabBinding.new),
};

// OKLCH
pub const OklchBinding = color_factory.createColorBinding(
    "Oklch",
    zignal.Oklch,
    .{
        .validation_fn = color_registry.ColorRegistry[7].validation_fn,
        .validation_error = color_registry.ColorRegistry[7].validation_error,
        .doc = color_registry.ColorRegistry[7].doc,
    },
);
var oklch_getset = OklchBinding.generateGetters();
var oklch_methods = OklchBinding.generateMethods();
pub var OklchType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Oklch",
    .tp_basicsize = @sizeOf(OklchBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&OklchBinding.dealloc),
    .tp_repr = @ptrCast(&OklchBinding.repr),
    .tp_str = @ptrCast(&OklchBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.ColorRegistry[7].doc.ptr,
    .tp_methods = @ptrCast(&oklch_methods),
    .tp_getset = @ptrCast(&oklch_getset),
    .tp_init = @ptrCast(&OklchBinding.init),
    .tp_new = @ptrCast(&OklchBinding.new),
};

// ============================================================================
// REGISTRATION HELPER
// ============================================================================

/// Register all color types from the registry in one go
pub fn registerAllColorTypes(module: [*c]c.PyObject) !void {
    // Register each type - the order must match the registry
    try registerType(@ptrCast(module), "Rgb", @ptrCast(&RgbType));
    try registerType(@ptrCast(module), "Rgba", @ptrCast(&RgbaType));
    try registerType(@ptrCast(module), "Hsv", @ptrCast(&HsvType));
    try registerType(@ptrCast(module), "Hsl", @ptrCast(&HslType));
    try registerType(@ptrCast(module), "Lab", @ptrCast(&LabType));
    try registerType(@ptrCast(module), "Xyz", @ptrCast(&XyzType));
    try registerType(@ptrCast(module), "Oklab", @ptrCast(&OklabType));
    try registerType(@ptrCast(module), "Oklch", @ptrCast(&OklchType));
}

// ============================================================================
// CONVERSION HELPERS BETWEEN COLOR TYPES
// ============================================================================

/// Create a Python object from any zignal color type
pub fn createPyObject(color: anytype) ?*c.PyObject {
    const T = @TypeOf(color);
    
    // Handle each color type
    if (T == zignal.Rgb) {
        return createRgbPyObject(color);
    } else if (T == zignal.Rgba) {
        return createRgbaPyObject(color);
    } else if (T == zignal.Hsv) {
        return createHsvPyObject(color);
    } else if (T == zignal.Hsl) {
        return createHslPyObject(color);
    } else if (T == zignal.Lab) {
        return createLabPyObject(color);
    } else if (T == zignal.Xyz) {
        return createXyzPyObject(color);
    } else if (T == zignal.Oklab) {
        return createOklabPyObject(color);
    } else if (T == zignal.Oklch) {
        return createOklchPyObject(color);
    }
    
    return null;
}

// Helper functions for each type
fn createRgbPyObject(rgb: zignal.Rgb) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&RgbType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*RgbBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = rgb.r;
    py_obj.field1 = rgb.g;
    py_obj.field2 = rgb.b;
    
    return obj;
}

fn createRgbaPyObject(rgba: zignal.Rgba) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&RgbaType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*RgbaBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = rgba.r;
    py_obj.field1 = rgba.g;
    py_obj.field2 = rgba.b;
    py_obj.field3 = rgba.a;
    
    return obj;
}

fn createHsvPyObject(hsv: zignal.Hsv) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&HsvType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*HsvBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = hsv.h;
    py_obj.field1 = hsv.s;
    py_obj.field2 = hsv.v;
    
    return obj;
}

fn createHslPyObject(hsl: zignal.Hsl) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&HslType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*HslBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = hsl.h;
    py_obj.field1 = hsl.s;
    py_obj.field2 = hsl.l;
    
    return obj;
}

fn createLabPyObject(lab: zignal.Lab) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&LabType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*LabBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = lab.l;
    py_obj.field1 = lab.a;
    py_obj.field2 = lab.b;
    
    return obj;
}

fn createXyzPyObject(xyz: zignal.Xyz) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&XyzType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*XyzBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = xyz.x;
    py_obj.field1 = xyz.y;
    py_obj.field2 = xyz.z;
    
    return obj;
}

fn createOklabPyObject(oklab: zignal.Oklab) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&OklabType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*OklabBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = oklab.l;
    py_obj.field1 = oklab.a;
    py_obj.field2 = oklab.b;
    
    return obj;
}

fn createOklchPyObject(oklch: zignal.Oklch) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&OklchType), null, null);
    if (obj == null) return null;
    
    const py_obj = @as(*OklchBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = oklch.l;
    py_obj.field1 = oklch.c;
    py_obj.field2 = oklch.h;
    
    return obj;
}