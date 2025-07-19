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
        .validation_fn = color_registry.getBindingConfig(zignal.Rgb).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Rgb).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Rgb).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Rgb).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Rgba).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Rgba).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Rgba).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Rgba).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Hsv).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Hsv).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Hsv).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Hsv).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Hsl).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Hsl).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Hsl).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Hsl).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Lab).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Lab).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Lab).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Lab).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Xyz).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Xyz).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Xyz).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Xyz).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Oklab).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Oklab).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Oklab).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Oklab).?.doc.ptr,
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
        .validation_fn = color_registry.getBindingConfig(zignal.Oklch).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Oklch).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Oklch).?.doc,
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
    .tp_doc = color_registry.getBindingConfig(zignal.Oklch).?.doc.ptr,
    .tp_methods = @ptrCast(&oklch_methods),
    .tp_getset = @ptrCast(&oklch_getset),
    .tp_init = @ptrCast(&OklchBinding.init),
    .tp_new = @ptrCast(&OklchBinding.new),
};

// LCH
pub const LchBinding = color_factory.createColorBinding(
    "Lch",
    zignal.Lch,
    .{
        .validation_fn = color_registry.getBindingConfig(zignal.Lch).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Lch).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Lch).?.doc,
    },
);
var lch_getset = LchBinding.generateGetters();
var lch_methods = LchBinding.generateMethods();
pub var LchType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Lch",
    .tp_basicsize = @sizeOf(LchBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&LchBinding.dealloc),
    .tp_repr = @ptrCast(&LchBinding.repr),
    .tp_str = @ptrCast(&LchBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.getBindingConfig(zignal.Lch).?.doc.ptr,
    .tp_methods = @ptrCast(&lch_methods),
    .tp_getset = @ptrCast(&lch_getset),
    .tp_init = @ptrCast(&LchBinding.init),
    .tp_new = @ptrCast(&LchBinding.new),
};

// LMS
pub const LmsBinding = color_factory.createColorBinding(
    "Lms",
    zignal.Lms,
    .{
        .validation_fn = color_registry.getBindingConfig(zignal.Lms).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Lms).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Lms).?.doc,
    },
);
var lms_getset = LmsBinding.generateGetters();
var lms_methods = LmsBinding.generateMethods();
pub var LmsType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Lms",
    .tp_basicsize = @sizeOf(LmsBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&LmsBinding.dealloc),
    .tp_repr = @ptrCast(&LmsBinding.repr),
    .tp_str = @ptrCast(&LmsBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.getBindingConfig(zignal.Lms).?.doc.ptr,
    .tp_methods = @ptrCast(&lms_methods),
    .tp_getset = @ptrCast(&lms_getset),
    .tp_init = @ptrCast(&LmsBinding.init),
    .tp_new = @ptrCast(&LmsBinding.new),
};

// XYB
pub const XybBinding = color_factory.createColorBinding(
    "Xyb",
    zignal.Xyb,
    .{
        .validation_fn = color_registry.getBindingConfig(zignal.Xyb).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Xyb).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Xyb).?.doc,
    },
);
var xyb_getset = XybBinding.generateGetters();
var xyb_methods = XybBinding.generateMethods();
pub var XybType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Xyb",
    .tp_basicsize = @sizeOf(XybBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&XybBinding.dealloc),
    .tp_repr = @ptrCast(&XybBinding.repr),
    .tp_str = @ptrCast(&XybBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.getBindingConfig(zignal.Xyb).?.doc.ptr,
    .tp_methods = @ptrCast(&xyb_methods),
    .tp_getset = @ptrCast(&xyb_getset),
    .tp_init = @ptrCast(&XybBinding.init),
    .tp_new = @ptrCast(&XybBinding.new),
};

// YCBCR
pub const YcbcrBinding = color_factory.createColorBinding(
    "Ycbcr",
    zignal.Ycbcr,
    .{
        .validation_fn = color_registry.getBindingConfig(zignal.Ycbcr).?.validation_fn,
        .validation_error = color_registry.getBindingConfig(zignal.Ycbcr).?.validation_error,
        .doc = color_registry.getBindingConfig(zignal.Ycbcr).?.doc,
    },
);
var ycbcr_getset = YcbcrBinding.generateGetters();
var ycbcr_methods = YcbcrBinding.generateMethods();
pub var YcbcrType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.Ycbcr",
    .tp_basicsize = @sizeOf(YcbcrBinding.PyObjectType),
    .tp_dealloc = @ptrCast(&YcbcrBinding.dealloc),
    .tp_repr = @ptrCast(&YcbcrBinding.repr),
    .tp_str = @ptrCast(&YcbcrBinding.repr),
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = color_registry.getBindingConfig(zignal.Ycbcr).?.doc.ptr,
    .tp_methods = @ptrCast(&ycbcr_methods),
    .tp_getset = @ptrCast(&ycbcr_getset),
    .tp_init = @ptrCast(&YcbcrBinding.init),
    .tp_new = @ptrCast(&YcbcrBinding.new),
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
    try registerType(@ptrCast(module), "Lch", @ptrCast(&LchType));
    try registerType(@ptrCast(module), "Lms", @ptrCast(&LmsType));
    try registerType(@ptrCast(module), "Xyb", @ptrCast(&XybType));
    try registerType(@ptrCast(module), "Ycbcr", @ptrCast(&YcbcrType));
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
    } else if (T == zignal.Lch) {
        return createLchPyObject(color);
    } else if (T == zignal.Lms) {
        return createLmsPyObject(color);
    } else if (T == zignal.Xyb) {
        return createXybPyObject(color);
    } else if (T == zignal.Ycbcr) {
        return createYcbcrPyObject(color);
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

fn createLchPyObject(lch: zignal.Lch) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&LchType), null, null);
    if (obj == null) return null;

    const py_obj = @as(*LchBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = lch.l;
    py_obj.field1 = lch.c;
    py_obj.field2 = lch.h;

    return obj;
}

fn createLmsPyObject(lms: zignal.Lms) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&LmsType), null, null);
    if (obj == null) return null;

    const py_obj = @as(*LmsBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = lms.l;
    py_obj.field1 = lms.m;
    py_obj.field2 = lms.s;

    return obj;
}

fn createXybPyObject(xyb: zignal.Xyb) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&XybType), null, null);
    if (obj == null) return null;

    const py_obj = @as(*XybBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = xyb.x;
    py_obj.field1 = xyb.y;
    py_obj.field2 = xyb.b;

    return obj;
}

fn createYcbcrPyObject(ycbcr: zignal.Ycbcr) ?*c.PyObject {
    const obj = c.PyType_GenericNew(@ptrCast(&YcbcrType), null, null);
    if (obj == null) return null;

    const py_obj = @as(*YcbcrBinding.PyObjectType, @ptrCast(obj));
    py_obj.field0 = ycbcr.y;
    py_obj.field1 = ycbcr.cb;
    py_obj.field2 = ycbcr.cr;

    return obj;
}
