const std = @import("std");
const build_options = @import("build_options");

const color = @import("color.zig");
const image = @import("image.zig");
const canvas = @import("canvas.zig");
const py_utils = @import("py_utils.zig");
const fdm = @import("fdm.zig");
const interpolation = @import("interpolation.zig");
const stub_metadata = @import("stub_metadata.zig");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// MODULE FUNCTIONS
// ============================================================================

var zignal_module = c.PyModuleDef{
    .m_name = "zignal",
    .m_doc = "zero dependency image processing library",
    .m_size = -1,
    .m_methods = &zignal_methods,
    .m_slots = null,
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

var zignal_methods = [_]c.PyMethodDef{
    .{
        .ml_name = "feature_distribution_match",
        .ml_meth = @ptrCast(&fdm.feature_distribution_match),
        .ml_flags = c.METH_VARARGS,
        .ml_doc = fdm.feature_distribution_match_doc,
    },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

pub export fn PyInit__zignal() ?*c.PyObject {
    const m = c.PyModule_Create(&zignal_module);
    if (m == null) return null;

    // Register Image type
    py_utils.registerType(@ptrCast(m), "Image", @ptrCast(&image.ImageType)) catch |err| {
        std.log.err("Failed to register Image: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Register Canvas type
    py_utils.registerType(@ptrCast(m), "Canvas", @ptrCast(&canvas.CanvasType)) catch |err| {
        std.log.err("Failed to register Canvas: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Register DrawMode enum
    canvas.registerDrawMode(@ptrCast(m)) catch |err| {
        std.log.err("Failed to register DrawMode: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Register InterpolationMethod enum
    interpolation.registerInterpolationMethod(@ptrCast(m)) catch |err| {
        std.log.err("Failed to register InterpolationMethod: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Register all color types from the registry
    color.registerAllColorTypes(@ptrCast(m)) catch |err| {
        std.log.err("Failed to register color types: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Add __version__ as a module attribute from build options
    const version_str = c.PyUnicode_FromString(@ptrCast(build_options.version));
    if (version_str == null) {
        c.Py_DECREF(m);
        return null;
    }
    if (c.PyModule_AddObject(m, "__version__", version_str) < 0) {
        c.Py_DECREF(version_str);
        c.Py_DECREF(m);
        return null;
    }

    return m;
}

pub fn main() void {}

// ============================================================================
// MODULE STUB GENERATION METADATA
// ============================================================================

pub const module_functions = [_]stub_metadata.FunctionInfo{
    .{
        .name = "feature_distribution_match",
        .params = "source: Image, reference: Image",
        .returns = "None",
        .doc =
        \\Apply Feature Distribution Matching (FDM) to transfer color/style from reference to source image.
        \\
        \\This function modifies the source image in-place to match the color distribution
        \\(mean and covariance) of the reference image while preserving the structure of the source.
        \\
        \\Parameters
        \\----------
        \\source : Image
        \\    Source image to be modified (modified in-place)
        \\reference : Image
        \\    Reference image providing target color distribution
        \\
        \\Returns
        \\-------
        \\None
        \\    This function modifies the source image in-place
        ,
    },
};
