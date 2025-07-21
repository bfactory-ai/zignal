const std = @import("std");
const build_options = @import("build_options");

const color = @import("color.zig");
const image = @import("image.zig");
const py_utils = @import("py_utils.zig");

const zignal = @import("zignal");

const c = @cImport({
    @cDefine("PY_SSIZE_T_CLEAN", {});
    @cInclude("Python.h");
});

// ============================================================================
// MODULE FUNCTIONS
// ============================================================================

// Global allocator for module functions
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

fn feature_distribution_match(self: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self;

    var src_obj: ?*c.PyObject = undefined;
    var ref_obj: ?*c.PyObject = undefined;

    const format = comptime std.fmt.comptimePrint("OO", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &src_obj, &ref_obj) == 0) {
        return null;
    }

    // Check if both arguments are ImageRgb objects
    if (c.PyObject_IsInstance(src_obj, @ptrCast(&image.ImageRgbType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "First argument must be an ImageRgb object");
        return null;
    }

    if (c.PyObject_IsInstance(ref_obj, @ptrCast(&image.ImageRgbType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Second argument must be an ImageRgb object");
        return null;
    }

    // Cast to ImageRgbObject
    const src_img_obj = @as(*image.ImageRgbObject, @ptrCast(src_obj.?));
    const ref_img_obj = @as(*image.ImageRgbObject, @ptrCast(ref_obj.?));

    // Check if images are initialized
    if (src_img_obj.image_ptr == null) {
        c.PyErr_SetString(c.PyExc_ValueError, "Source image is not initialized");
        return null;
    }

    if (ref_img_obj.image_ptr == null) {
        c.PyErr_SetString(c.PyExc_ValueError, "Reference image is not initialized");
        return null;
    }

    // Get allocator
    const allocator = gpa.allocator();

    // Call the FDM function
    zignal.featureDistributionMatch(zignal.Rgb, allocator, src_img_obj.image_ptr.?.*, ref_img_obj.image_ptr.?.*) catch |err| {
        switch (err) {
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory during feature distribution matching"),
        }
        return null;
    };

    // Return None
    const none = c.Py_None();
    c.Py_INCREF(none);
    return none;
}

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
    .{ .ml_name = "feature_distribution_match", .ml_meth = feature_distribution_match, .ml_flags = c.METH_VARARGS, .ml_doc = 
    \\Apply Feature Distribution Matching (FDM) to transfer color/style from reference to source image.
    \\
    \\This function modifies the source image in-place to match the color distribution
    \\(mean and covariance) of the reference image while preserving the structure of the source.
    \\
    \\Args:
    \\    source (ImageRgb): Source image to be modified (modified in-place)
    \\    reference (ImageRgb): Reference image providing target color distribution
    \\
    \\Returns:
    \\    None
    \\
    \\Example:
    \\    src_img = ImageRgb.load("source.png")
    \\    ref_img = ImageRgb.load("reference.png")
    \\    zignal.feature_distribution_match(src_img, ref_img)
    \\    src_img.save("result.png")
    },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

pub export fn PyInit__zignal() ?*c.PyObject {
    const m = c.PyModule_Create(&zignal_module);
    if (m == null) return null;

    // Register all color types from the registry
    color.registerAllColorTypes(@ptrCast(m)) catch |err| {
        std.log.err("Failed to register color types: {}", .{err});
        c.Py_DECREF(m);
        return null;
    };

    // Register ImageRgb type
    py_utils.registerType(@ptrCast(m), "ImageRgb", @ptrCast(&image.ImageRgbType)) catch {
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
