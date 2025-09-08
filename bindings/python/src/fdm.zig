const std = @import("std");

const zignal = @import("zignal");
const FeatureDistributionMatching = zignal.FeatureDistributionMatching;
const Rgb = zignal.Rgb;
const Image = zignal.Image;

const image = @import("image.zig");
const py_utils = @import("py_utils.zig");
const allocator = py_utils.allocator;
const c = py_utils.c;
const stub_metadata = @import("stub_metadata.zig");

// FeatureDistributionMatching Python object
pub const FeatureDistributionMatchingObject = extern struct {
    ob_base: c.PyObject,
    // Store a pointer to the heap-allocated FDM struct
    fdm_ptr: ?*FeatureDistributionMatching(Rgb),
};

fn fdm_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*FeatureDistributionMatchingObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.fdm_ptr = null;
    }
    return @as(?*c.PyObject, @ptrCast(self));
}

fn fdm_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = args;
    _ = kwds;
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    // Create and store the FDM struct
    const fdm_ptr = allocator.create(FeatureDistributionMatching(Rgb)) catch {
        c.PyErr_SetString(c.PyExc_MemoryError, "Failed to allocate FeatureDistributionMatching");
        return -1;
    };

    // Initialize the FDM instance
    fdm_ptr.* = FeatureDistributionMatching(Rgb).init(allocator);
    self.fdm_ptr = fdm_ptr;

    return 0;
}

fn fdm_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    // Free the FDM struct
    if (self.fdm_ptr) |ptr| {
        ptr.deinit();
        allocator.destroy(ptr);
    }

    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn fdm_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    if (self.fdm_ptr == null) {
        return c.PyUnicode_FromString("FeatureDistributionMatching(uninitialized)");
    }

    return c.PyUnicode_FromString("FeatureDistributionMatching()");
}

// set_target method
const set_target_doc =
    \\Set the target image whose distribution will be matched.
    \\
    \\This method computes and stores the target distribution statistics (mean and covariance)
    \\for reuse across multiple source images. This is more efficient than recomputing
    \\the statistics for each image when applying the same style to multiple images.
    \\
    \\## Parameters
    \\- `image` (`Image`): Target image providing the color distribution to match. Must be RGB.
    \\
    \\## Examples
    \\```python
    \\fdm = FeatureDistributionMatching()
    \\target = Image.load("sunset.png")
    \\fdm.set_target(target)
    \\```
;

fn fdm_set_target(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    const fdm_ptr = py_utils.validateNonNull(*FeatureDistributionMatching(Rgb), self.fdm_ptr, "FeatureDistributionMatching") catch return null;

    var target_obj: ?*c.PyObject = undefined;
    const kw = comptime py_utils.kw(&.{"image"});
    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &target_obj) == 0) {
        return null;
    }

    // Check if argument is an Image object
    if (c.PyObject_IsInstance(target_obj, @ptrCast(&image.ImageType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Argument must be an Image object");
        return null;
    }

    const target_img_obj = @as(*image.ImageObject, @ptrCast(target_obj.?));

    // Accept only RGB format
    if (target_img_obj.py_image == null) {
        c.PyErr_SetString(c.PyExc_TypeError, "Target image must be an Image");
        return null;
    }
    const target_rgb: zignal.Image(zignal.Rgb) = switch (target_img_obj.py_image.?.data) {
        .rgb => |imgv| imgv,
        else => {
            c.PyErr_SetString(c.PyExc_TypeError, "Target image must be RGB");
            return null;
        },
    };
    fdm_ptr.setTarget(target_rgb) catch |err| {
        switch (err) {
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory setting target"),
        }
        return null;
    };
    // Success

    const none = c.Py_None();
    c.Py_INCREF(none);
    return none;
}

// set_source method
const set_source_doc =
    \\Set the source image to be transformed.
    \\
    \\The source image will be modified in-place when update() is called.
    \\
    \\## Parameters
    \\- `image` (`Image`): Source image to be modified. Must be RGB.
    \\
    \\## Examples
    \\```python
    \\fdm = FeatureDistributionMatching()
    \\source = Image.load("portrait.png")
    \\fdm.set_source(source)
    \\```
;

fn fdm_set_source(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    const fdm_ptr = py_utils.validateNonNull(*FeatureDistributionMatching(Rgb), self.fdm_ptr, "FeatureDistributionMatching") catch return null;

    var source_obj: ?*c.PyObject = undefined;
    const kw = comptime py_utils.kw(&.{"image"});
    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &source_obj) == 0) {
        return null;
    }

    // Check if argument is an Image object
    if (c.PyObject_IsInstance(source_obj, @ptrCast(&image.ImageType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Argument must be an Image object");
        return null;
    }

    const source_img_obj = @as(*image.ImageObject, @ptrCast(source_obj.?));

    if (source_img_obj.py_image == null) {
        c.PyErr_SetString(c.PyExc_TypeError, "Source image must be an Image");
        return null;
    }
    const src_rgb: zignal.Image(zignal.Rgb) = switch (source_img_obj.py_image.?.data) {
        .rgb => |imgv| imgv,
        else => {
            c.PyErr_SetString(c.PyExc_TypeError, "Source image must be RGB");
            return null;
        },
    };
    fdm_ptr.setSource(src_rgb) catch |err| {
        switch (err) {
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory setting source"),
        }
        return null;
    };
    // Success

    const none = c.Py_None();
    c.Py_INCREF(none);
    return none;
}

// match method
const match_doc =
    \\Set both source and target images and apply the transformation.
    \\
    \\This is a convenience method that combines set_source(), set_target(), and update()
    \\into a single call. The source image is modified in-place.
    \\
    \\## Parameters
    \\- `source` (`Image`): Source image to be modified (RGB)
    \\- `target` (`Image`): Target image providing the color distribution to match (RGB)
    \\
    \\## Examples
    \\```python
    \\fdm = FeatureDistributionMatching()
    \\source = Image.load("portrait.png")
    \\target = Image.load("sunset.png")
    \\fdm.match(source, target)  # source is now modified
    \\source.save("portrait_sunset.png")
    \\```
;

fn fdm_match(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    const fdm_ptr = py_utils.validateNonNull(*FeatureDistributionMatching(Rgb), self.fdm_ptr, "FeatureDistributionMatching") catch return null;

    var source_obj: ?*c.PyObject = undefined;
    var target_obj: ?*c.PyObject = undefined;

    const kw = comptime py_utils.kw(&.{ "source", "target" });
    const format = std.fmt.comptimePrint("OO", .{});
    if (c.PyArg_ParseTupleAndKeywords(args, kwds, format.ptr, @ptrCast(@constCast(&kw)), &source_obj, &target_obj) == 0) {
        return null;
    }

    // Check if both arguments are Image objects
    if (c.PyObject_IsInstance(source_obj, @ptrCast(&image.ImageType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "First argument must be an Image object");
        return null;
    }

    if (c.PyObject_IsInstance(target_obj, @ptrCast(&image.ImageType)) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Second argument must be an Image object");
        return null;
    }

    // Cast to ImageObject
    const source_img_obj = @as(*image.ImageObject, @ptrCast(source_obj.?));
    const target_img_obj = @as(*image.ImageObject, @ptrCast(target_obj.?));

    // Accept only RGB format
    if (source_img_obj.py_image == null or target_img_obj.py_image == null) {
        c.PyErr_SetString(c.PyExc_TypeError, "Source and target must be Image objects");
        return null;
    }
    const src_rgb2: zignal.Image(zignal.Rgb) = switch (source_img_obj.py_image.?.data) {
        .rgb => |imgv| imgv,
        else => {
            c.PyErr_SetString(c.PyExc_TypeError, "Source image must be RGB");
            return null;
        },
    };
    const dst_rgb2: zignal.Image(zignal.Rgb) = switch (target_img_obj.py_image.?.data) {
        .rgb => |imgv| imgv,
        else => {
            c.PyErr_SetString(c.PyExc_TypeError, "Target image must be RGB");
            return null;
        },
    };

    // Call match
    fdm_ptr.match(src_rgb2, dst_rgb2) catch |err| {
        switch (err) {
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory during match"),
            error.NoTargetSet => c.PyErr_SetString(c.PyExc_RuntimeError, "No target image set"),
            error.NoSourceSet => c.PyErr_SetString(c.PyExc_RuntimeError, "No source image set"),
        }
        return null;
    };

    const none = c.Py_None();
    c.Py_INCREF(none);
    return none;
}

// update method
const update_doc =
    \\Apply the feature distribution matching transformation.
    \\
    \\This method modifies the source image in-place to match the target distribution.
    \\Both source and target must be set before calling this method.
    \\
    \\## Raises
    \\- `RuntimeError`: If source or target has not been set
    \\
    \\## Examples
    \\```python
    \\fdm = FeatureDistributionMatching()
    \\fdm.set_target(target)
    \\fdm.set_source(source)
    \\fdm.update()  # source is now modified
    \\```
    \\
    \\### Batch processing
    \\```python
    \\fdm.set_target(style_image)
    \\for img in images:
    \\    fdm.set_source(img)
    \\    fdm.update()  # Each img is modified in-place
    \\```
;

fn fdm_update(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*FeatureDistributionMatchingObject, @ptrCast(self_obj.?));

    if (self.fdm_ptr == null) {
        c.PyErr_SetString(c.PyExc_ValueError, "FeatureDistributionMatching not initialized");
        return null;
    }

    // Apply the transformation
    self.fdm_ptr.?.update() catch |err| {
        switch (err) {
            error.OutOfMemory => c.PyErr_SetString(c.PyExc_MemoryError, "Out of memory during update"),
            error.NoTargetSet => c.PyErr_SetString(c.PyExc_RuntimeError, "No target image set. Call set_target() or match() first"),
            error.NoSourceSet => c.PyErr_SetString(c.PyExc_RuntimeError, "No source image set. Call set_source() or match() first"),
        }
        return null;
    };

    const none = c.Py_None();
    c.Py_INCREF(none);
    return none;
}

// Method definitions with metadata
pub const fdm_methods_metadata = [_]stub_metadata.MethodWithMetadata{
    .{
        .name = "set_target",
        .meth = @ptrCast(&fdm_set_target),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = set_target_doc,
        .params = "self, image: Image",
        .returns = "None",
    },
    .{
        .name = "set_source",
        .meth = @ptrCast(&fdm_set_source),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = set_source_doc,
        .params = "self, image: Image",
        .returns = "None",
    },
    .{
        .name = "match",
        .meth = @ptrCast(&fdm_match),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = match_doc,
        .params = "self, source: Image, target: Image",
        .returns = "None",
    },
    .{
        .name = "update",
        .meth = @ptrCast(&fdm_update),
        .flags = c.METH_NOARGS,
        .doc = update_doc,
        .params = "self",
        .returns = "None",
    },
};

// Generate PyMethodDef array from metadata
var fdm_methods = stub_metadata.toPyMethodDefArray(&fdm_methods_metadata);

// Class documentation - keep it simple
const fdm_class_doc = "Feature Distribution Matching for image style transfer.";

// Init documentation - detailed explanation
pub const fdm_init_doc =
    \\Initialize a new FeatureDistributionMatching instance.
    \\
    \\Creates a new FDM instance that can be used to transfer color distributions
    \\between images. The instance maintains internal state for efficient batch
    \\processing of multiple images with the same target distribution.
    \\
    \\## Examples
    \\```python
    \\# Create an FDM instance
    \\fdm = FeatureDistributionMatching()
    \\
    \\# Single image transformation
    \\source = Image.load("portrait.png")
    \\target = Image.load("sunset.png")
    \\fdm.match(source, target)  # source is modified in-place
    \\source.save("portrait_sunset.png")
    \\
    \\# Batch processing with same style
    \\style = Image.load("style_reference.png")
    \\fdm.set_target(style)
    \\for filename in image_files:
    \\    img = Image.load(filename)
    \\    fdm.set_source(img)
    \\    fdm.update()
    \\    img.save(f"styled_{filename}")
    \\```
    \\
    \\## Notes
    \\- The algorithm matches mean and covariance of pixel distributions
    \\- Target statistics are computed once and can be reused for multiple sources
    \\- See: https://facebookresearch.github.io/dino/blog/
;

// Special methods metadata for stub generation
pub const fdm_special_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "__init__",
        .params = "self",
        .returns = "None",
        .doc = fdm_init_doc,
    },
};

// PyTypeObject definition
pub var FeatureDistributionMatchingType = c.PyTypeObject{
    .ob_base = .{ .ob_base = .{}, .ob_size = 0 },
    .tp_name = "zignal.FeatureDistributionMatching",
    .tp_basicsize = @sizeOf(FeatureDistributionMatchingObject),
    .tp_dealloc = fdm_dealloc,
    .tp_repr = fdm_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = fdm_class_doc,
    .tp_methods = @ptrCast(&fdm_methods),
    .tp_init = fdm_init,
    .tp_new = fdm_new,
};
