const std = @import("std");

const zignal = @import("zignal");
const Rectangle = zignal.Rectangle;

const py_utils = @import("py_utils.zig");
pub const registerType = py_utils.registerType;
const c = py_utils.c;
const stub_metadata = @import("stub_metadata.zig");

pub const RectangleObject = extern struct {
    ob_base: c.PyObject,
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
};

fn rectangle_new(type_obj: ?*c.PyTypeObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    _ = kwds;

    const self = @as(?*RectangleObject, @ptrCast(c.PyType_GenericAlloc(type_obj, 0)));
    if (self) |obj| {
        obj.left = 0;
        obj.top = 0;
        obj.right = 0;
        obj.bottom = 0;
    }
    return @as(?*c.PyObject, @ptrCast(self));
}

fn rectangle_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    _ = kwds;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    // Parse arguments - expect left, top, right, bottom
    var left: f64 = undefined;
    var top: f64 = undefined;
    var right: f64 = undefined;
    var bottom: f64 = undefined;
    const format = std.fmt.comptimePrint("dddd", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &left, &top, &right, &bottom) == 0) {
        return -1;
    }

    // Validate rectangle
    if (right < left) {
        c.PyErr_SetString(c.PyExc_ValueError, "Right must be greater than or equal to left");
        return -1;
    }
    if (bottom < top) {
        c.PyErr_SetString(c.PyExc_ValueError, "Bottom must be greater than or equal to top");
        return -1;
    }

    self.left = left;
    self.top = top;
    self.right = right;
    self.bottom = bottom;

    return 0;
}

fn rectangle_dealloc(self_obj: ?*c.PyObject) callconv(.c) void {
    c.Py_TYPE(self_obj).*.tp_free.?(self_obj);
}

fn rectangle_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var buffer: [128]u8 = undefined;
    const formatted = std.fmt.bufPrintZ(&buffer, "Rectangle({d:.2}, {d:.2}, {d:.2}, {d:.2})", .{ self.left, self.top, self.right, self.bottom }) catch return null;
    return c.PyUnicode_FromString(formatted.ptr);
}

// Class method: init_center
const rectangle_init_center_doc =
    \\Create a Rectangle from center coordinates.
    \\
    \\## Parameters
    \\- `x` (float): Center x coordinate
    \\- `y` (float): Center y coordinate
    \\- `width` (float): Rectangle width
    \\- `height` (float): Rectangle height
    \\
    \\## Examples
    \\```python
    \\# Create a 100x50 rectangle centered at (50, 50)
    \\rect = Rectangle.init_center(50, 50, 100, 50)
    \\# This creates Rectangle(0, 25, 100, 75)
    \\```
;

fn rectangle_init_center(type_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    var x: f64 = undefined;
    var y: f64 = undefined;
    var width: f64 = undefined;
    var height: f64 = undefined;

    const format = std.fmt.comptimePrint("dddd", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &x, &y, &width, &height) == 0) {
        return null;
    }

    // Validate dimensions
    _ = py_utils.validatePositive(f64, width, "Width") catch return null;
    _ = py_utils.validatePositive(f64, height, "Height") catch return null;

    // Calculate bounds
    const left = x - width / 2;
    const top = y - height / 2;
    const right = left + width;
    const bottom = top + height;

    // Create new Rectangle instance
    const rect_args = c.Py_BuildValue("(dddd)", left, top, right, bottom) orelse return null;
    defer c.Py_DECREF(rect_args);

    const instance = c.PyObject_CallObject(@ptrCast(type_obj), rect_args);
    return instance;
}

// Instance methods
const rectangle_is_empty_doc =
    \\Check if the rectangle is ill-formed (empty).
    \\
    \\A rectangle is considered empty if its left >= right or top >= bottom.
    \\
    \\## Examples
    \\```python
    \\rect1 = Rectangle(0, 0, 100, 100)
    \\print(rect1.is_empty())  # False
    \\
    \\rect2 = Rectangle(100, 100, 100, 100)
    \\print(rect2.is_empty())  # True
    \\```
;

fn rectangle_is_empty(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    // For floats, use >= comparison (matches Rectangle.zig logic)
    const is_empty = self.top >= self.bottom or self.left >= self.right;
    return @ptrCast(py_utils.getPyBool(is_empty));
}

const rectangle_area_doc =
    \\Calculate the area of the rectangle.
    \\
    \\## Examples
    \\```python
    \\rect = Rectangle(0, 0, 100, 50)
    \\print(rect.area())  # 5000.0
    \\```
;

fn rectangle_area(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    const width = self.right - self.left;
    const height = self.bottom - self.top;
    const area = if (width < 0 or height < 0) 0 else width * height;

    return c.PyFloat_FromDouble(@as(f64, area));
}

const rectangle_contains_doc =
    \\Check if a point is inside the rectangle.
    \\
    \\Uses exclusive bounds for right and bottom edges.
    \\
    \\## Parameters
    \\- `x` (float): X coordinate to check
    \\- `y` (float): Y coordinate to check
    \\
    \\## Examples
    \\```python
    \\rect = Rectangle(0, 0, 100, 100)
    \\print(rect.contains(50, 50))   # True - inside
    \\print(rect.contains(100, 50))  # False - on right edge (exclusive)
    \\print(rect.contains(99.9, 99.9))  # True - just inside
    \\print(rect.contains(150, 50))  # False - outside
    \\```
;

fn rectangle_contains(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var x: f64 = undefined;
    var y: f64 = undefined;

    const format = std.fmt.comptimePrint("dd", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &x, &y) == 0) {
        return null;
    }

    const x_f32 = @as(f32, @floatCast(x));
    const y_f32 = @as(f32, @floatCast(y));

    const contains = x_f32 >= self.left and x_f32 < self.right and
        y_f32 >= self.top and y_f32 < self.bottom;

    return @ptrCast(py_utils.getPyBool(contains));
}

const rectangle_grow_doc =
    \\Create a new rectangle expanded by the given amount.
    \\
    \\## Parameters
    \\- `amount` (float): Amount to expand each border by
    \\
    \\## Examples
    \\```python
    \\rect = Rectangle(50, 50, 100, 100)
    \\grown = rect.grow(10)
    \\# Creates Rectangle(40, 40, 110, 110)
    \\```
;

fn rectangle_grow(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var amount: f64 = undefined;

    const format = std.fmt.comptimePrint("d", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &amount) == 0) {
        return null;
    }

    const amount_f32 = @as(f32, @floatCast(amount));

    // Create new rectangle with grown bounds
    const new_args = c.Py_BuildValue("(dddd)", @as(f64, self.left - amount_f32), @as(f64, self.top - amount_f32), @as(f64, self.right + amount_f32), @as(f64, self.bottom + amount_f32)) orelse return null;
    defer c.Py_DECREF(new_args);

    const new_rect = c.PyObject_CallObject(@ptrCast(&RectangleType), new_args);
    return new_rect;
}

const rectangle_shrink_doc =
    \\Create a new rectangle shrunk by the given amount.
    \\
    \\## Parameters
    \\- `amount` (float): Amount to shrink each border by
    \\
    \\## Examples
    \\```python
    \\rect = Rectangle(40, 40, 110, 110)
    \\shrunk = rect.shrink(10)
    \\# Creates Rectangle(50, 50, 100, 100)
    \\```
;

fn rectangle_shrink(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var amount: f64 = undefined;

    const format = std.fmt.comptimePrint("d", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &amount) == 0) {
        return null;
    }

    const amount_f32 = @as(f32, @floatCast(amount));

    // Create new rectangle with shrunk bounds
    const new_args = c.Py_BuildValue("(dddd)", @as(f64, self.left + amount_f32), @as(f64, self.top + amount_f32), @as(f64, self.right - amount_f32), @as(f64, self.bottom - amount_f32)) orelse return null;
    defer c.Py_DECREF(new_args);

    const new_rect = c.PyObject_CallObject(@ptrCast(&RectangleType), new_args);
    return new_rect;
}

const rectangle_intersect_doc =
    \\Calculate the intersection of this rectangle with another.
    \\
    \\## Parameters
    \\- `other` (Rectangle | tuple[float, float, float, float]): The other rectangle to intersect with
    \\
    \\## Examples
    \\```python
    \\rect1 = Rectangle(0, 0, 100, 100)
    \\rect2 = Rectangle(50, 50, 150, 150)
    \\intersection = rect1.intersect(rect2)
    \\# Returns Rectangle(50, 50, 100, 100)
    \\
    \\# Can also use a tuple
    \\intersection = rect1.intersect((50, 50, 150, 150))
    \\# Returns Rectangle(50, 50, 100, 100)
    \\
    \\rect3 = Rectangle(200, 200, 250, 250)
    \\result = rect1.intersect(rect3)  # Returns None (no overlap)
    \\```
;

fn rectangle_intersect(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var other_obj: ?*c.PyObject = undefined;

    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &other_obj) == 0) {
        return null;
    }

    // Parse the other rectangle (can be Rectangle or tuple)
    const other_rect = py_utils.parseRectangle(f64, other_obj) catch return null;

    // Calculate intersection bounds
    const left = @max(self.left, other_rect.l);
    const top = @max(self.top, other_rect.t);
    const right = @min(self.right, other_rect.r);
    const bottom = @min(self.bottom, other_rect.b);

    // Check if the intersection is empty (for floats, use >=)
    if (left >= right or top >= bottom) {
        // Return None
        const none = c.Py_None();
        c.Py_INCREF(none);
        return none;
    }

    // Create intersection rectangle
    const new_args = c.Py_BuildValue("(dddd)", @as(f64, left), @as(f64, top), @as(f64, right), @as(f64, bottom)) orelse return null;
    defer c.Py_DECREF(new_args);

    const new_rect = c.PyObject_CallObject(@ptrCast(&RectangleType), new_args);
    return new_rect;
}

const rectangle_iou_doc =
    \\Calculate the Intersection over Union (IoU) with another rectangle.
    \\
    \\## Parameters
    \\- `other` (Rectangle | tuple[float, float, float, float]): The other rectangle to calculate IoU with
    \\
    \\## Returns
    \\- `float`: IoU value between 0.0 (no overlap) and 1.0 (identical rectangles)
    \\
    \\## Examples
    \\```python
    \\rect1 = Rectangle(0, 0, 100, 100)
    \\rect2 = Rectangle(50, 50, 150, 150)
    \\iou = rect1.iou(rect2)  # Returns ~0.143
    \\
    \\# Can also use a tuple
    \\iou = rect1.iou((0, 0, 100, 100))  # Returns 1.0 (identical)
    \\
    \\# Non-overlapping rectangles
    \\rect3 = Rectangle(200, 200, 250, 250)
    \\iou = rect1.iou(rect3)  # Returns 0.0
    \\```
;

fn rectangle_iou(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var other_obj: ?*c.PyObject = undefined;

    const format = std.fmt.comptimePrint("O", .{});
    if (c.PyArg_ParseTuple(args, format.ptr, &other_obj) == 0) {
        return null;
    }

    // Parse the other rectangle (can be Rectangle or tuple)
    const other_rect = py_utils.parseRectangle(f64, other_obj) catch return null;

    // Convert self to Rectangle(f64)
    const self_rect = Rectangle(f64).init(self.left, self.top, self.right, self.bottom);

    // Calculate IoU
    const iou_value = self_rect.iou(other_rect);

    return c.PyFloat_FromDouble(iou_value);
}

const rectangle_overlaps_doc =
    \\Check if this rectangle overlaps with another based on IoU and coverage thresholds.
    \\
    \\## Parameters
    \\- `other` (Rectangle | tuple[float, float, float, float]): The other rectangle to check overlap with
    \\- `iou_thresh` (float, optional): IoU threshold for considering overlap. Default: 0.5
    \\- `coverage_thresh` (float, optional): Coverage threshold for considering overlap. Default: 1.0
    \\
    \\## Returns
    \\- `bool`: True if rectangles overlap enough based on the thresholds
    \\
    \\## Description
    \\Returns True if any of these conditions are met:
    \\- IoU > iou_thresh
    \\- intersection.area / self.area > coverage_thresh
    \\- intersection.area / other.area > coverage_thresh
    \\
    \\## Examples
    \\```python
    \\rect1 = Rectangle(0, 0, 100, 100)
    \\rect2 = Rectangle(50, 50, 150, 150)
    \\
    \\# Default thresholds
    \\overlaps = rect1.overlaps(rect2)  # Uses IoU > 0.5
    \\
    \\# Custom IoU threshold
    \\overlaps = rect1.overlaps(rect2, iou_thresh=0.1)  # True
    \\
    \\# Coverage threshold (useful for small rectangle inside large)
    \\small = Rectangle(25, 25, 75, 75)
    \\overlaps = rect1.overlaps(small, coverage_thresh=0.9)  # True (small is 100% covered)
    \\
    \\# Can use tuple
    \\overlaps = rect1.overlaps((50, 50, 150, 150), iou_thresh=0.1)
    \\```
;

fn rectangle_overlaps(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwargs: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));

    var other_obj: ?*c.PyObject = undefined;
    var iou_thresh: f64 = 0.5;
    var coverage_thresh: f64 = 1.0;

    // Parse arguments with keywords
    const kw = comptime py_utils.kw(&.{ "other", "iou_thresh", "coverage_thresh" });
    const format = std.fmt.comptimePrint("O|dd", .{});
    if (c.PyArg_ParseTupleAndKeywords(args, kwargs, format.ptr, @ptrCast(@constCast(&kw)), &other_obj, &iou_thresh, &coverage_thresh) == 0) {
        return null;
    }

    // Validate thresholds
    _ = py_utils.validateRange(f64, iou_thresh, 0.0, 1.0, "iou_thresh") catch return null;
    _ = py_utils.validateRange(f64, coverage_thresh, 0.0, 1.0, "coverage_thresh") catch return null;

    // Parse the other rectangle (can be Rectangle or tuple)
    const other_rect = py_utils.parseRectangle(f64, other_obj) catch return null;

    // Convert self to Rectangle(f64)
    const self_rect = Rectangle(f64).init(self.left, self.top, self.right, self.bottom);

    // Check overlap
    const overlaps = self_rect.overlaps(other_rect, iou_thresh, coverage_thresh);

    return @ptrCast(py_utils.getPyBool(overlaps));
}

// Property getters

fn rectangle_get_width(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    const width = self.right - self.left;
    return c.PyFloat_FromDouble(@as(f64, width));
}

fn rectangle_get_height(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = @as(*RectangleObject, @ptrCast(self_obj.?));
    const height = self.bottom - self.top;
    return c.PyFloat_FromDouble(@as(f64, height));
}

pub const rectangle_methods_metadata = [_]stub_metadata.MethodWithMetadata{
    .{
        .name = "init_center",
        .meth = @ptrCast(&rectangle_init_center),
        .flags = c.METH_VARARGS | c.METH_CLASS,
        .doc = rectangle_init_center_doc,
        .params = "cls, x: float, y: float, width: float, height: float",
        .returns = "Rectangle",
    },
    .{
        .name = "is_empty",
        .meth = @ptrCast(&rectangle_is_empty),
        .flags = c.METH_NOARGS,
        .doc = rectangle_is_empty_doc,
        .params = "self",
        .returns = "bool",
    },
    .{
        .name = "area",
        .meth = @ptrCast(&rectangle_area),
        .flags = c.METH_NOARGS,
        .doc = rectangle_area_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "contains",
        .meth = @ptrCast(&rectangle_contains),
        .flags = c.METH_VARARGS,
        .doc = rectangle_contains_doc,
        .params = "self, x: float, y: float",
        .returns = "bool",
    },
    .{
        .name = "grow",
        .meth = @ptrCast(&rectangle_grow),
        .flags = c.METH_VARARGS,
        .doc = rectangle_grow_doc,
        .params = "self, amount: float",
        .returns = "Rectangle",
    },
    .{
        .name = "shrink",
        .meth = @ptrCast(&rectangle_shrink),
        .flags = c.METH_VARARGS,
        .doc = rectangle_shrink_doc,
        .params = "self, amount: float",
        .returns = "Rectangle",
    },
    .{
        .name = "intersect",
        .meth = @ptrCast(&rectangle_intersect),
        .flags = c.METH_VARARGS,
        .doc = rectangle_intersect_doc,
        .params = "self, other: Rectangle | tuple[float, float, float, float]",
        .returns = "Rectangle | None",
    },
    .{
        .name = "iou",
        .meth = @ptrCast(&rectangle_iou),
        .flags = c.METH_VARARGS,
        .doc = rectangle_iou_doc,
        .params = "self, other: Rectangle | tuple[float, float, float, float]",
        .returns = "float",
    },
    .{
        .name = "overlaps",
        .meth = @ptrCast(&rectangle_overlaps),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = rectangle_overlaps_doc,
        .params = "self, other: Rectangle | tuple[float, float, float, float], iou_thresh: float = 0.5, coverage_thresh: float = 1.0",
        .returns = "bool",
    },
};

var rectangle_methods = stub_metadata.toPyMethodDefArray(&rectangle_methods_metadata);

pub const rectangle_properties_metadata = [_]stub_metadata.PropertyWithMetadata{
    .{
        .name = "left",
        .get = py_utils.getterForField(RectangleObject, "left"),
        .set = null,
        .doc = "Left coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "top",
        .get = py_utils.getterForField(RectangleObject, "top"),
        .set = null,
        .doc = "Top coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "right",
        .get = py_utils.getterForField(RectangleObject, "right"),
        .set = null,
        .doc = "Right coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "bottom",
        .get = py_utils.getterForField(RectangleObject, "bottom"),
        .set = null,
        .doc = "Bottom coordinate of the rectangle",
        .type = "float",
    },
    .{
        .name = "width",
        .get = @ptrCast(&rectangle_get_width),
        .set = null,
        .doc = "Width of the rectangle (right - left)",
        .type = "float",
    },
    .{
        .name = "height",
        .get = @ptrCast(&rectangle_get_height),
        .set = null,
        .doc = "Height of the rectangle (bottom - top)",
        .type = "float",
    },
};

// Auto-generate getset for field-backed properties, then append derived ones
fn make_rectangle_getset() [6 + 1]c.PyGetSetDef {
    comptime {
        const auto_defs = py_utils.autoGetSet(RectangleObject, &.{ "left", "top", "right", "bottom" });
        var result: [6 + 1]c.PyGetSetDef = undefined;
        // Copy auto-generated field-backed properties (without their sentinel)
        result[0] = auto_defs[0];
        result[1] = auto_defs[1];
        result[2] = auto_defs[2];
        result[3] = auto_defs[3];
        // Append derived width/height getters
        result[4] = .{
            .name = "width".ptr,
            .get = @ptrCast(@alignCast(&rectangle_get_width)),
            .set = null,
            .doc = "Width of the rectangle (right - left)".ptr,
            .closure = null,
        };
        result[5] = .{
            .name = "height".ptr,
            .get = @ptrCast(@alignCast(&rectangle_get_height)),
            .set = null,
            .doc = "Height of the rectangle (bottom - top)".ptr,
            .closure = null,
        };
        // Sentinel
        result[6] = .{ .name = null, .get = null, .set = null, .doc = null, .closure = null };
        return result;
    }
}

var rectangle_getset = make_rectangle_getset();

// Class documentation - keep it simple
const rectangle_class_doc = "A rectangle defined by its left, top, right, and bottom coordinates.";

// Init documentation - detailed explanation
pub const rectangle_init_doc =
    \\Initialize a Rectangle with specified coordinates.
    \\
    \\Creates a rectangle from its bounding coordinates. The rectangle is defined
    \\by four values: left (x-min), top (y-min), right (x-max), and bottom (y-max).
    \\The right and bottom bounds are exclusive.
    \\
    \\## Parameters
    \\- `left` (float): Left edge x-coordinate (inclusive)
    \\- `top` (float): Top edge y-coordinate (inclusive)
    \\- `right` (float): Right edge x-coordinate (exclusive)
    \\- `bottom` (float): Bottom edge y-coordinate (exclusive)
    \\
    \\## Examples
    \\```python
    \\# Create a rectangle from (10, 20) to (110, 70)
    \\rect = Rectangle(10, 20, 110, 70)
    \\print(rect.width)  # 100.0 (110 - 10)
    \\print(rect.height)  # 50.0 (70 - 20)
    \\print(rect.contains(109.9, 69.9))  # True
    \\print(rect.contains(110, 70))  # False
    \\
    \\# Create a square
    \\square = Rectangle(0, 0, 50, 50)
    \\print(square.width)  # 50.0
    \\```
    \\
    \\## Notes
    \\- The constructor validates that right >= left and bottom >= top
    \\- Use Rectangle.init_center() for center-based construction
    \\- Coordinates follow image convention: origin at top-left, y increases downward
    \\- Right and bottom bounds are exclusive
;

// Special methods metadata for stub generation
pub const rectangle_special_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "__init__",
        .params = "self, left: float, top: float, right: float, bottom: float",
        .returns = "None",
        .doc = rectangle_init_doc,
    },
};

pub var RectangleType = c.PyTypeObject{
    .ob_base = .{
        .ob_base = .{},
        .ob_size = 0,
    },
    .tp_name = "zignal.Rectangle",
    .tp_basicsize = @sizeOf(RectangleObject),
    .tp_dealloc = rectangle_dealloc,
    .tp_repr = rectangle_repr,
    .tp_flags = c.Py_TPFLAGS_DEFAULT,
    .tp_doc = rectangle_class_doc,
    .tp_methods = @ptrCast(&rectangle_methods),
    .tp_getset = @ptrCast(&rectangle_getset),
    .tp_init = rectangle_init,
    .tp_new = rectangle_new,
};

/// Convert a Python Rectangle object to a Zignal Rectangle(f32)
pub fn toZignalRectangle(rect_obj: ?*c.PyObject) !Rectangle(f32) {
    if (rect_obj == null) {
        c.PyErr_SetString(c.PyExc_TypeError, "Rectangle object is null");
        return error.InvalidRectangle;
    }

    // Check if it's a Rectangle instance
    if (c.PyObject_IsInstance(rect_obj, @ptrCast(&RectangleType)) <= 0) {
        c.PyErr_SetString(c.PyExc_TypeError, "Object must be a Rectangle instance");
        return error.InvalidRectangle;
    }

    const rect = @as(*RectangleObject, @ptrCast(rect_obj.?));
    return Rectangle(f32).init(rect.left, rect.top, rect.right, rect.bottom);
}
