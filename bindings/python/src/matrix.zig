const std = @import("std");

const zignal = @import("zignal");
const Matrix = zignal.Matrix;

const py_utils = @import("py_utils.zig");
const allocator = py_utils.allocator;
const c = py_utils.c;
const stub_metadata = @import("stub_metadata.zig");

const matrix_class_doc =
    \\Matrix for numerical computations with f64 (float64) values.
    \\
    \\This class provides a bridge between zignal's Matrix type and NumPy arrays,
    \\with zero-copy operations when possible.
    \\
    \\## Examples
    \\```python
    \\import zignal
    \\import numpy as np
    \\
    \\# Create from list of lists
    \\m = zignal.Matrix([[1, 2, 3], [4, 5, 6]])
    \\
    \\# Create with dimensions using full()
    \\m = zignal.Matrix.full(3, 4)  # 3x4 matrix of zeros
    \\m = zignal.Matrix.full(3, 4, fill_value=1.0)  # filled with 1.0
    \\
    \\# From numpy (zero-copy for float64 contiguous arrays)
    \\arr = np.random.randn(10, 5)
    \\m = zignal.Matrix.from_numpy(arr)
    \\
    \\# To numpy (zero-copy)
    \\arr = m.to_numpy()
    \\```
;

pub const MatrixObject = extern struct {
    ob_base: c.PyObject,
    matrix_ptr: ?*Matrix(f64),
    numpy_ref: ?*c.PyObject, // Reference to numpy array if created from numpy
    owns_memory: bool, // True if we allocated the matrix
};

// Using genericNew helper for standard object creation
const matrix_new = py_utils.genericNew(MatrixObject);

const matrix_init_doc =
    \\Create a new Matrix from a list of lists.
    \\
    \\## Parameters
    \\- `data` (List[List[float]]): List of lists containing matrix data
    \\
    \\## Examples
    \\```python
    \\# Create from list of lists
    \\m = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
    \\m = Matrix([[1.0, 2.5], [3.7, 4.2]])  # 2x2 matrix
    \\```
;

fn matrix_init(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) c_int {
    const self = py_utils.safeCast(MatrixObject, self_obj);

    // Check if already initialized (e.g., from from_numpy)
    if (self.matrix_ptr != null) {
        return 0;
    }

    // Parse single argument: list of lists
    const Params = struct {
        data: ?*c.PyObject,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return -1;

    const list_obj = params.data;

    // Check if it's a list
    if (c.PyList_Check(list_obj) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Matrix data must be a list of lists");
        return -1;
    }

    // Initialize from list of lists
    return matrix_init_from_list(self, list_obj);
}

const matrix_full_doc =
    \\Create a Matrix filled with a specified value.
    \\
    \\## Parameters
    \\- `rows` (int): Number of rows
    \\- `cols` (int): Number of columns
    \\- `fill_value` (float, optional): Value to fill the matrix with (default: 0.0)
    \\
    \\## Returns
    \\Matrix: A new Matrix of the specified dimensions filled with fill_value
    \\
    \\## Examples
    \\```python
    \\# Create 3x4 matrix of zeros
    \\m = Matrix.full(3, 4)
    \\
    \\# Create 3x4 matrix of ones
    \\m = Matrix.full(3, 4, 1.0)
    \\
    \\# Create 5x5 matrix filled with 3.14
    \\m = Matrix.full(5, 5, 3.14)
    \\```
;

fn matrix_full(type_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        rows: c_int,
        cols: c_int,
        fill_value: f64 = 0.0,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const rows = params.rows;
    const cols = params.cols;
    const fill_value = params.fill_value;

    // Validate dimensions
    const rows_pos = py_utils.validatePositive(usize, rows, "rows") catch return null;
    const cols_pos = py_utils.validatePositive(usize, cols, "cols") catch return null;

    // Create new Matrix object
    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) return null;

    // Create the matrix
    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        // TODO(py3.10): remove explicit cast once Python 3.10 support is dropped
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    matrix_ptr.* = Matrix(f64).init(allocator, rows_pos, cols_pos) catch {
        allocator.destroy(matrix_ptr);
        // TODO(py3.10): remove explicit cast once Python 3.10 support is dropped
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("matrix data");
        return null;
    };

    // Fill with value
    @memset(matrix_ptr.items, fill_value);

    self.?.matrix_ptr = matrix_ptr;
    self.?.owns_memory = true;
    self.?.numpy_ref = null;

    return @ptrCast(self);
}

fn matrix_init_from_list(self: *MatrixObject, list_obj: ?*c.PyObject) c_int {
    // Get dimensions
    const n_rows = c.PyList_Size(list_obj);
    if (n_rows == 0) {
        py_utils.setValueError("Cannot create Matrix from empty list", .{});
        return -1;
    }

    // Check first row to get number of columns
    const first_row = c.PyList_GetItem(list_obj, 0);
    if (c.PyList_Check(first_row) != 1) {
        c.PyErr_SetString(c.PyExc_TypeError, "Matrix data must be a list of lists");
        return -1;
    }

    const n_cols = c.PyList_Size(first_row);
    if (n_cols == 0) {
        py_utils.setValueError("Cannot create Matrix with empty rows", .{});
        return -1;
    }

    // Create the matrix
    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        py_utils.setMemoryError("Matrix");
        return -1;
    };

    matrix_ptr.* = Matrix(f64).init(allocator, @intCast(n_rows), @intCast(n_cols)) catch {
        allocator.destroy(matrix_ptr);
        py_utils.setMemoryError("matrix data");
        return -1;
    };

    // Fill matrix with data from list
    var row_idx: usize = 0;
    while (row_idx < @as(usize, @intCast(n_rows))) : (row_idx += 1) {
        const row_obj = c.PyList_GetItem(list_obj, @intCast(row_idx));

        // Check that this is a list
        if (c.PyList_Check(row_obj) != 1) {
            matrix_ptr.deinit();
            allocator.destroy(matrix_ptr);
            py_utils.setTypeError("list", row_obj);
            return -1;
        }

        // Check that row has correct number of columns
        const row_size = c.PyList_Size(row_obj);
        if (row_size != n_cols) {
            matrix_ptr.deinit();
            allocator.destroy(matrix_ptr);
            py_utils.setValueError("All rows must have the same number of columns", .{});
            return -1;
        }

        // Extract values from row
        var col_idx: usize = 0;
        while (col_idx < @as(usize, @intCast(n_cols))) : (col_idx += 1) {
            const item = c.PyList_GetItem(row_obj, @intCast(col_idx));

            // Convert to float
            const float_obj = c.PyFloat_FromDouble(0); // Create a float to check conversion
            defer c.Py_XDECREF(float_obj);

            const value = c.PyFloat_AsDouble(item);
            if (value == -1.0 and c.PyErr_Occurred() != null) {
                matrix_ptr.deinit();
                allocator.destroy(matrix_ptr);
                c.PyErr_Clear();
                c.PyErr_SetString(c.PyExc_TypeError, "Matrix elements must be numeric");
                return -1;
            }

            matrix_ptr.at(row_idx, col_idx).* = value;
        }
    }

    self.matrix_ptr = matrix_ptr;
    self.owns_memory = true;
    self.numpy_ref = null;

    return 0;
}

// Helper function for custom cleanup
fn matrixDeinit(self: *MatrixObject) void {
    // Free the matrix if we own it
    if (self.matrix_ptr) |ptr| {
        if (self.owns_memory) {
            ptr.deinit();
        }
        allocator.destroy(ptr);
    }

    // Decref numpy array if we hold a reference
    if (self.numpy_ref) |ref| {
        c.Py_DECREF(ref);
    }
}

// Using genericDealloc helper
const matrix_dealloc = py_utils.genericDealloc(MatrixObject, matrixDeinit);

fn matrix_repr(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = py_utils.safeCast(MatrixObject, self_obj);

    if (self.matrix_ptr) |ptr| {
        // Create a string representation
        var buffer: [256]u8 = undefined;
        const slice = std.fmt.bufPrintZ(&buffer, "Matrix({} x {}, float64)", .{ ptr.rows, ptr.cols }) catch {
            return c.PyUnicode_FromString("Matrix(error formatting)");
        };
        return c.PyUnicode_FromString(slice.ptr);
    }

    return c.PyUnicode_FromString("Matrix(uninitialized)");
}

fn matrix_str(self_obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = py_utils.safeCast(MatrixObject, self_obj);

    if (self.matrix_ptr) |ptr| {
        // For small matrices, show the actual values
        if (ptr.rows <= 6 and ptr.cols <= 6) {
            var list = std.ArrayList(u8).initCapacity(allocator, 256) catch return null;
            defer list.deinit(allocator);

            list.appendSlice(allocator, "Matrix[\n") catch return null;

            for (0..ptr.rows) |i| {
                list.appendSlice(allocator, "  [") catch return null;
                for (0..ptr.cols) |j| {
                    const val = ptr.at(i, j).*;
                    if (j > 0) list.appendSlice(allocator, ", ") catch return null;
                    var buf: [32]u8 = undefined;
                    const s = std.fmt.bufPrint(&buf, "{d:.4}", .{val}) catch return null;
                    list.appendSlice(allocator, s) catch return null;
                }
                list.appendSlice(allocator, "]\n") catch return null;
            }
            list.append(allocator, ']') catch return null;

            const str = list.toOwnedSlice(allocator) catch return null;
            defer allocator.free(str);
            // Use FromStringAndSize to ensure proper length
            return c.PyUnicode_FromStringAndSize(str.ptr, @intCast(str.len));
        } else {
            // For large matrices, just show dimensions
            return matrix_repr(self_obj);
        }
    }

    return c.PyUnicode_FromString("Matrix(uninitialized)");
}

// Properties
fn matrix_rows_getter(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = py_utils.safeCast(MatrixObject, self_obj);

    if (self.matrix_ptr) |ptr| {
        return c.PyLong_FromLong(@intCast(ptr.rows));
    }

    py_utils.setValueError("Matrix not initialized", .{});
    return null;
}

fn matrix_cols_getter(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = py_utils.safeCast(MatrixObject, self_obj);

    if (self.matrix_ptr) |ptr| {
        return c.PyLong_FromLong(@intCast(ptr.cols));
    }

    py_utils.setValueError("Matrix not initialized", .{});
    return null;
}

fn matrix_shape_getter(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    const self = py_utils.safeCast(MatrixObject, self_obj);

    if (self.matrix_ptr) |ptr| {
        return c.PyTuple_Pack(2, c.PyLong_FromLong(@intCast(ptr.rows)), c.PyLong_FromLong(@intCast(ptr.cols)));
    }

    py_utils.setValueError("Matrix not initialized", .{});
    return null;
}

const matrix_to_numpy_doc =
    \\Convert the matrix to a NumPy array (zero-copy).
    \\
    \\Returns a float64 NumPy array that shares memory with the Matrix.
    \\Modifying the array will modify the Matrix.
    \\
    \\## Returns
    \\NDArray[np.float64]: A 2D NumPy array with shape (rows, cols)
    \\
    \\## Examples
    \\```python
    \\m = Matrix(3, 4, fill_value=1.0)
    \\arr = m.to_numpy()  # shape (3, 4), dtype float64
    \\```
;

fn matrix_to_numpy(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);

    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    // Import numpy
    const np_module = c.PyImport_ImportModule("numpy") orelse {
        py_utils.setImportError("NumPy is not installed. Please install it with: pip install numpy", .{});
        return null;
    };
    defer c.Py_DECREF(np_module);

    // Create a memoryview from our matrix data
    var buffer = c.Py_buffer{
        .buf = @ptrCast(ptr.items.ptr),
        .obj = self_obj,
        .len = @intCast(ptr.rows * ptr.cols * @sizeOf(f64)),
        .itemsize = @sizeOf(f64),
        .readonly = 0,
        .ndim = 2,
        .format = @ptrCast(@constCast("d")), // 'd' for double/float64
        .shape = null,
        .strides = null,
        .suboffsets = null,
        .internal = null,
    };

    // Set up shape and strides
    var shape = [_]c.Py_ssize_t{ @intCast(ptr.rows), @intCast(ptr.cols) };
    var strides = [_]c.Py_ssize_t{ @intCast(ptr.cols * @sizeOf(f64)), @sizeOf(f64) };
    buffer.shape = &shape;
    buffer.strides = &strides;

    // Create memoryview
    const memview = c.PyMemoryView_FromBuffer(&buffer) orelse return null;
    defer c.Py_DECREF(memview);

    // Get numpy.frombuffer function
    const frombuffer = c.PyObject_GetAttrString(np_module, "frombuffer") orelse return null;
    defer c.Py_DECREF(frombuffer);

    // Call numpy.frombuffer with dtype='float64'
    const dtype_str = c.PyUnicode_FromString("float64") orelse return null;
    defer c.Py_DECREF(dtype_str);

    const flat_array = c.PyObject_CallFunctionObjArgs(frombuffer, memview, dtype_str, @as(?*c.PyObject, null)) orelse return null;
    defer c.Py_DECREF(flat_array);

    // Reshape to 2D
    const reshape_method = c.PyObject_GetAttrString(flat_array, "reshape") orelse return null;
    defer c.Py_DECREF(reshape_method);

    const shape_tuple = c.PyTuple_Pack(2, c.PyLong_FromLong(@intCast(ptr.rows)), c.PyLong_FromLong(@intCast(ptr.cols))) orelse return null;
    defer c.Py_DECREF(shape_tuple);

    return c.PyObject_CallObject(reshape_method, shape_tuple);
}

const matrix_from_numpy_doc =
    \\Create a Matrix from a NumPy array (zero-copy when possible).
    \\
    \\The array must be 2D with dtype float64 and be C-contiguous.
    \\If the array is not contiguous or not float64, an error is raised.
    \\
    \\## Parameters
    \\- `array` (NDArray[np.float64]): A 2D NumPy array with dtype float64
    \\
    \\## Returns
    \\Matrix: A new Matrix that shares memory with the NumPy array
    \\
    \\## Examples
    \\```python
    \\import numpy as np
    \\arr = np.random.randn(10, 5)  # float64 by default
    \\m = Matrix.from_numpy(arr)
    \\# Modifying arr will modify m and vice versa
    \\```
;

fn matrix_from_numpy(type_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        array: ?*c.PyObject,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const array_obj = params.array;

    if (array_obj == null or array_obj == c.Py_None()) {
        py_utils.setTypeError("non-None array", array_obj);
        return null;
    }

    // Get buffer interface from the array
    var buffer: c.Py_buffer = undefined;
    buffer = std.mem.zeroes(c.Py_buffer);

    // Request buffer with format and strides info
    const flags: c_int = c.PyBUF_FORMAT | c.PyBUF_ND | c.PyBUF_STRIDES;
    if (c.PyObject_GetBuffer(array_obj, &buffer, flags) != 0) {
        return null;
    }
    defer c.PyBuffer_Release(&buffer);

    // Check dimensions
    if (buffer.ndim != 2) {
        py_utils.setValueError("Array must be 2-dimensional", .{});
        return null;
    }

    // Check dtype - must be float64 ('d')
    if (buffer.format == null or buffer.format[0] != 'd' or buffer.format[1] != 0) {
        py_utils.setTypeError("float64 array", array_obj);
        return null;
    }

    // Check if array is C-contiguous
    const rows: usize = @intCast(buffer.shape[0]);
    const cols: usize = @intCast(buffer.shape[1]);
    const expected_stride_row = cols * @sizeOf(f64);
    const expected_stride_col = @sizeOf(f64);

    if (buffer.strides[0] != @as(c.Py_ssize_t, @intCast(expected_stride_row)) or
        buffer.strides[1] != @as(c.Py_ssize_t, @intCast(expected_stride_col)))
    {
        py_utils.setValueError("Array must be C-contiguous. Use np.ascontiguousarray() to convert.", .{});
        return null;
    }

    // Create new Matrix object
    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) return null;

    // Create Matrix that references the numpy data
    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        // TODO: remove explicit cast when we don't use Python < 3.13
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    // Create a Matrix that uses the numpy buffer's memory
    const data_slice = @as([*]f64, @ptrCast(@alignCast(buffer.buf)))[0 .. rows * cols];
    matrix_ptr.* = Matrix(f64){
        .items = data_slice,
        .rows = rows,
        .cols = cols,
        .allocator = allocator,
    };

    self.?.matrix_ptr = matrix_ptr;
    self.?.numpy_ref = array_obj;
    self.?.owns_memory = false;

    // Increment reference to numpy array to keep it alive
    c.Py_INCREF(array_obj);

    return @ptrCast(self);
}

// Indexing support
fn matrix_getitem(self_obj: ?*c.PyObject, key: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = py_utils.safeCast(MatrixObject, self_obj);

    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    // Parse the key - expecting (row, col) tuple or two integers
    if (c.PyTuple_Check(key) == 1) {
        const size = c.PyTuple_Size(key);
        if (size != 2) {
            py_utils.setIndexError("Matrix indices must be a tuple of two integers", .{});
            return null;
        }

        const row_obj = c.PyTuple_GetItem(key, 0);
        const col_obj = c.PyTuple_GetItem(key, 1);

        const row = c.PyLong_AsLong(row_obj);
        const col = c.PyLong_AsLong(col_obj);

        if (c.PyErr_Occurred() != null) {
            return null;
        }

        // Handle negative indices
        const actual_row: usize = if (row < 0)
            @intCast(@as(i64, @intCast(ptr.rows)) + row)
        else
            @intCast(row);

        const actual_col: usize = if (col < 0)
            @intCast(@as(i64, @intCast(ptr.cols)) + col)
        else
            @intCast(col);

        // Bounds checking
        if (actual_row >= ptr.rows or actual_col >= ptr.cols) {
            py_utils.setIndexError("Matrix index out of bounds", .{});
            return null;
        }

        const value = ptr.at(actual_row, actual_col).*;
        return c.PyFloat_FromDouble(value);
    }

    py_utils.setTypeError("tuple of two integers", key);
    return null;
}

fn matrix_setitem(self_obj: ?*c.PyObject, key: ?*c.PyObject, value: ?*c.PyObject) callconv(.c) c_int {
    const self = py_utils.safeCast(MatrixObject, self_obj);

    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return -1;
    };

    // Parse the key - expecting (row, col) tuple
    if (c.PyTuple_Check(key) == 1) {
        const size = c.PyTuple_Size(key);
        if (size != 2) {
            py_utils.setIndexError("Matrix indices must be a tuple of two integers", .{});
            return -1;
        }

        const row_obj = c.PyTuple_GetItem(key, 0);
        const col_obj = c.PyTuple_GetItem(key, 1);

        const row = c.PyLong_AsLong(row_obj);
        const col = c.PyLong_AsLong(col_obj);

        if (c.PyErr_Occurred() != null) {
            return -1;
        }

        // Handle negative indices
        const actual_row: usize = if (row < 0)
            @intCast(@as(i64, @intCast(ptr.rows)) + row)
        else
            @intCast(row);

        const actual_col: usize = if (col < 0)
            @intCast(@as(i64, @intCast(ptr.cols)) + col)
        else
            @intCast(col);

        // Bounds checking
        if (actual_row >= ptr.rows or actual_col >= ptr.cols) {
            py_utils.setIndexError("Matrix index out of bounds", .{});
            return -1;
        }

        // Get the value as a float
        const val = c.PyFloat_AsDouble(value);
        if (c.PyErr_Occurred() != null) {
            return -1;
        }

        ptr.at(actual_row, actual_col).* = val;
        return 0;
    }

    py_utils.setTypeError("tuple of two integers", key);
    return -1;
}

// Mapping methods for indexing
var matrix_as_mapping = c.PyMappingMethods{
    .mp_length = null,
    .mp_subscript = matrix_getitem,
    .mp_ass_subscript = matrix_setitem,
};

// ===== Operator overloads for number protocol =====

fn matrix_add(left: ?*c.PyObject, right: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    // Check if both are matrices
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1 and
        c.PyObject_IsInstance(right, @ptrCast(&MatrixType)) == 1)
    {
        // Matrix + Matrix: element-wise addition
        const self = py_utils.safeCast(MatrixObject, left);
        const other = py_utils.safeCast(MatrixObject, right);

        const self_ptr = self.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };
        const other_ptr = other.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };

        const result_matrix = self_ptr.add(other_ptr.*);
        return matrixToObject(result_matrix);
    }

    // Check if right is a scalar
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1) {
        const scalar = c.PyFloat_AsDouble(right);
        if (c.PyErr_Occurred() == null) {
            // Matrix + scalar: offset
            const self = py_utils.safeCast(MatrixObject, left);
            const self_ptr = self.matrix_ptr orelse {
                py_utils.setValueError("Matrix not initialized", .{});
                return null;
            };

            const result_matrix = self_ptr.offset(scalar);
            return matrixToObject(result_matrix);
        }
        c.PyErr_Clear();
    }

    c.Py_INCREF(c.Py_NotImplemented());
    return c.Py_NotImplemented();
}

fn matrix_subtract(left: ?*c.PyObject, right: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    // Check if both are matrices
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1 and
        c.PyObject_IsInstance(right, @ptrCast(&MatrixType)) == 1)
    {
        // Matrix - Matrix: element-wise subtraction
        const self = py_utils.safeCast(MatrixObject, left);
        const other = py_utils.safeCast(MatrixObject, right);

        const self_ptr = self.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };
        const other_ptr = other.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };

        const result_matrix = self_ptr.sub(other_ptr.*);
        return matrixToObject(result_matrix);
    }

    // Check if right is a scalar
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1) {
        const scalar = c.PyFloat_AsDouble(right);
        if (c.PyErr_Occurred() == null) {
            // Matrix - scalar: offset by negative
            const self = py_utils.safeCast(MatrixObject, left);
            const self_ptr = self.matrix_ptr orelse {
                py_utils.setValueError("Matrix not initialized", .{});
                return null;
            };

            const result_matrix = self_ptr.offset(-scalar);
            return matrixToObject(result_matrix);
        }
        c.PyErr_Clear();
    }

    c.Py_INCREF(c.Py_NotImplemented());
    return c.Py_NotImplemented();
}

fn matrix_multiply(left: ?*c.PyObject, right: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    // Check if both are matrices
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1 and
        c.PyObject_IsInstance(right, @ptrCast(&MatrixType)) == 1)
    {
        // Matrix * Matrix: element-wise multiplication
        const self = py_utils.safeCast(MatrixObject, left);
        const other = py_utils.safeCast(MatrixObject, right);

        const self_ptr = self.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };
        const other_ptr = other.matrix_ptr orelse {
            py_utils.setValueError("Matrix not initialized", .{});
            return null;
        };

        const result_matrix = self_ptr.times(other_ptr.*);
        return matrixToObject(result_matrix);
    }

    // Check if left is matrix and right is scalar
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) == 1) {
        const scalar = c.PyFloat_AsDouble(right);
        if (c.PyErr_Occurred() == null) {
            const self = py_utils.safeCast(MatrixObject, left);
            const self_ptr = self.matrix_ptr orelse {
                py_utils.setValueError("Matrix not initialized", .{});
                return null;
            };

            const result_matrix = self_ptr.scale(scalar);
            return matrixToObject(result_matrix);
        }
        c.PyErr_Clear();
    }

    // Check if left is scalar and right is matrix (for rmul)
    if (c.PyObject_IsInstance(right, @ptrCast(&MatrixType)) == 1) {
        const scalar = c.PyFloat_AsDouble(left);
        if (c.PyErr_Occurred() == null) {
            const self = py_utils.safeCast(MatrixObject, right);
            const self_ptr = self.matrix_ptr orelse {
                py_utils.setValueError("Matrix not initialized", .{});
                return null;
            };

            const result_matrix = self_ptr.scale(scalar);
            return matrixToObject(result_matrix);
        }
        c.PyErr_Clear();
    }

    c.Py_INCREF(c.Py_NotImplemented());
    return c.Py_NotImplemented();
}

fn matrix_matmul(left: ?*c.PyObject, right: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) != 1 or
        c.PyObject_IsInstance(right, @ptrCast(&MatrixType)) != 1)
    {
        c.Py_INCREF(c.Py_NotImplemented());
        return c.Py_NotImplemented();
    }

    const self = py_utils.safeCast(MatrixObject, left);
    const other = py_utils.safeCast(MatrixObject, right);

    const self_ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };
    const other_ptr = other.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = self_ptr.dot(other_ptr.*);
    return matrixToObject(result_matrix);
}

fn matrix_truediv(left: ?*c.PyObject, right: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    if (c.PyObject_IsInstance(left, @ptrCast(&MatrixType)) != 1) {
        c.Py_INCREF(c.Py_NotImplemented());
        return c.Py_NotImplemented();
    }

    const scalar = c.PyFloat_AsDouble(right);
    if (c.PyErr_Occurred() != null) {
        c.Py_INCREF(c.Py_NotImplemented());
        return c.Py_NotImplemented();
    }

    if (scalar == 0.0) {
        c.PyErr_SetString(c.PyExc_ZeroDivisionError, "Cannot divide matrix by zero");
        return null;
    }

    const self = py_utils.safeCast(MatrixObject, left);
    const self_ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = self_ptr.scale(1.0 / scalar);
    return matrixToObject(result_matrix);
}

fn matrix_negative(obj: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const self = py_utils.safeCast(MatrixObject, obj);
    const self_ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = self_ptr.scale(-1.0);
    return matrixToObject(result_matrix);
}

// Helper function to convert Zig Matrix to Python MatrixObject
fn matrixToObject(matrix: Matrix(f64)) ?*c.PyObject {
    if (matrix.err) |e| {
        switch (e) {
            error.DimensionMismatch => py_utils.setValueError("Matrix dimension mismatch", .{}),
            error.NotSquare => py_utils.setValueError("Matrix must be square", .{}),
            error.Singular => py_utils.setValueError("Matrix is singular", .{}),
            error.OutOfBounds => py_utils.setIndexError("Matrix index out of bounds", .{}),
            error.OutOfMemory => py_utils.setMemoryError("Matrix"),
            error.NotConverged => py_utils.setValueError("Matrix operation did not converge", .{}),
        }
        return null;
    }

    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(&MatrixType), 0)));
    if (self == null) return null;

    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    matrix_ptr.* = matrix;
    self.?.matrix_ptr = matrix_ptr;
    self.?.owns_memory = true;
    self.?.numpy_ref = null;

    return @ptrCast(self);
}

// Number protocol for arithmetic operations
var matrix_as_number = c.PyNumberMethods{
    .nb_add = matrix_add,
    .nb_subtract = matrix_subtract,
    .nb_multiply = matrix_multiply,
    .nb_remainder = null,
    .nb_divmod = null,
    .nb_power = null,
    .nb_negative = matrix_negative,
    .nb_positive = null,
    .nb_absolute = null,
    .nb_bool = null,
    .nb_invert = null,
    .nb_lshift = null,
    .nb_rshift = null,
    .nb_and = null,
    .nb_xor = null,
    .nb_or = null,
    .nb_int = null,
    .nb_reserved = null,
    .nb_float = null,
    .nb_inplace_add = null,
    .nb_inplace_subtract = null,
    .nb_inplace_multiply = null,
    .nb_inplace_remainder = null,
    .nb_inplace_power = null,
    .nb_inplace_lshift = null,
    .nb_inplace_rshift = null,
    .nb_inplace_and = null,
    .nb_inplace_xor = null,
    .nb_inplace_or = null,
    .nb_floor_divide = null,
    .nb_true_divide = matrix_truediv,
    .nb_inplace_floor_divide = null,
    .nb_inplace_true_divide = null,
    .nb_index = null,
    .nb_matrix_multiply = matrix_matmul,
    .nb_inplace_matrix_multiply = null,
};

// ===== Matrix methods =====

const matrix_transpose_doc =
    \\Transpose the matrix.
    \\
    \\## Returns
    \\Matrix: A new transposed matrix where rows and columns are swapped
    \\
    \\## Examples
    \\```python
    \\m = Matrix([[1, 2, 3], [4, 5, 6]])
    \\t = m.transpose()  # shape (3, 2)
    \\```
;

fn matrix_transpose(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.transpose();
    return matrixToObject(result_matrix);
}

const matrix_inverse_doc =
    \\Compute the matrix inverse.
    \\
    \\## Returns
    \\Matrix: The inverse matrix such that A @ A.inverse() ≈ I
    \\
    \\## Raises
    \\ValueError: If matrix is not square or is singular
    \\
    \\## Examples
    \\```python
    \\m = Matrix([[2, 0], [0, 2]])
    \\inv = m.inverse()  # [[0.5, 0], [0, 0.5]]
    \\```
;

fn matrix_inverse(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.inverse();
    return matrixToObject(result_matrix);
}

const matrix_dot_doc =
    \\Matrix multiplication (dot product).
    \\
    \\## Parameters
    \\- `other` (Matrix): Matrix to multiply with
    \\
    \\## Returns
    \\Matrix: Result of matrix multiplication
    \\
    \\## Examples
    \\```python
    \\a = Matrix([[1, 2], [3, 4]])
    \\b = Matrix([[5, 6], [7, 8]])
    \\c = a.dot(b)  # or a @ b
    \\```
;

fn matrix_dot_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        other: ?*c.PyObject,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    if (c.PyObject_IsInstance(params.other, @ptrCast(&MatrixType)) != 1) {
        py_utils.setTypeError("Matrix", params.other);
        return null;
    }

    return matrix_matmul(self_obj, params.other);
}

const matrix_sum_doc =
    \\Sum of all matrix elements.
    \\
    \\## Returns
    \\float: The sum of all elements
    \\
    \\## Examples
    \\```python
    \\m = Matrix([[1, 2], [3, 4]])
    \\s = m.sum()  # 10.0
    \\```
;

fn matrix_sum_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.sum();
    return c.PyFloat_FromDouble(result);
}

const matrix_mean_doc =
    \\Mean (average) of all matrix elements.
    \\
    \\## Returns
    \\float: The mean of all elements
;

fn matrix_mean_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.mean();
    return c.PyFloat_FromDouble(result);
}

const matrix_min_doc =
    \\Minimum element in the matrix.
    \\
    \\## Returns
    \\float: The minimum value
;

fn matrix_min_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.min();
    return c.PyFloat_FromDouble(result);
}

const matrix_max_doc =
    \\Maximum element in the matrix.
    \\
    \\## Returns
    \\float: The maximum value
;

fn matrix_max_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.max();
    return c.PyFloat_FromDouble(result);
}

const matrix_trace_doc =
    \\Sum of diagonal elements (trace).
    \\
    \\## Returns
    \\float: The trace of the matrix
    \\
    \\## Raises
    \\ValueError: If matrix is not square
;

fn matrix_trace_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    if (ptr.rows != ptr.cols) {
        py_utils.setValueError("Matrix must be square to compute trace", .{});
        return null;
    }

    const result = ptr.trace();
    return c.PyFloat_FromDouble(result);
}

const matrix_identity_doc =
    \\Create an identity matrix.
    \\
    \\## Parameters
    \\- `rows` (int): Number of rows
    \\- `cols` (int): Number of columns
    \\
    \\## Returns
    \\Matrix: Identity matrix with ones on diagonal
;

fn matrix_identity(type_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        rows: c_int,
        cols: c_int,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const rows_pos = py_utils.validatePositive(usize, params.rows, "rows") catch return null;
    const cols_pos = py_utils.validatePositive(usize, params.cols, "cols") catch return null;

    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) return null;

    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    matrix_ptr.* = Matrix(f64).identity(allocator, rows_pos, cols_pos) catch {
        allocator.destroy(matrix_ptr);
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("matrix data");
        return null;
    };

    self.?.matrix_ptr = matrix_ptr;
    self.?.owns_memory = true;
    self.?.numpy_ref = null;

    return @ptrCast(self);
}

const matrix_zeros_doc =
    \\Create a matrix filled with zeros.
    \\
    \\## Parameters
    \\- `rows` (int): Number of rows
    \\- `cols` (int): Number of columns
    \\
    \\## Returns
    \\Matrix: Matrix filled with zeros
;

fn matrix_zeros(type_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        rows: c_int,
        cols: c_int,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const rows_pos = py_utils.validatePositive(usize, params.rows, "rows") catch return null;
    const cols_pos = py_utils.validatePositive(usize, params.cols, "cols") catch return null;

    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) return null;

    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    matrix_ptr.* = Matrix(f64).init(allocator, rows_pos, cols_pos) catch {
        allocator.destroy(matrix_ptr);
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("matrix data");
        return null;
    };

    @memset(matrix_ptr.items, 0.0);

    self.?.matrix_ptr = matrix_ptr;
    self.?.owns_memory = true;
    self.?.numpy_ref = null;

    return @ptrCast(self);
}

const matrix_ones_doc =
    \\Create a matrix filled with ones.
    \\
    \\## Parameters
    \\- `rows` (int): Number of rows
    \\- `cols` (int): Number of columns
    \\
    \\## Returns
    \\Matrix: Matrix filled with ones
;

fn matrix_ones(type_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        rows: c_int,
        cols: c_int,
        fill_value: f64 = 1.0,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const rows_pos = py_utils.validatePositive(usize, params.rows, "rows") catch return null;
    const cols_pos = py_utils.validatePositive(usize, params.cols, "cols") catch return null;

    const self = @as(?*MatrixObject, @ptrCast(c.PyType_GenericAlloc(@ptrCast(type_obj), 0)));
    if (self == null) return null;

    const matrix_ptr = allocator.create(Matrix(f64)) catch {
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("Matrix");
        return null;
    };

    matrix_ptr.* = Matrix(f64).init(allocator, rows_pos, cols_pos) catch {
        allocator.destroy(matrix_ptr);
        c.Py_DECREF(@as(?*c.PyObject, @ptrCast(self)));
        py_utils.setMemoryError("matrix data");
        return null;
    };

    @memset(matrix_ptr.items, 1.0);

    self.?.matrix_ptr = matrix_ptr;
    self.?.owns_memory = true;
    self.?.numpy_ref = null;

    return @ptrCast(self);
}

const matrix_copy_doc =
    \\Create a copy of the matrix.
    \\
    \\## Returns
    \\Matrix: A new matrix with the same values
;

fn matrix_copy_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.dupe(allocator) catch {
        py_utils.setMemoryError("Matrix copy");
        return null;
    };

    return matrixToObject(result_matrix);
}

const matrix_determinant_doc =
    \\Compute the determinant of the matrix.
    \\
    \\## Returns
    \\float: The determinant value
    \\
    \\## Raises
    \\ValueError: If matrix is not square
;

fn matrix_determinant_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    if (ptr.rows != ptr.cols) {
        py_utils.setValueError("Matrix must be square to compute determinant", .{});
        return null;
    }

    const result = ptr.determinant() catch {
        py_utils.setMemoryError("determinant computation");
        return null;
    };

    return c.PyFloat_FromDouble(result);
}

const matrix_gram_doc =
    \\Compute Gram matrix (X × X^T).
    \\
    \\## Returns
    \\Matrix: The Gram matrix (rows × rows)
;

fn matrix_gram_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.gram();
    return matrixToObject(result_matrix);
}

const matrix_covariance_doc =
    \\Compute covariance matrix (X^T × X).
    \\
    \\## Returns
    \\Matrix: The covariance matrix (cols × cols)
;

fn matrix_covariance_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.covariance();
    return matrixToObject(result_matrix);
}

const matrix_norm_doc =
    \\Compute matrix norm.
    \\
    \\## Parameters
    \\- `kind` (str, optional): Norm type - 'fro' (Frobenius, default), 'l1', 'max'
    \\
    \\## Returns
    \\float: The norm value
;

fn matrix_norm_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    // Parse with optional string argument
    const Params = struct {
        kind: ?*c.PyObject = null,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    // Default to Frobenius norm
    var kind_str: []const u8 = "fro";
    if (params.kind) |kind_obj| {
        if (kind_obj != c.Py_None()) {
            const kind_cstr = c.PyUnicode_AsUTF8(kind_obj);
            if (kind_cstr == null) {
                py_utils.setTypeError("string for norm kind", kind_obj);
                return null;
            }
            kind_str = std.mem.span(kind_cstr);
        }
    }

    const result = if (std.mem.eql(u8, kind_str, "fro"))
        ptr.frobeniusNorm()
    else if (std.mem.eql(u8, kind_str, "l1"))
        ptr.l1Norm()
    else if (std.mem.eql(u8, kind_str, "max"))
        ptr.maxNorm()
    else {
        py_utils.setValueError("Unknown norm type: {s}. Use 'fro', 'l1', or 'max'", .{kind_str});
        return null;
    };

    return c.PyFloat_FromDouble(result);
}

const matrix_variance_doc =
    \\Compute variance of all matrix elements.
    \\
    \\## Returns
    \\float: The variance
;

fn matrix_variance_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.variance();
    return c.PyFloat_FromDouble(result);
}

const matrix_std_doc =
    \\Compute standard deviation of all matrix elements.
    \\
    \\## Returns
    \\float: The standard deviation
;

fn matrix_std_method(self_obj: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = args;
    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result = ptr.stdDev();
    return c.PyFloat_FromDouble(result);
}

const matrix_pow_doc =
    \\Raise all elements to power n (element-wise).
    \\
    \\## Parameters
    \\- `n` (float): The exponent
    \\
    \\## Returns
    \\Matrix: Matrix with elements raised to power n
;

fn matrix_pow_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        n: f64,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const result_matrix = ptr.pow(params.n);
    return matrixToObject(result_matrix);
}

const matrix_row_doc =
    \\Extract a row as a column vector.
    \\
    \\## Parameters
    \\- `idx` (int): Row index
    \\
    \\## Returns
    \\Matrix: Column vector containing the row
;

fn matrix_row_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        idx: c_int,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const idx: usize = if (params.idx < 0)
        @intCast(@as(i64, @intCast(ptr.rows)) + params.idx)
    else
        @intCast(params.idx);

    if (idx >= ptr.rows) {
        py_utils.setIndexError("Row index out of bounds", .{});
        return null;
    }

    const result_matrix = ptr.row(idx);
    return matrixToObject(result_matrix);
}

const matrix_col_doc =
    \\Extract a column as a column vector.
    \\
    \\## Parameters
    \\- `idx` (int): Column index
    \\
    \\## Returns
    \\Matrix: Column vector
;

fn matrix_col_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        idx: c_int,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    const idx: usize = if (params.idx < 0)
        @intCast(@as(i64, @intCast(ptr.cols)) + params.idx)
    else
        @intCast(params.idx);

    if (idx >= ptr.cols) {
        py_utils.setIndexError("Column index out of bounds", .{});
        return null;
    }

    const result_matrix = ptr.col(idx);
    return matrixToObject(result_matrix);
}

const matrix_submatrix_doc =
    \\Extract a submatrix.
    \\
    \\## Parameters
    \\- `row_start` (int): Starting row index
    \\- `col_start` (int): Starting column index
    \\- `row_count` (int): Number of rows
    \\- `col_count` (int): Number of columns
    \\
    \\## Returns
    \\Matrix: Submatrix
;

fn matrix_submatrix_method(self_obj: ?*c.PyObject, args: ?*c.PyObject, kwds: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    const Params = struct {
        row_start: c_int,
        col_start: c_int,
        row_count: c_int,
        col_count: c_int,
    };
    var params: Params = undefined;
    py_utils.parseArgs(Params, args, kwds, &params) catch return null;

    const self = py_utils.safeCast(MatrixObject, self_obj);
    const ptr = self.matrix_ptr orelse {
        py_utils.setValueError("Matrix not initialized", .{});
        return null;
    };

    // Starting indices can be zero, but counts must be positive
    if (params.row_start < 0) {
        py_utils.setValueError("row_start must be non-negative", .{});
        return null;
    }
    if (params.col_start < 0) {
        py_utils.setValueError("col_start must be non-negative", .{});
        return null;
    }

    const row_start_pos: usize = @intCast(params.row_start);
    const col_start_pos: usize = @intCast(params.col_start);
    const row_count_pos = py_utils.validatePositive(usize, params.row_count, "row_count") catch return null;
    const col_count_pos = py_utils.validatePositive(usize, params.col_count, "col_count") catch return null;

    const result_matrix = ptr.subMatrix(row_start_pos, col_start_pos, row_count_pos, col_count_pos);
    return matrixToObject(result_matrix);
}

// Metadata for stub generation
pub const matrix_methods_metadata = [_]stub_metadata.MethodWithMetadata{
    .{
        .name = "full",
        .meth = @ptrCast(&matrix_full),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_CLASS,
        .doc = matrix_full_doc,
        .params = "cls, rows: int, cols: int, fill_value: float = 0.0",
        .returns = "Matrix",
    },
    .{
        .name = "from_numpy",
        .meth = @ptrCast(&matrix_from_numpy),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_CLASS,
        .doc = matrix_from_numpy_doc,
        .params = "cls, array: NDArray[np.float64]",
        .returns = "Matrix",
    },
    .{
        .name = "identity",
        .meth = @ptrCast(&matrix_identity),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_CLASS,
        .doc = matrix_identity_doc,
        .params = "cls, rows: int, cols: int",
        .returns = "Matrix",
    },
    .{
        .name = "zeros",
        .meth = @ptrCast(&matrix_zeros),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_CLASS,
        .doc = matrix_zeros_doc,
        .params = "cls, rows: int, cols: int",
        .returns = "Matrix",
    },
    .{
        .name = "ones",
        .meth = @ptrCast(&matrix_ones),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS | c.METH_CLASS,
        .doc = matrix_ones_doc,
        .params = "cls, rows: int, cols: int",
        .returns = "Matrix",
    },
    .{
        .name = "to_numpy",
        .meth = @ptrCast(&matrix_to_numpy),
        .flags = c.METH_NOARGS,
        .doc = matrix_to_numpy_doc,
        .params = "self",
        .returns = "NDArray[np.float64]",
    },
    .{
        .name = "transpose",
        .meth = @ptrCast(&matrix_transpose),
        .flags = c.METH_NOARGS,
        .doc = matrix_transpose_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "inverse",
        .meth = @ptrCast(&matrix_inverse),
        .flags = c.METH_NOARGS,
        .doc = matrix_inverse_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "dot",
        .meth = @ptrCast(&matrix_dot_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_dot_doc,
        .params = "self, other: Matrix",
        .returns = "Matrix",
    },
    .{
        .name = "sum",
        .meth = @ptrCast(&matrix_sum_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_sum_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "mean",
        .meth = @ptrCast(&matrix_mean_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_mean_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "min",
        .meth = @ptrCast(&matrix_min_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_min_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "max",
        .meth = @ptrCast(&matrix_max_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_max_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "trace",
        .meth = @ptrCast(&matrix_trace_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_trace_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "copy",
        .meth = @ptrCast(&matrix_copy_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_copy_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "determinant",
        .meth = @ptrCast(&matrix_determinant_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_determinant_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "det",
        .meth = @ptrCast(&matrix_determinant_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_determinant_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "gram",
        .meth = @ptrCast(&matrix_gram_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_gram_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "covariance",
        .meth = @ptrCast(&matrix_covariance_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_covariance_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "cov",
        .meth = @ptrCast(&matrix_covariance_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_covariance_doc,
        .params = "self",
        .returns = "Matrix",
    },
    .{
        .name = "norm",
        .meth = @ptrCast(&matrix_norm_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_norm_doc,
        .params = "self, kind: str = 'fro'",
        .returns = "float",
    },
    .{
        .name = "variance",
        .meth = @ptrCast(&matrix_variance_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_variance_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "std",
        .meth = @ptrCast(&matrix_std_method),
        .flags = c.METH_NOARGS,
        .doc = matrix_std_doc,
        .params = "self",
        .returns = "float",
    },
    .{
        .name = "pow",
        .meth = @ptrCast(&matrix_pow_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_pow_doc,
        .params = "self, n: float",
        .returns = "Matrix",
    },
    .{
        .name = "row",
        .meth = @ptrCast(&matrix_row_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_row_doc,
        .params = "self, idx: int",
        .returns = "Matrix",
    },
    .{
        .name = "col",
        .meth = @ptrCast(&matrix_col_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_col_doc,
        .params = "self, idx: int",
        .returns = "Matrix",
    },
    .{
        .name = "submatrix",
        .meth = @ptrCast(&matrix_submatrix_method),
        .flags = c.METH_VARARGS | c.METH_KEYWORDS,
        .doc = matrix_submatrix_doc,
        .params = "self, row_start: int, col_start: int, row_count: int, col_count: int",
        .returns = "Matrix",
    },
};

var matrix_methods = stub_metadata.toPyMethodDefArray(&matrix_methods_metadata);

fn matrix_T_getter(self_obj: ?*c.PyObject, closure: ?*anyopaque) callconv(.c) ?*c.PyObject {
    _ = closure;
    return matrix_transpose(self_obj, null);
}

pub const matrix_properties_metadata = [_]stub_metadata.PropertyWithMetadata{
    .{
        .name = "rows",
        .get = matrix_rows_getter,
        .set = null,
        .doc = "Number of rows",
        .type = "int",
    },
    .{
        .name = "cols",
        .get = matrix_cols_getter,
        .set = null,
        .doc = "Number of columns",
        .type = "int",
    },
    .{
        .name = "shape",
        .get = matrix_shape_getter,
        .set = null,
        .doc = "Shape as (rows, cols) tuple",
        .type = "tuple[int, int]",
    },
    .{
        .name = "dtype",
        .get = @ptrCast(@alignCast(py_utils.getterStaticString("float64"))),
        .set = null,
        .doc = "Data type (always 'float64')",
        .type = "str",
    },
    .{
        .name = "T",
        .get = matrix_T_getter,
        .set = null,
        .doc = "Transpose of the matrix (read-only property)",
        .type = "Matrix",
    },
};

var matrix_getset = stub_metadata.toPyGetSetDefArray(&matrix_properties_metadata);

pub const matrix_special_methods_metadata = [_]stub_metadata.MethodInfo{
    .{
        .name = "__init__",
        .params = "self, data: list[list[float]]",
        .returns = "None",
        .doc = matrix_init_doc,
    },
    .{
        .name = "__getitem__",
        .params = "self, key: tuple[int, int]",
        .returns = "float",
        .doc = "Get matrix element at (row, col)",
    },
    .{
        .name = "__setitem__",
        .params = "self, key: tuple[int, int], value: float",
        .returns = "None",
        .doc = "Set matrix element at (row, col)",
    },
    .{
        .name = "__repr__",
        .params = "self",
        .returns = "str",
        .doc = null,
    },
    .{
        .name = "__str__",
        .params = "self",
        .returns = "str",
        .doc = null,
    },
    .{
        .name = "__add__",
        .params = "self, other: Matrix | float",
        .returns = "Matrix",
        .doc = "Element-wise addition or scalar offset",
    },
    .{
        .name = "__radd__",
        .params = "self, other: float",
        .returns = "Matrix",
        .doc = "Reverse addition (scalar + matrix)",
    },
    .{
        .name = "__sub__",
        .params = "self, other: Matrix | float",
        .returns = "Matrix",
        .doc = "Element-wise subtraction or scalar offset",
    },
    .{
        .name = "__rsub__",
        .params = "self, other: float",
        .returns = "Matrix",
        .doc = "Reverse subtraction (scalar - matrix)",
    },
    .{
        .name = "__mul__",
        .params = "self, other: Matrix | float",
        .returns = "Matrix",
        .doc = "Element-wise multiplication or scalar multiplication",
    },
    .{
        .name = "__rmul__",
        .params = "self, other: float",
        .returns = "Matrix",
        .doc = "Reverse multiplication (scalar * matrix)",
    },
    .{
        .name = "__truediv__",
        .params = "self, other: float",
        .returns = "Matrix",
        .doc = "Scalar division",
    },
    .{
        .name = "__matmul__",
        .params = "self, other: Matrix",
        .returns = "Matrix",
        .doc = "Matrix multiplication",
    },
    .{
        .name = "__neg__",
        .params = "self",
        .returns = "Matrix",
        .doc = "Unary negation",
    },
};

// Using buildTypeObject helper for cleaner initialization
pub var MatrixType = py_utils.buildTypeObject(.{
    .name = "zignal.Matrix",
    .basicsize = @sizeOf(MatrixObject),
    .doc = matrix_class_doc,
    .methods = @ptrCast(&matrix_methods),
    .getset = @ptrCast(&matrix_getset),
    .init = matrix_init,
    .new = matrix_new,
    .dealloc = matrix_dealloc,
    .repr = matrix_repr,
    .str = matrix_str,
    .as_mapping = &matrix_as_mapping,
    .as_number = &matrix_as_number,
});
