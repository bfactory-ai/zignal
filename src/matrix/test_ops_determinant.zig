const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Matrix = @import("Matrix.zig").Matrix;

test "Matrix determinant - small matrices" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 1x1 matrix
    const mat1: Matrix(f64) = try .init(arena.allocator(), 1, 1);
    mat1.at(0, 0).* = 5.0;
    try expectEqual(@as(f64, 5.0), try mat1.determinant());

    // Test 2x2 matrix
    const mat2: Matrix(f64) = try .init(arena.allocator(), 2, 2);
    mat2.at(0, 0).* = 4.0;
    mat2.at(0, 1).* = 7.0;
    mat2.at(1, 0).* = 2.0;
    mat2.at(1, 1).* = 6.0;
    // det = 4*6 - 7*2 = 24 - 14 = 10
    try expectEqual(@as(f64, 10.0), try mat2.determinant());

    // Test 3x3 matrix
    const mat3: Matrix(f64) = try .init(arena.allocator(), 3, 3);
    mat3.at(0, 0).* = 1.0;
    mat3.at(0, 1).* = 2.0;
    mat3.at(0, 2).* = 3.0;
    mat3.at(1, 0).* = 0.0;
    mat3.at(1, 1).* = 1.0;
    mat3.at(1, 2).* = 4.0;
    mat3.at(2, 0).* = 5.0;
    mat3.at(2, 1).* = 6.0;
    mat3.at(2, 2).* = 0.0;
    // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //     = 1*(-24) - 2*(-20) + 3*(-5)
    //     = -24 + 40 - 15 = 1
    try expectEqual(@as(f64, 1.0), try mat3.determinant());
}

test "Matrix determinant - large matrices using LU" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();

    // Test 4x4 matrix
    var mat4: Matrix(f64) = try .init(arena.allocator(), 4, 4);
    mat4.at(0, 0).* = 5.0;
    mat4.at(0, 1).* = 1.0;
    mat4.at(0, 2).* = 0.0;
    mat4.at(0, 3).* = 2.0;
    mat4.at(1, 0).* = 1.0;
    mat4.at(1, 1).* = 4.0;
    mat4.at(1, 2).* = 1.0;
    mat4.at(1, 3).* = 1.0;
    mat4.at(2, 0).* = 0.0;
    mat4.at(2, 1).* = 1.0;
    mat4.at(2, 2).* = 3.0;
    mat4.at(2, 3).* = 0.0;
    mat4.at(3, 0).* = 2.0;
    mat4.at(3, 1).* = 1.0;
    mat4.at(3, 2).* = 0.0;
    mat4.at(3, 3).* = 4.0;

    const det4 = try mat4.determinant();

    // This matrix should have a non-zero determinant
    try std.testing.expect(@abs(det4) > 1e-10);

    // Test singular matrix (determinant should be 0)
    var sing: Matrix(f64) = try .init(arena.allocator(), 4, 4);
    sing.at(0, 0).* = 1.0;
    sing.at(0, 1).* = 2.0;
    sing.at(0, 2).* = 3.0;
    sing.at(0, 3).* = 4.0;
    sing.at(1, 0).* = 2.0;
    sing.at(1, 1).* = 4.0;
    sing.at(1, 2).* = 6.0;
    sing.at(1, 3).* = 8.0; // Row 2 = 2 * Row 1
    sing.at(2, 0).* = 3.0;
    sing.at(2, 1).* = 5.0;
    sing.at(2, 2).* = 7.0;
    sing.at(2, 3).* = 9.0;
    sing.at(3, 0).* = 4.0;
    sing.at(3, 1).* = 6.0;
    sing.at(3, 2).* = 8.0;
    sing.at(3, 3).* = 10.0;

    const det_sing = try sing.determinant();

    // Singular matrix should have determinant 0
    try std.testing.expect(@abs(det_sing) < 1e-10);

    // Test 5x5 identity matrix (determinant should be 1)
    var identity5: Matrix(f64) = try .init(arena.allocator(), 5, 5);
    @memset(identity5.items, 0);
    for (0..5) |i| {
        identity5.at(i, i).* = 1.0;
    }

    const det_id = try identity5.determinant();

    try expectEqual(@as(f64, 1.0), det_id);
}
