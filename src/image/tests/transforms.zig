//! Transform and geometry tests

const std = @import("std");
const expectEqual = std.testing.expectEqual;
const expectEqualDeep = std.testing.expectEqualDeep;
const Image = @import("../image.zig").Image;
const color = @import("../../color.zig");
const Rectangle = @import("../../geometry.zig").Rectangle;
const Point2d = @import("../../geometry/Point.zig").Point2d;

test "getRectangle" {
    var image: Image(color.Rgba) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    const rect = image.getRectangle();
    try expectEqual(rect.width(), image.cols);
    try expectEqual(rect.height(), image.rows);
}

test "copy function with views" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 5, 7);
    defer image.deinit(std.testing.allocator);

    // Fill with pattern
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            image.at(r, c).* = @intCast(r * 10 + c);
        }
    }

    // Create a view
    const view = image.view(.{ .l = 1, .t = 1, .r = 4, .b = 3 });

    // Copy view to new image
    var copied: Image(u8) = try .initAlloc(std.testing.allocator, view.rows, view.cols);
    defer copied.deinit(std.testing.allocator);

    view.copy(copied);

    // Verify copied data matches view
    for (0..view.rows) |r| {
        for (0..view.cols) |c| {
            try expectEqual(view.at(r, c).*, copied.at(r, c).*);
        }
    }

    // Test copy from regular image to view
    var target: Image(u8) = try .initAlloc(std.testing.allocator, 6, 8);
    defer target.deinit(std.testing.allocator);

    // Fill target with different pattern
    for (0..target.rows) |r| {
        for (0..target.cols) |c| {
            target.at(r, c).* = 99;
        }
    }

    // Create view of target
    const target_view = target.view(.{ .l = 2, .t = 2, .r = 5, .b = 4 });

    // Copy original view to target view
    view.copy(target_view);

    // Verify the view area was copied correctly
    for (0..view.rows) |r| {
        for (0..view.cols) |c| {
            try expectEqual(view.at(r, c).*, target_view.at(r, c).*);
        }
    }

    // Verify areas outside the view weren't touched
    try expectEqual(@as(u8, 99), target.at(0, 0).*);
    try expectEqual(@as(u8, 99), target.at(5, 7).*);
}

test "copy function in-place behavior" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 3, 3);
    defer image.deinit(std.testing.allocator);

    // Fill with pattern
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            image.at(r, c).* = @intCast(r * 3 + c);
        }
    }

    // Store original values
    var original_values: [9]u8 = undefined;
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            original_values[r * 3 + c] = image.at(r, c).*;
        }
    }

    // In-place copy should be no-op
    image.copy(image);

    // Values should be unchanged
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            try expectEqual(original_values[r * 3 + c], image.at(r, c).*);
        }
    }
}
test "view" {
    var image: Image(color.Rgba) = try .initAlloc(std.testing.allocator, 21, 13);
    defer image.deinit(std.testing.allocator);
    const rect: Rectangle(usize) = .{ .l = 0, .t = 0, .r = 8, .b = 10 };
    const view = image.view(rect);
    try expectEqual(view.isView(), true);
    try expectEqual(image.isView(), false);
    try expectEqualDeep(rect, view.getRectangle());
}

test "rotateBounds" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 100, 200);
    defer image.deinit(std.testing.allocator);

    // Test 0 degrees - should be same size
    const bounds_0 = image.rotateBounds(0);
    try expectEqual(@as(usize, 200), bounds_0.cols);
    try expectEqual(@as(usize, 100), bounds_0.rows);

    // Test 90 degrees - should be swapped exactly
    const bounds_90 = image.rotateBounds(std.math.pi / 2.0);
    try expectEqual(@as(usize, 100), bounds_90.cols);
    try expectEqual(@as(usize, 200), bounds_90.rows);

    // Test 180 degrees - should be same size
    const bounds_180 = image.rotateBounds(std.math.pi);
    try expectEqual(@as(usize, 200), bounds_180.cols);
    try expectEqual(@as(usize, 100), bounds_180.rows);

    // Test 270 degrees - should be swapped exactly
    const bounds_270 = image.rotateBounds(3.0 * std.math.pi / 2.0);
    try expectEqual(@as(usize, 100), bounds_270.cols);
    try expectEqual(@as(usize, 200), bounds_270.rows);

    // Test 45 degrees - should be larger
    const bounds_45 = image.rotateBounds(std.math.pi / 4.0);
    try expectEqual(bounds_45.cols > 200, true);
    try expectEqual(bounds_45.rows > 100, true);
}

test "rotate orthogonal fast paths" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 3, 4);
    defer image.deinit(std.testing.allocator);

    // Create a pattern to verify correct rotation
    image.at(0, 0).* = 1;
    image.at(0, 1).* = 2;
    image.at(0, 2).* = 3;
    image.at(0, 3).* = 4;
    image.at(1, 0).* = 5;
    image.at(1, 1).* = 6;
    image.at(1, 2).* = 7;
    image.at(1, 3).* = 8;
    image.at(2, 0).* = 9;
    image.at(2, 1).* = 10;
    image.at(2, 2).* = 11;
    image.at(2, 3).* = 12;

    // Test 0 degree rotation
    var rotated_0: Image(u8) = .empty;
    try image.rotate(std.testing.allocator, 0, &rotated_0);
    defer rotated_0.deinit(std.testing.allocator);
    try expectEqual(@as(u8, 1), rotated_0.at(0, 0).*);

    // Test 90 degree rotation
    var rotated_90: Image(u8) = .empty;
    try image.rotate(std.testing.allocator, std.math.pi / 2.0, &rotated_90);
    defer rotated_90.deinit(std.testing.allocator);
    // After 90° rotation, top-left becomes bottom-left
    // Original (0,0)=1 should be at (2,0) in rotated image (accounting for centering)

    // Test 180 degree rotation
    var rotated_180: Image(u8) = .empty;
    try image.rotate(std.testing.allocator, std.math.pi, &rotated_180);
    defer rotated_180.deinit(std.testing.allocator);

    // Test 270 degree rotation
    var rotated_270: Image(u8) = .empty;
    try image.rotate(std.testing.allocator, 3.0 * std.math.pi / 2.0, &rotated_270);
    defer rotated_270.deinit(std.testing.allocator);

    // Verify dimensions are as expected
    try expectEqual(@as(usize, 3), rotated_0.rows);
    try expectEqual(@as(usize, 4), rotated_0.cols);
    // 90° rotation should have exact swapped dimensions
    try expectEqual(@as(usize, 4), rotated_90.rows);
    try expectEqual(@as(usize, 3), rotated_90.cols);
    // 180° rotation should have same dimensions as original
    try expectEqual(@as(usize, 3), rotated_180.rows);
    try expectEqual(@as(usize, 4), rotated_180.cols);
    // 270° rotation should have exact swapped dimensions
    try expectEqual(@as(usize, 4), rotated_270.rows);
    try expectEqual(@as(usize, 3), rotated_270.cols);
}

test "rotate arbitrary angle" {
    var image: Image(u8) = try .initAlloc(std.testing.allocator, 10, 10);
    defer image.deinit(std.testing.allocator);

    // Fill with pattern
    for (0..image.rows) |r| {
        for (0..image.cols) |c| {
            image.at(r, c).* = if ((r + c) % 2 == 0) 255 else 0;
        }
    }

    // Test 45 degree rotation
    var rotated: Image(u8) = .empty;
    try image.rotate(std.testing.allocator, std.math.pi / 4.0, &rotated);
    defer rotated.deinit(std.testing.allocator);

    // Should be larger than original to fit rotated content
    try expectEqual(rotated.rows > 10, true);
    try expectEqual(rotated.cols > 10, true);
}
