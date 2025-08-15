const std = @import("std");
const zignal = @import("zignal");
const Image = zignal.Image;
const Rgb = zignal.Rgb;

// Helper function to normalize DoG results for better visualization
fn normalizeForVisualization(img: Image(Rgb)) void {
    for (0..img.rows) |r| {
        for (0..img.cols) |c| {
            const pixel = img.at(r, c);
            pixel.r +|= 128;
            pixel.g +|= 128;
            pixel.b +|= 128;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load test image
    var img = try Image(Rgb).load(allocator, "../assets/liza.jpg");
    defer img.deinit(allocator);

    std.debug.print("Loaded image: {}x{}\n", .{ img.cols, img.rows });

    // Test Gaussian blur with different sigmas
    {
        var blurred: Image(Rgb) = .empty;
        try img.blurGaussian(allocator, 2.0, &blurred);
        defer blurred.deinit(allocator);
        try blurred.save(allocator, "gaussian_blur_2.png");
        std.debug.print("Saved gaussian_blur_2.png\n", .{});
    }

    {
        var blurred: Image(Rgb) = .empty;
        try img.blurGaussian(allocator, 5.0, &blurred);
        defer blurred.deinit(allocator);
        try blurred.save(allocator, "gaussian_blur_5.png");
        std.debug.print("Saved gaussian_blur_5.png\n", .{});
    }

    // Test edge detection with Sobel
    {
        var edges: Image(u8) = .empty;
        try img.sobel(allocator, &edges);
        defer edges.deinit(allocator);
        try edges.save(allocator, "sobel_edges.png");
        std.debug.print("Saved sobel_edges.png\n", .{});
    }

    // Test custom convolution - sharpen kernel
    {
        const sharpen_kernel = [3][3]f32{
            .{ 0, -1, 0 },
            .{ -1, 5, -1 },
            .{ 0, -1, 0 },
        };

        var sharpened: Image(Rgb) = .empty;
        try img.convolve(allocator, sharpen_kernel, &sharpened, .replicate);
        defer sharpened.deinit(allocator);
        try sharpened.save(allocator, "custom_sharpen.png");
        std.debug.print("Saved custom_sharpen.png\n", .{});
    }

    // Test edge enhancement kernel
    {
        const edge_enhance = [3][3]f32{
            .{ -1, -1, -1 },
            .{ -1, 9, -1 },
            .{ -1, -1, -1 },
        };

        var enhanced: Image(Rgb) = .empty;
        try img.convolve(allocator, edge_enhance, &enhanced, .mirror);
        defer enhanced.deinit(allocator);
        try enhanced.save(allocator, "edge_enhanced.png");
        std.debug.print("Saved edge_enhanced.png\n", .{});
    }

    // Test Difference of Gaussians (DoG) for edge detection
    {
        var dog_edges: Image(Rgb) = .empty;
        try img.differenceOfGaussians(allocator, 1.0, 1.6, &dog_edges);
        defer dog_edges.deinit(allocator);
        normalizeForVisualization(dog_edges); // Normalize for better visualization
        try dog_edges.save(allocator, "dog_edges_1.0_1.6.png");
        std.debug.print("Saved dog_edges_1.0_1.6.png (DoG edge detection - normalized)\n", .{});
    }

    // Test DoG with different sigma ratios for various frequency bands
    {
        var dog_fine: Image(Rgb) = .empty;
        try img.differenceOfGaussians(allocator, 0.5, 1.0, &dog_fine);
        defer dog_fine.deinit(allocator);
        normalizeForVisualization(dog_fine); // Normalize for better visualization
        try dog_fine.save(allocator, "dog_fine_details.png");
        std.debug.print("Saved dog_fine_details.png (fine details - normalized)\n", .{});

        var dog_medium: Image(Rgb) = .empty;
        try img.differenceOfGaussians(allocator, 2.0, 4.0, &dog_medium);
        defer dog_medium.deinit(allocator);
        normalizeForVisualization(dog_medium); // Normalize for better visualization
        try dog_medium.save(allocator, "dog_medium_features.png");
        std.debug.print("Saved dog_medium_features.png (medium features - normalized)\n", .{});
    }

    std.debug.print("All convolution demos completed successfully!\n", .{});
}
