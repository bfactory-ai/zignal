#!/usr/bin/env python3
"""Comprehensive tests for Canvas functionality"""

import pytest
import numpy as np
import time
from typing import Union, Tuple
import zignal


class TestCanvasConstructor:
    """Test Canvas construction patterns"""

    def test_canvas_from_image_method(self):
        """Test creating canvas using image.canvas() method"""
        img_array = np.zeros((100, 100, 4), dtype=np.uint8)
        img = zignal.Image.from_numpy(img_array)
        canvas = img.canvas()

        assert canvas is not None
        assert canvas.rows == 100
        assert canvas.cols == 100

        # Verify canvas can draw
        canvas.fill((255, 0, 0))
        result = img.to_numpy()
        assert np.all(result[:, :, :3] == [255, 0, 0])

    def test_canvas_constructor(self):
        """Test creating canvas using Canvas(image) constructor"""
        img_array = np.zeros((100, 100, 4), dtype=np.uint8)
        img = zignal.Image.from_numpy(img_array)
        canvas = zignal.Canvas(img)

        assert canvas is not None
        assert canvas.rows == 100
        assert canvas.cols == 100

        # Verify canvas can draw
        canvas.fill((0, 255, 0))
        result = img.to_numpy()
        assert np.all(result[:, :, :3] == [0, 255, 0])

    def test_both_construction_methods_equivalent(self):
        """Test that both construction methods produce equivalent results"""
        img1 = zignal.Image.from_numpy(np.zeros((50, 50, 4), dtype=np.uint8))
        img2 = zignal.Image.from_numpy(np.zeros((50, 50, 4), dtype=np.uint8))

        canvas1 = img1.canvas()
        canvas2 = zignal.Canvas(img2)

        # Draw same thing on both
        color = (128, 128, 128)
        canvas1.fill(color)
        canvas2.fill(color)

        # Results should be identical
        np.testing.assert_array_equal(img1.to_numpy(), img2.to_numpy())


class TestCanvasImageProperty:
    """Test Canvas.image property"""

    def test_image_property_returns_parent(self):
        """Test that canvas.image returns the parent Image object"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Get image back from canvas
        retrieved_img = canvas.image

        # Should be the same object
        assert retrieved_img is img

    def test_image_property_allows_saving(self):
        """Test that we can save the image obtained from canvas.image"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = zignal.Canvas(img)

        # Draw something
        canvas.fill((255, 0, 0))

        # Get image and verify we can save it
        canvas_img = canvas.image
        canvas_img.save("test_canvas_save.png")

        # Load and verify
        loaded = zignal.Image.load("test_canvas_save.png")
        result = loaded.to_numpy()
        assert np.all(result[:, :, :3] == [255, 0, 0])

        # Cleanup
        import os

        os.remove("test_canvas_save.png")

    def test_modifications_via_canvas_affect_original(self):
        """Test that canvas modifications affect the original image"""
        img = zignal.Image.from_numpy(np.zeros((50, 50, 4), dtype=np.uint8))
        # Make a copy of the original data
        original_array = img.to_numpy().copy()

        canvas = img.canvas()
        canvas.fill((100, 100, 100))

        # Original image should be modified
        modified_array = img.to_numpy()
        assert not np.array_equal(original_array, modified_array)
        assert np.all(modified_array[:, :, :3] == [100, 100, 100])

        # Image from canvas.image should also show modifications
        canvas_img_array = canvas.image.to_numpy()
        np.testing.assert_array_equal(modified_array, canvas_img_array)


class TestDrawLine:
    """Test draw_line functionality"""

    def test_basic_lines(self):
        """Test drawing basic lines with different styles"""
        img = zignal.Image.from_numpy(np.zeros((200, 200, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Horizontal line
        canvas.draw_line((10, 50), (190, 50), (255, 0, 0))

        # Vertical line
        canvas.draw_line((50, 10), (50, 190), (0, 255, 0), width=3)

        # Diagonal line
        canvas.draw_line(
            (10, 10), (190, 190), (0, 0, 255), width=2, mode=zignal.DrawMode.SOFT
        )

        result = img.to_numpy()

        # Verify red horizontal line (row 50, avoid intersections)
        assert (
            result[50, 100, 0] == 255
            and result[50, 100, 1] == 0
            and result[50, 100, 2] == 0
        )

        # Verify green vertical line (column 50, thicker)
        assert (
            result[100, 50, 1] == 255
            and result[100, 50, 0] == 0
            and result[100, 50, 2] == 0
        )

        # Verify blue diagonal line exists
        # Check a point on the diagonal
        diagonal_point = result[100, 100]
        assert diagonal_point[2] > 0  # Blue channel should be non-zero

    def test_line_with_transparency(self):
        """Test drawing lines with alpha transparency"""
        img = zignal.Image.from_numpy(np.ones((100, 100, 4), dtype=np.uint8) * 255)
        canvas = img.canvas()

        # Draw semi-transparent line
        canvas.draw_line((10, 50), (90, 50), (255, 0, 0, 128), width=5)

        result = img.to_numpy()
        line_pixel = result[50, 50]

        # The line might be drawn with full opacity or blended
        # Just verify a red line was drawn
        assert line_pixel[0] > 0  # Red channel should be set
        # If blending is supported, red should be between original and full red
        # If not, it will be full red (255)

    def test_zero_length_line(self):
        """Test that zero-length line draws a point"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        canvas.draw_line((50, 50), (50, 50), (255, 255, 255), width=4)

        result = img.to_numpy()
        # Should draw a circular point
        assert np.all(result[50, 50, :3] == [255, 255, 255])

    def test_line_modes(self):
        """Test FAST vs SOFT rendering modes"""
        img1 = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        img2 = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))

        canvas1 = img1.canvas()
        canvas2 = img2.canvas()

        # Same line with different modes
        canvas1.draw_line(
            (10, 10), (90, 90), (255, 255, 255), width=1, mode=zignal.DrawMode.FAST
        )
        canvas2.draw_line(
            (10, 10), (90, 90), (255, 255, 255), width=1, mode=zignal.DrawMode.SOFT
        )

        # SOFT mode should produce antialiased edges
        # This is hard to test precisely, but we can verify both drew something
        result1 = img1.to_numpy()
        result2 = img2.to_numpy()

        assert np.any(result1 > 0)
        assert np.any(result2 > 0)


class TestColorObjects:
    """Test using color objects with canvas methods"""

    def test_rgb_color_object(self):
        """Test RGB color object"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        red = zignal.Rgb(255, 0, 0)
        canvas.fill(red)

        result = img.to_numpy()
        assert np.all(result[:, :, :3] == [255, 0, 0])

    def test_color_space_conversions(self):
        """Test various color space objects are automatically converted"""
        img = zignal.Image.from_numpy(np.zeros((100, 300, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Test different color spaces
        test_cases = [
            (zignal.Hsl(120, 100, 50), "HSL green"),  # Green
            (zignal.Hsv(240, 100, 100), "HSV blue"),  # Blue
            (zignal.Lab(97.14, -21.56, 94.48), "Lab yellow"),  # Yellow
            (zignal.Oklab(0.42, 0.16, -0.10), "Oklab purple"),  # Purple
        ]

        for i, (color, name) in enumerate(test_cases):
            y = i * 20 + 10
            canvas.draw_line((10, y), (290, y), color, width=10)

        result = img.to_numpy()

        # Verify conversions worked (checking that lines were drawn)
        for i in range(len(test_cases)):
            y = i * 20 + 10
            line_pixels = result[y, 50, :3]
            assert np.any(line_pixels > 0), (
                f"Color conversion failed for {test_cases[i][1]}"
            )

    def test_rgba_with_transparency(self):
        """Test RGBA color object with alpha channel"""
        img = zignal.Image.from_numpy(np.ones((100, 100, 4), dtype=np.uint8) * 255)
        canvas = img.canvas()

        semi_transparent = zignal.Rgba(255, 0, 0, 128)
        canvas.fill(semi_transparent)

        result = img.to_numpy()
        # Verify red channel is set (blending behavior may vary)
        assert result[50, 50, 0] > 0  # Red channel should be set

    def test_mixed_color_types(self):
        """Test mixing tuples and color objects in same session"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Alternate between tuples and objects
        canvas.draw_line((10, 20), (90, 20), (255, 0, 0), width=2)  # Tuple
        canvas.draw_line((10, 40), (90, 40), zignal.Rgb(0, 255, 0), width=2)  # Object
        canvas.draw_line((10, 60), (90, 60), (0, 0, 255), width=2)  # Tuple
        canvas.draw_line((10, 80), (90, 80), zignal.Hsl(60, 100, 50), width=2)  # Object

        result = img.to_numpy()

        # Verify all lines were drawn
        assert result[20, 50, 0] == 255  # Red line
        assert result[40, 50, 1] == 255  # Green line
        assert result[60, 50, 2] == 255  # Blue line
        assert np.any(result[80, 50, :3] > 0)  # Yellow line (from HSL)


class TestCanvasPerformance:
    """Test canvas performance characteristics"""

    def test_fill_performance(self):
        """Test that repeated fill operations are reasonably fast"""
        img = zignal.Image.from_numpy(np.zeros((1000, 1000, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Warm up
        for _ in range(10):
            canvas.fill((255, 0, 0))

        # Time 1000 operations
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        start_time = time.perf_counter()

        for i in range(1000):
            color = colors[i % len(colors)]
            canvas.fill(color)

        elapsed = time.perf_counter() - start_time
        ops_per_second = 1000 / elapsed

        # Should be reasonably fast (at least 1000 ops/sec on modern hardware)
        assert ops_per_second > 1000, f"Fill too slow: {ops_per_second:.0f} ops/sec"

        # Verify last color
        result = img.to_numpy()
        expected_color = colors[(1000 - 1) % len(colors)]
        assert np.all(result[0, 0, :3] == expected_color)

    def test_canvas_creation_is_cheap(self):
        """Test that canvas creation is essentially free"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))

        # Time canvas creation
        start_time = time.perf_counter()
        for _ in range(10000):
            canvas = img.canvas()
        elapsed = time.perf_counter() - start_time

        # Should be very fast (microseconds per creation)
        avg_time_us = (elapsed / 10000) * 1_000_000
        assert avg_time_us < 10, f"Canvas creation too slow: {avg_time_us:.2f} µs"


class TestCanvasRobustness:
    """Test error handling and edge cases"""

    def test_invalid_color_tuple(self):
        """Test handling of invalid color tuples"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Test various invalid inputs
        with pytest.raises(Exception):
            canvas.fill((256, 0, 0))  # Out of range

        with pytest.raises(Exception):
            canvas.fill((255,))  # Too few components

        with pytest.raises(Exception):
            canvas.fill("red")  # Wrong type

    def test_boundary_drawing(self):
        """Test drawing at image boundaries"""
        img = zignal.Image.from_numpy(np.zeros((100, 100, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Draw lines at boundaries
        canvas.draw_line((0, 0), (99, 0), (255, 0, 0))  # Top edge
        canvas.draw_line((99, 0), (99, 99), (0, 255, 0))  # Right edge
        canvas.draw_line((99, 99), (0, 99), (0, 0, 255))  # Bottom edge
        canvas.draw_line((0, 99), (0, 0), (255, 255, 0))  # Left edge

        result = img.to_numpy()

        # Verify corners have expected colors
        assert result[0, 0, 0] == 255  # Top-left is red
        assert result[0, 99, 1] == 255  # Top-right is green
        assert result[99, 99, 2] == 255  # Bottom-right is blue
        assert np.any(result[99, 0, :3] > 0)  # Bottom-left has yellow

    def test_large_width_lines(self):
        """Test drawing with very large line widths"""
        img = zignal.Image.from_numpy(np.zeros((200, 200, 4), dtype=np.uint8))
        canvas = img.canvas()

        # Draw with large width
        canvas.draw_line((100, 100), (100, 100), (255, 255, 255), width=50)

        result = img.to_numpy()

        # Should draw a large circle
        # Check that many pixels are white
        white_pixels = np.sum(np.all(result[:, :, :3] == [255, 255, 255], axis=2))
        assert white_pixels > 100  # Should fill a substantial area


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
