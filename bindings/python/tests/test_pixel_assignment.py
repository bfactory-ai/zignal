import zignal


class TestPixelAssignment:
    def test_assign_any_colorspace_to_rgb_image(self):
        img = zignal.Image(2, 2, dtype=zignal.Rgb)

        gray = zignal.Gray(128)
        img[0, 0] = gray
        px00 = img[0, 0].item()
        assert (px00.r, px00.g, px00.b) == (128, 128, 128)

        hsl = zignal.Hsl(0.0, 1.0, 0.5)  # red
        img[0, 1] = hsl
        expected = hsl.to(zignal.Rgb)
        px01 = img[0, 1].item()
        assert (px01.r, px01.g, px01.b) == (expected.r, expected.g, expected.b)

        rgba = zignal.Rgba(1, 2, 3, 4)
        img[1, 0] = rgba
        px10 = img[1, 0].item()
        assert (px10.r, px10.g, px10.b) == (1, 2, 3)

    def test_assign_any_colorspace_to_gray_image(self):
        img = zignal.Image(2, 2, dtype=zignal.Gray)

        rgb = zignal.Rgb(255, 255, 255)
        img[0, 0] = rgb
        assert img[0, 0] == rgb.to(zignal.Gray).y

        hsl = zignal.Hsl(0.33, 1.0, 0.5)
        img[0, 1] = hsl
        assert img[0, 1] == hsl.to(zignal.Gray).y
