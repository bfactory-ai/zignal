import os

import pytest

import zignal

SYSTEM_FONTS = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/Roboto-Regular.ttf",
]
SYSTEM_FONT = next((p for p in SYSTEM_FONTS if os.path.exists(p)), None)


def inked(img):
    return sum(1 for row in range(img.rows) for col in range(img.cols) if img[row, col] != 0)


def test_font8x8_metrics_and_draw():
    font = zignal.Font.font8x8()
    assert isinstance(font, zignal.Font)
    assert font is zignal.Font.font8x8()
    assert font.kind == "bitmap"
    assert font.name.startswith("8x8")
    assert font.height == 8
    assert font.line_height(16) == 16
    assert font.ascent(16) > 0
    assert font.has_glyph("A")
    assert font.has_glyph(ord("A"))
    assert font.get_text_bounds("abc", 8).width == 24
    assert font.get_text_bounds("abc", 16).width == 48
    assert 'kind="bitmap"' in repr(font)

    small = zignal.Image(40, 80, 0)
    small.canvas().draw_text("Hi", (5, 5), 255, font)
    assert inked(small) > 0
    big = zignal.Image(40, 80, 0)
    big.canvas().draw_text("Hi", (5, 5), 255, font, size=16)
    assert inked(big) == 4 * inked(small)

    default = zignal.Image(40, 80, 0)
    default.canvas().draw_text("Hi", (5, 5), 255)
    assert default == small


def test_bitmap_font_round_trip(tmp_path):
    font = zignal.Font.font8x8()
    for name in ("f.bdf", "f.pcf.gz"):
        path = str(tmp_path / name)
        font.save(path)
        loaded = zignal.Font.load(path)
        assert loaded.kind == "bitmap"
        assert loaded.height == 8
        assert loaded.get_text_bounds("abc", 8).width == font.get_text_bounds("abc", 8).width


def test_font_errors(tmp_path):
    canvas = zignal.Image(20, 40, 0).canvas()
    with pytest.raises(TypeError):
        canvas.draw_text("Hi", (0, 0), 255, "not a font")
    with pytest.raises(FileNotFoundError):
        zignal.Font.load("/definitely/missing.bdf")
    junk = tmp_path / "junk.ttf"
    junk.write_bytes(b"\x00\x01\x00\x00" + b"\xff" * 64)
    with pytest.raises(ValueError):
        zignal.Font.load(str(junk))
    with pytest.raises(TypeError):
        zignal.Font.font8x8().has_glyph("ab")
    with pytest.raises(TypeError):
        zignal.Font.font8x8().has_glyph(1.5)


@pytest.mark.skipif(SYSTEM_FONT is None, reason="no TrueType font installed")
def test_truetype_font(tmp_path):
    font = zignal.Font.load(SYSTEM_FONT)
    assert font.kind == "vector"
    assert font.name is None
    assert font.height is None
    assert 'kind="vector"' in repr(font)
    assert font.ascent(24) > 0
    assert font.line_height(24) > font.ascent(24)
    assert font.has_glyph("A")
    assert not font.has_glyph("\U0001F600")

    bounds = font.get_text_bounds("Hello", 24)
    assert bounds.width > 24 and bounds.height == pytest.approx(font.line_height(24))
    assert font.get_text_bounds("Hello", 48).width == pytest.approx(2 * bounds.width)
    # Kerning pulls "AV" closer than two "A"s worth of advance.
    assert font.get_text_bounds("AV", 24).width < font.get_text_bounds("A", 24).width + font.get_text_bounds("V", 24).width
    tight = font.get_text_bounds_tight("Hello", 24)
    assert 0 < tight.width <= bounds.width

    small = zignal.Image(60, 200, 0)
    small.canvas().draw_text("Hello", (2, 2), 255, font, size=24, mode=zignal.DrawMode.SOFT)
    big = zignal.Image(60, 200, 0)
    big.canvas().draw_text("Hello", (2, 2), 255, font, size=48, mode=zignal.DrawMode.SOFT)
    assert 0 < inked(small) < inked(big)

    with pytest.raises(ValueError):
        font.save(str(tmp_path / "out.bdf"))
