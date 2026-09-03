#!/usr/bin/env bash
# Generates BMP, PNG, JPEG, and GIF fixtures with ImageMagick that the
# `imagemagick_compat` example loads back and verifies against
# ImageMagick's own decode.
#
# For every generated file foo.{bmp,png,jpg,gif} we also write a foo.ref.png
# that is ImageMagick's own decode of the same bytes. The Zig example loads
# the fixture with zignal, saves a PNG, and checks PSNR against the .ref.png.
#
# The example also uses src_rgb.png / src_rgba.png (the synthetic sources)
# for encoder round-trip tests, so we leave them in place after generation.
#
# Usage:
#   examples/scripts/generate_imagemagick_fixtures.sh [output_dir]
# Default output_dir is examples/fixtures/imagemagick.

set -euo pipefail

if ! command -v magick >/dev/null 2>&1; then
    echo "error: ImageMagick ('magick') not found in PATH" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="${1:-${script_dir}/../fixtures/imagemagick}"
mkdir -p "$out_dir"

# Source: a small synthetic image rich enough to exercise palette quantization
# and color subsampling. We generate two source PNGs: one opaque RGB, one with
# alpha. ImageMagick uses these as the input to all subsequent conversions,
# and the encoder round-trip tests in the Zig example load them too.
src_rgb="${out_dir}/src_rgb.png"
src_rgba="${out_dir}/src_rgba.png"

# 128x96 RGB plasma + a few solid bars for sharp edges.
magick -size 128x96 plasma:fractal -colorspace sRGB -depth 8 PNG24:"$src_rgb"
magick "$src_rgb" \
    -fill 'rgb(255,0,0)'   -draw 'rectangle 0,0   31,15'  \
    -fill 'rgb(0,255,0)'   -draw 'rectangle 32,0  63,15'  \
    -fill 'rgb(0,0,255)'   -draw 'rectangle 64,0  95,15'  \
    -fill 'rgb(0,0,0)'     -draw 'rectangle 96,0  127,15' \
    -fill 'rgb(255,255,255)' -draw 'rectangle 0,80 127,95' \
    PNG24:"$src_rgb"

# RGBA: take the RGB source and punch a transparent disc through the middle.
magick "$src_rgb" \
    \( -size 128x96 xc:none -fill white -draw 'circle 64,48 64,28' \) \
    -compose CopyOpacity -composite \
    -define png:color-type=6 -depth 8 PNG32:"$src_rgba"

# emit_pair <name> <suffix> <imagemagick_args...>
# Writes "$out_dir/$name.$suffix" via ImageMagick from the chosen source, then
# decodes it back with ImageMagick to "$out_dir/$name.ref.png" so the Zig side
# can diff zignal's decode against IM's decode of the same bytes.
emit_pair() {
    local name="$1" suffix="$2" src="$3"
    shift 3
    local fixture="${out_dir}/${name}.${suffix}"
    local reference="${out_dir}/${name}.ref.png"
    magick "$src" "$@" "$fixture"
    # Decode the fixture back through IM. For RGBA-capable formats we keep the
    # alpha channel; otherwise we flatten to opaque so the PSNR check matches
    # what zignal would produce loading into Image(Rgba).
    magick "$fixture" -define png:color-type=6 -depth 8 PNG32:"$reference"
    printf '  %-40s -> %s\n' "$(basename "$fixture")" "$(basename "$reference")"
}

echo "writing fixtures to: $out_dir"

# ---------- BMP variants ----------
echo "BMP fixtures:"
# 24bpp BI_RGB (the canonical case the encoder also produces).
emit_pair bmp_24bpp_rgb       bmp "$src_rgb"  -type TrueColor
# 32bpp BI_BITFIELDS with alpha (BMP4 carries the bitfield masks).
emit_pair bmp_32bpp_rgba      bmp "$src_rgba" -type TrueColorAlpha -define bmp:format=bmp4
# 8bpp indexed (palette).
emit_pair bmp_8bpp_palette    bmp "$src_rgb"  -type Palette -colors 256
# 8bpp indexed grayscale (the form zignal emits for Image(u8)).
emit_pair bmp_8bpp_grayscale  bmp "$src_rgb"  -colorspace Gray -type Grayscale -depth 8
# 4bpp palette.
emit_pair bmp_4bpp_palette    bmp "$src_rgb"  -type Palette -colors 16 -depth 4
# 1bpp monochrome.
emit_pair bmp_1bpp_mono       bmp "$src_rgb"  -monochrome -type Bilevel -depth 1
# RLE8 compressed 8bpp palette (BI_RLE8).
emit_pair bmp_rle8            bmp "$src_rgb"  -type Palette -colors 256 -compress RLE

# ---------- PNG variants ----------
echo "PNG fixtures:"
# 8-bit RGB (color type 2).
emit_pair png_rgb              png "$src_rgb"  -define png:color-type=2 -depth 8
# 8-bit RGBA (color type 6).
emit_pair png_rgba             png "$src_rgba" -define png:color-type=6 -depth 8
# 8-bit grayscale (color type 0).
emit_pair png_grayscale        png "$src_rgb"  -colorspace Gray -define png:color-type=0 -depth 8
# 8-bit gray + alpha (color type 4).
emit_pair png_grayscale_alpha  png "$src_rgba" -colorspace Gray -define png:color-type=4 -depth 8
# 8-bit indexed palette (color type 3).
emit_pair png_palette          png "$src_rgb"  -colors 256 -define png:color-type=3 -depth 8
# 16-bit RGB (high-bit-depth).
emit_pair png_rgb_16           png "$src_rgb"  -define png:color-type=2 -depth 16
# Adam7 interlaced (color type 2).
emit_pair png_interlaced       png "$src_rgb"  -interlace plane -define png:color-type=2 -depth 8

# ---------- JPEG variants ----------
echo "JPEG fixtures:"
# Baseline, no chroma subsampling (4:4:4).
emit_pair jpeg_4_4_4           jpg "$src_rgb"  -sampling-factor 1x1,1x1,1x1 -quality 92
# Baseline, 4:2:2 horizontal subsampling.
emit_pair jpeg_4_2_2           jpg "$src_rgb"  -sampling-factor 2x1,1x1,1x1 -quality 90
# Baseline, 4:2:0 (default for most JPEG encoders).
emit_pair jpeg_4_2_0           jpg "$src_rgb"  -sampling-factor 2x2,1x1,1x1 -quality 85
# Grayscale (single-component) JPEG.
emit_pair jpeg_grayscale       jpg "$src_rgb"  -colorspace Gray -quality 90
# Progressive scan (multiple scans, decoder must accumulate).
emit_pair jpeg_progressive     jpg "$src_rgb"  -interlace plane -sampling-factor 2x2,1x1,1x1 -quality 90

# ---------- GIF variants ----------
echo "GIF fixtures:"
# Static, full 256-color palette.
emit_pair gif_static_256      gif "$src_rgb"  -colors 256
# Static with a small palette to stress the LZW code-size logic.
emit_pair gif_static_16       gif "$src_rgb"  -colors 16
# Static with transparency from the RGBA source.
emit_pair gif_transparent     gif "$src_rgba" -colors 255 -dispose Background
# Interlaced GIF (interlace bit set in the image descriptor).
emit_pair gif_interlaced      gif "$src_rgb"  -colors 256 -interlace GIF

# Animated GIF: 4 frames of the source rotating in 90-degree steps, 50ms each.
anim_in_dir="${out_dir}/_anim_frames"
mkdir -p "$anim_in_dir"
magick "$src_rgb" -rotate 0   "${anim_in_dir}/00.png"
magick "$src_rgb" -rotate 90  "${anim_in_dir}/01.png"
magick "$src_rgb" -rotate 180 "${anim_in_dir}/02.png"
magick "$src_rgb" -rotate 270 "${anim_in_dir}/03.png"
anim_fixture="${out_dir}/gif_animated.gif"
magick -delay 5 -loop 0 "${anim_in_dir}/0?.png" -colors 256 "$anim_fixture"
# For the animated GIF we can't single-PNG-reference it; instead, dump each
# decoded frame so Zig can diff its frames against IM's per-frame decode.
magick "$anim_fixture" -coalesce "${out_dir}/gif_animated.ref-%02d.png"
frame_count=$(ls "${out_dir}/gif_animated.ref-"*.png | wc -l)
printf '  %-40s -> %d frame refs\n' "gif_animated.gif" "$frame_count"

# Keep $src_rgb and $src_rgba — the Zig example reuses them for round-trip
# tests. Drop only the per-frame intermediates that were merged into the
# animated GIF.
rm -rf "$anim_in_dir"

echo "done."
