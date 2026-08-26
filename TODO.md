# Follow-ups

Tracking the items left after the text/rasterizer work of August 2026 (#404–#413).
Check them off as they merge.

## Next

- [ ] **`fillBoxSoft`** — the axis-aligned ring with no inner rectangle, shared by
      `drawAxisAlignedThickLine` and `fillRingSoft`; drop the "inverted inner" generality
      nobody calls.
- [ ] **Join style on `DrawOptions`** (`.round` / `.miter`) — `drawPolygon` gives round
      corners and `drawRectangle` square ones for the same four points. API decision; discuss
      before implementing.

## When needed

- [ ] Glyph cache: `GlyphCache.phases = 8` if the quarter-pixel snap ever shows; LRU instead
      of clear-all eviction; one glyph lookup for metrics + bounds in `Layout.place`.

## Larger font work

- [ ] Variable-font axes (`gvar` for TrueType; CFF2 `blend` beyond the default instance).
- [ ] Shaping: marks, ligatures, complex scripts (GSUB/GPOS beyond pair kerning).
- [ ] Colour/emoji fonts (`COLR`/`CBDT`/`sbix`).

## Done

- [x] **Band-limit `renderRing`** (#414) — two runs per row from the outer and inner radii;
      circles 1.6–6× faster, fast arcs 1.4–2.4×, output identical. The `fillArcRing`
      rotation loop was left out: a sincos per vertex is ~1 µs on a ~45 µs soft arc and
      swapping it would move the arc fixtures for nothing.

- [x] **macOS wheel size** — the ≤ 0.10.0 macOS wheels bundled a 16 MB copy of the Python
      framework library (`zignal/.dylibs/Python`, added by `delocate` because the extension
      linked libpython). #367 switched extension modules to `-undefined dynamic_lookup`;
      wheels built from master are 1.5 MB with nothing bundled. Fixed by the next release.

- [x] #409 glyph cache, #410 text blocks lay out once, #411 halo through `blitMask`,
      #412 one edge sweep for the fills, #413 TestPyPI on releases/dispatch only.
- [x] Dropped: WOFF1/WOFF2 support.
