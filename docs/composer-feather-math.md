# Composer border fade — feather + edge blur (single source of truth)

Per-layer "Feather" and "Edge Blur" sliders fade a layer's rectangular border
into the background so a hard cut-out edge does not survive compositing.
Everything runs in **display space**: post scale/flip, pre rotation/paste.
A rotation preserves distance, so a rotated layer keeps a uniform band.
Alpha values are 0..255 uint8; RGB is sRGB.

## Slider mapping (quadratic, same pattern as the layer blur)
```
feather:   F = clamp(v,0,100)/100; F = F*F * 200 px     (FEATHER_MAX_PX = 200)
edge blur: R = clamp(v,0,100)/100; R = R*R * 50 px      (EDGE_BLUR_MAX_PX = 50)
```
Unreadable values (undefined/null/NaN) read as 0 (no fade). Layer fields:
`feather`, `edgeBlur` (0-100, default 0).

## Distance + falloff
```
d = min(x, w-1-x, y, h-1-y)                       // px to nearest border
s(t) = t*t*(3-2*t)   for t = clamp(d/r, 0, 1)     // smoothstep
r <= 0  ->  s = 1                                   // no radius, no fade
```

## Treatment (order is fixed)
1. **Edge blur (RGB only, alpha untouched)** — blend a precomputed gaussian of
   the RGB channels (sigma = R, edge-clamped like PIL `GaussianBlur`) in over
   the last R px of the border:
   ```
   m = s(d/R)
   rgb = m*sharp + (1-m)*gauss(rgb, sigma=R)
   ```
2. **Feather (alpha only)** — fade the existing alpha to 0 over the last F px:
   ```
   alpha = alpha * s(d/F)
   ```
Both `F = R = 0` -> exact identity (buffer copied, no math).

## Pipeline order
crop -> eraser mask -> power pin -> display resize/flip -> **edge blur ->
feather** -> rotate + paste onto doc -> existing per-layer Gaussian blur
(`ctx.filter` / document-sized `GaussianBlur`) -> blend mode.

## Rounding + parity
Results are stored through `np.rint` (Python) / `Uint8ClampedArray` store
(JS) — both round half-to-even. Cross-language tolerance: **<= 1 LSB per
channel**. The JS edge-blur kernel is `ctx.filter = blur(Rpx)` over a canvas
padded with the border replicated `ceil(2R)+2` px (PIL edge-clamp
approximation); kernel reach beyond 2R can shift the outermost 1-2 px by a
tiny amount — negligible because the feather alpha there is ~0. Python
clamps exactly; treat that 1-2 px as the documented tolerance class.

## Pinned layers
The treatment band is the AABB border of the warped (pin-quad) display
frame — never the pin quad itself (out of scope, see the power-pin spec:
"No pin feathering/masking").

## Implementations (keep in lockstep)
- `nodes/node_composition.py`: `_feather_radius_px`, `_edge_blur_radius_px`,
  `_border_profile`, `_border_fade_arr`, `_apply_border_fade`,
  `_border_fade_pre`, `_apply_layer_transform(..., pre_display=...)`.
- `js/composer/feather.mjs`: `featherRadiusPx`, `edgeBlurRadiusPx`,
  `borderProfile`, `applyBorderFade` (pure, node-testable), `edgeBlurRGB`,
  `borderFadeCanvas` (canvas plumbing).
- Render sites: `js/composer/render.mjs` `_drawImpl` (cached treated canvas,
  <=512 px two-tier while slider input is <150 ms apart —
  `_borderFadeCanvas` / `_lastFadeInput`), `js/composer/index.js`
  `rebuildPreview` drawLayer (one-shot, no cache).
- UI: `js/composer/ui.mjs` Feather / Edge Blur rows in Transform Properties
  (wired via `syncSliderTrans` in `interaction.mjs`, history on release).
- Save/restore: `layerEntry.feather` / `layerEntry.edgeBlur` in
  `interaction.mjs` (only written when > 0); restore sites in `render.mjs`.

## Tests
- `tests/test_feather.py` (python, incl. cross-language parity via node).
- `tests/test_feather_math.mjs` (node --test, pure math).
